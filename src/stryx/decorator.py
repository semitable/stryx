"""Decorator-based CLI for Stryx.

The @stryx.cli decorator transforms a function into a full CLI:

    @stryx.cli(schema=Config)
    def main(cfg: Config):
        train(cfg)

    if __name__ == "__main__":
        main()

Commands:
    train.py                                 Run with schema defaults
    train.py lr=1e-4 batch=32                Run with overrides
    train.py new [name] [overrides...]       Save recipe (auto-names if no name)
    train.py new [name] --from <src> [ov]    Copy + modify recipe
    train.py run <name|path> [overrides...]  Run from recipe or file path
    train.py edit <name>                     Edit recipe in TUI
    train.py show [name|path] [overrides...] Show config with sources
    train.py list                            List all recipes
"""
from __future__ import annotations

import functools
import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal, TypeVar, Union

from pydantic import BaseModel, ValidationError

from .cli_parser import ParsedArgs, parse_argv
from .commands import cmd_edit, cmd_fork, cmd_list, cmd_schema, cmd_show, cmd_try
from .config_builder import (
    apply_override,
    build_config,
    load_and_override,
    parse_value,
    read_config_file,
    validate_or_die,
)
from .run_id import derive_run_id, parse_run_id_options
from .utils import read_yaml, set_dotpath, write_yaml

T = TypeVar("T", bound=BaseModel)




def cli(
    schema: type[T] | None = None,
    recipes_dir: str | Path = "configs",
    runs_dir: str | Path = "runs",
) -> Callable[[Callable[[T], Any]], Callable[[], Any]]:
    """Decorator that adds Stryx CLI to a function.

    Args:
        schema: Pydantic model class for config. If None, inferred from type hints.
        recipes_dir: Directory for storing recipes (default: configs)
        runs_dir: Directory for storing run manifests (default: runs)

    Example:
        @stryx.cli(schema=Config)
        def main(cfg: Config):
            train(cfg)

        # Or with type hint inference:
        @stryx.cli()
        def main(cfg: Config):
            train(cfg)
    """
    recipes_path = Path(recipes_dir)
    runs_path = Path(runs_dir)

    def decorator(func: Callable[[T], Any]) -> Callable[[], Any]:
        nonlocal schema

        # Infer schema from type hints if not provided
        if schema is None:
            inferred = _infer_schema(func)
            if inferred is None:
                raise TypeError(
                    f"Could not infer schema for {func.__name__}. "
                    "Either provide schema= argument or add type hint like: def main(cfg: Config)"
                )
            resolved_schema = inferred
        else:
            resolved_schema = schema

        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # If called with a config directly, just run the function
            if args and isinstance(args[0], resolved_schema):
                return func(*args, **kwargs)
            if kwargs:
                return func(*args, **kwargs)

            # Otherwise, parse CLI and dispatch
            argv = sys.argv[1:]
            return _dispatch(func, resolved_schema, recipes_path, runs_path, argv)

        # Attach metadata for introspection
        wrapper._stryx_schema = resolved_schema  # type: ignore
        wrapper._stryx_recipes_dir = recipes_path  # type: ignore

        return wrapper

    return decorator


def _infer_schema(func: Callable) -> type[T] | None:
    """Infer schema type from function's type hints."""
    hints = getattr(func, "__annotations__", {})
    for param_type in hints.values():
        if isinstance(param_type, type) and issubclass(param_type, BaseModel):
            return param_type
    return None


def _dispatch(
    func: Callable[[T], Any],
    schema: type[T],
    recipes_dir: Path,
    runs_dir: Path,
    argv: list[str],
) -> Any:
    """Parse argv and dispatch to appropriate handler."""
    args = parse_argv(argv)
    effective_runs_dir = Path(args.run_dir_override or os.getenv("STRYX_RUN_DIR") or runs_dir)
    effective_recipes_dir = Path(args.recipes_dir_override or os.getenv("STRYX_CONFIGS_DIR") or recipes_dir)

    if args.command == "create-run-id":
        if _wants_help(argv):
            _print_run_id_help(Path(sys.argv[0]).name)
            return
        run_id = derive_run_id(
            label=Path(sys.argv[0]).stem,
            run_id_override=args.run_id_override,
        )
        print(run_id)
        return

    if args.command == "help":
        _print_help(schema)
        return

    if args.command == "list":
        return cmd_list(effective_recipes_dir)

    if args.command == "fork":
        return cmd_fork(schema, effective_recipes_dir, args)

    if args.command == "edit":
        return cmd_edit(schema, effective_recipes_dir, args.recipe)

    if args.command == "show":
        return cmd_show(schema, effective_recipes_dir, args)

    if args.command == "schema":
        return cmd_schema(schema)

    target_path: Path | None = None

    if args.command == "try":
        target_path = cmd_try(schema, effective_recipes_dir, args)

    elif args.command == "run":
        if args.overrides:
            raise SystemExit(
                "Error: 'run' is strict and does not accept overrides.\n"
                "Use 'try' to experiment: stryx try <recipe> [overrides...]\n"
                "Use 'fork' to save changes: stryx fork <recipe> <new_name> [overrides...]"
            )

        if args.config_path:
            target_path = args.config_path
        elif args.recipe:
            name = args.recipe
            if not name.endswith(".yaml"):
                name += ".yaml"

            # Check canonical
            p = effective_recipes_dir / name
            if p.exists():
                target_path = p
            else:
                # Check scratch
                p = effective_recipes_dir / "scratches" / name
                if p.exists():
                    target_path = p
                else:
                    raise SystemExit(f"Recipe not found: {args.recipe}")
        else:
            raise SystemExit("Error: 'run' requires a recipe name.")

    # Load and run
    if not target_path or not target_path.exists():
        raise SystemExit(f"Config file not found: {target_path}")

    # We load from file without further overrides
    cfg = load_and_override(schema, target_path, [])

    _record_run_manifest(
        schema=schema,
        cfg=cfg,
        runs_dir=effective_runs_dir,
        source={"kind": "file", "path": str(target_path), "name": target_path.stem},
        overrides=args.overrides if args.command == "try" else [],
        run_id_override=args.run_id_override,
    )

    return func(cfg)





# =============================================================================
# Run Manifest
# =============================================================================




def _is_rank_zero() -> bool:
    """Check if the current process is rank 0 (leader) in a distributed run.

    Checks common environment variables from torchrun, SLURM, MPI, etc.
    """
    # Standard rank variables
    for env_var in ("RANK", "LOCAL_RANK", "SLURM_PROCID", "OMPI_COMM_WORLD_RANK"):
        val = os.getenv(env_var)
        if val is not None:
            return int(val) == 0

    # Default to True if no distributed env detected
    return True


def _record_run_manifest(
    schema: type[BaseModel],
    cfg: BaseModel,
    runs_dir: Path,
    source: dict[str, Any],
    overrides: list[str],
    run_id_override: str | None,
) -> None:
    """Write a per-run manifest with resolved config and metadata."""
    if not _is_rank_zero():
        return

    run_id = derive_run_id(
        label=source.get("name") or source.get("path"),
        run_id_override=run_id_override,
    )
    run_root = runs_dir / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    manifest_path = run_root / "manifest.yaml"
    resolved_path = run_root / "config.yaml"

    patch_path = _write_git_patch(run_root)

    # Write resolved config to its own file for reproducibility
    try:
        write_yaml(resolved_path, cfg.model_dump(mode="python"))
    except Exception as exc:  # pragma: no cover - avoid failing user runs on write issues
        print(
            f"Warning: failed to write resolved config to {resolved_path}: {exc}",
            file=sys.stderr,
        )

    manifest = {
        "run_id": run_id,
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
        "schema": f"{schema.__module__}:{schema.__name__}",
        "config_source": source,
        "overrides": overrides or [],
        "config": cfg.model_dump(mode="python"),
        "resolved_config_path": str(resolved_path),
        "git": _git_info(),
        "python": {"version": _python_version()},
        "uv": {"lock_hash": _uv_lock_hash()},
    }
    manifest["git"]["untracked"] = _git_untracked_files()
    if patch_path:
        manifest["git"]["patch_file"] = str(patch_path)

    try:
        write_yaml(manifest_path, manifest)
    except Exception as exc:  # pragma: no cover - avoid failing user runs on manifest write issues
        print(
            f"Warning: failed to write run manifest to {manifest_path}: {exc}",
            file=sys.stderr,
        )


def _wants_help(argv: list[str]) -> bool:
    """Return True if help flags are present in argv (excluding program name)."""
    return any(tok in ("--help", "-h") for tok in argv)


def _run_cmd(cmd: list[str], timeout: float = 2.0) -> str | None:
    """Run a subprocess command and return stdout, or None on failure."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def _git_info() -> dict[str, Any]:
    """Return git SHA and dirty flag, if available."""
    sha = _run_cmd(["git", "rev-parse", "HEAD"])
    dirty: bool | None = None

    if sha:
        status = _run_cmd(["git", "status", "--porcelain"])
        dirty = bool(status)

    return {"sha": sha, "dirty": dirty}


def _git_untracked_files() -> list[str]:
    """Return a list of untracked file paths, relative to repo root."""
    if not _run_cmd(["git", "rev-parse", "--is-inside-work-tree"]):
        return []

    output = _run_cmd(["git", "ls-files", "--others", "--exclude-standard"])
    if not output:
        return []
    return [line for line in output.splitlines() if line.strip()]


def _uv_lock_hash(lock_path: Path | None = None) -> str | None:
    """Compute a hash of uv.lock to capture dependency state."""
    path = lock_path or Path("uv.lock")
    if not path.exists():
        return None

    hasher = hashlib.sha256()
    try:
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    except OSError:
        return None


def _write_git_patch(run_root: Path) -> Path | None:
    """Write a git patch of the working tree to the run directory if dirty."""
    # Check if we are in a git repo
    if not _run_cmd(["git", "rev-parse", "--is-inside-work-tree"]):
        return None

    patch = _run_cmd(["git", "diff", "--patch", "HEAD"])
    if not patch:
        return None

    out_path = run_root / "git.patch"
    try:
        out_path.write_text(patch + "\n", encoding="utf-8")
        return out_path
    except OSError:
        return None


def _python_version() -> str:
    """Return short Python version string."""
    return platform.python_version()





# =============================================================================
# Help
# =============================================================================




def _print_help(schema: type[BaseModel]) -> None:
    """Print CLI help message."""
    prog = Path(sys.argv[0]).name
    print(f"""Stryx - Typed Configuration Compiler

Usage:
  {prog}                                 Run with defaults
  {prog} <key>=<value> ...               Run with overrides
  {prog} new [name] [overrides...]       Create recipe (auto-names if no name)
  {prog} new [name] --from <src> [ov]    Copy + modify recipe
  {prog} run <name|path> [overrides...]  Run from recipe or file
  {prog} edit <name>                     Edit recipe (TUI)
  {prog} show [name|path] [overrides...] Show config with sources
  {prog} list                            List all recipes
  {prog} schema                          Show configuration schema
  {prog} create-run-id [options]         Print a generated run id

Run ID options:
  --run-id <id>              Use an explicit run id (conflicts with STRYX_RUN_ID)

Directories:
  --configs-dir <path>       Override recipes directory (or STRYX_CONFIGS_DIR)
  --run-dir <path>           Override runs directory (or STRYX_RUN_DIR)

Examples:
  {prog} lr=1e-4 train.steps=1000
  {prog} new my_exp lr=1e-4
  {prog} run my_exp
  {prog} show my_exp lr=1e-5             # See where values come from

Schema: {schema.__module__}:{schema.__name__} (use '{prog} schema' to inspect)
""")


def _print_run_id_help(prog: str) -> None:
    """Print help for run-id generation."""
    print(
        f"""Generate a run id and exit.

Usage:
  {prog} create-run-id [options]

Options:
  --run-id <id>              Use an explicit run id (conflicts with STRYX_RUN_ID)
"""
    )
