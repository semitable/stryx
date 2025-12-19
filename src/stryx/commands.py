import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from stryx.config_builder import build_config, load_and_override
from stryx.context import Ctx
from stryx.lifecycle import RunContext, get_rank, record_run_manifest
from stryx.run_id import derive_run_id
from stryx.utils import (
    get_next_sequential_name,
    resolve_recipe_path,
    flatten_config,
    save_recipe,
)


def cmd_new(ctx: Ctx, ns: argparse.Namespace) -> Path:
    """Handle: new [name] [overrides...] - create from defaults."""
    from filelock import FileLock

    cfg = build_config(ctx.schema, ns.overrides)
    cfg_data = cfg.model_dump(mode="python")

    # Create directory
    ctx.configs_dir.mkdir(parents=True, exist_ok=True)

    try:
        if ns.recipe:
            name = ns.recipe
            if "." not in name:
                name = f"{name}.yaml"
            out_path = ctx.configs_dir / name

            save_recipe(
                path=out_path,
                cfg_data=cfg_data,
                schema_cls=ctx.schema,
                overrides=ns.overrides,
                description=ns.message,
                force=ns.force,
                kind="canonical",
            )
        else:
            # Auto-generate name with lock
            lock_path = ctx.configs_dir / ".stryx.lock"
            with FileLock(lock_path):
                name = get_next_sequential_name(ctx.configs_dir)
                out_path = ctx.configs_dir / f"{name}.yaml"

                save_recipe(
                    path=out_path,
                    cfg_data=cfg_data,
                    schema_cls=ctx.schema,
                    overrides=ns.overrides,
                    description=ns.message,
                    force=False,
                    kind="canonical",
                )

    except FileExistsError as e:
        raise SystemExit(f"Error: {e} Use --force to overwrite.")

    print(f"Created recipe: {out_path}")
    return out_path


def cmd_fork(ctx: Ctx, ns: argparse.Namespace) -> Path:
    """Handle: fork <source> <name> [overrides...]"""

    # Resolve and Load Source
    try:
        from_path = resolve_recipe_path(ctx.configs_dir, ns.source)
    except FileNotFoundError:
        raise SystemExit(f"Source recipe not found: {ns.source}")

    cfg = load_and_override(ctx.schema, from_path, ns.overrides)
    cfg_data = cfg.model_dump(mode="python")

    # Determine output path
    name = ns.name
    if "." not in name:
        name = f"{name}.yaml"
    out_path = ctx.configs_dir / name

    ctx.configs_dir.mkdir(parents=True, exist_ok=True)

    try:
        save_recipe(
            path=out_path,
            cfg_data=cfg_data,
            schema_cls=ctx.schema,
            overrides=ns.overrides,
            description=ns.message,
            force=ns.force,
            kind="canonical",
            source=str(ns.source),
        )
    except FileExistsError as e:
        raise SystemExit(f"Error: {e} Use --force to overwrite.")

    print(f"Forked recipe: {out_path}")
    return out_path


def cmd_run(ctx: Ctx, ns: argparse.Namespace) -> Any:
    """Handle: run <target> - run a recipe exactly."""
    try:
        path = resolve_recipe_path(ctx.configs_dir, ns.target)
    except FileNotFoundError:
        raise SystemExit(f"Recipe not found: {ns.target}")

    # Load config (run is strict, no overrides)
    cfg = load_and_override(ctx.schema, path, [])

    return _execute(
        ctx,
        cfg,
        source={"kind": "file", "path": str(path), "name": path.stem},
        overrides=[],
        run_id_override=getattr(ns, "run_id", None),
    )


def cmd_try(ctx: Ctx, ns: argparse.Namespace) -> Any:
    """Handle: try [target] [overrides...] - run experimental variant."""
    import petname

    recipe_token = ns.target
    overrides = ns.overrides

    # If recipe token looks like an override (contains '='), shift it
    if recipe_token and "=" in recipe_token:
        overrides = [recipe_token] + overrides
        recipe_token = None

    # Resolve source
    if recipe_token:
        try:
            from_path = resolve_recipe_path(ctx.configs_dir, recipe_token)
            cfg = load_and_override(ctx.schema, from_path, overrides)
            lineage = recipe_token
            name_label = from_path.stem
        except FileNotFoundError:
            raise SystemExit(f"Source recipe not found: {recipe_token}")
    else:
        cfg = build_config(ctx.schema, overrides)
        lineage = None
        name_label = "defaults"

    # Generate scratch name
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    name = f"{timestamp}_{petname.generate(2)}"

    out_dir = ctx.configs_dir / "scratches"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}.yaml"

    # Save scratch
    save_recipe(
        path=out_path,
        cfg_data=cfg.model_dump(mode="python"),
        schema_cls=ctx.schema,
        overrides=overrides,
        kind="scratch",
        source=lineage,
        description=ns.message,
        force=False,
    )

    print(f"Running scratch: scratches/{name}.yaml")

    return _execute(
        ctx,
        cfg,
        source={"kind": "scratch", "path": str(out_path), "name": name_label},
        overrides=overrides,
        run_id_override=getattr(ns, "run_id", None),
    )


def _execute(
    ctx: Ctx,
    cfg: Any,
    source: dict[str, Any],
    overrides: list[str],
    run_id_override: str | None = None,
) -> Any:
    """Orchestrate the execution lifecycle."""
    # 1. Derive Run ID
    run_id = derive_run_id(
        label=source.get("name") or "run", run_id_override=run_id_override
    )

    # 2. Setup Manifest (only on rank 0)
    rank = get_rank()
    if rank == 0:
        record_run_manifest(ctx, cfg, run_id, source, overrides)

    manifest_path = ctx.runs_dir / run_id / "manifest.yaml"

    # 3. Execute User Function
    with RunContext(manifest_path, rank) as run_ctx:
        result = ctx.func(cfg)
        run_ctx.record_result(result)
        return result
