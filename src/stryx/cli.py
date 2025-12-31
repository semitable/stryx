"""Decorator-based CLI for Stryx."""

from __future__ import annotations

import argparse
import functools
import sys
from pathlib import Path
from typing import Any, Callable, TypeVar

from pydantic import BaseModel

from stryx.commands import (
    cmd_new,
    cmd_fork,
    cmd_run,
    cmd_try,
    cmd_list_configs,
    cmd_list_runs,
    cmd_edit,
    cmd_show,
    cmd_diff,
    cmd_schema,
)

T = TypeVar("T", bound=BaseModel)


def cli(
    schema: type[T],
    recipes_dir: str | Path = "configs",
    runs_dir: str | Path = "runs",
) -> Callable[[Callable[[T], Any]], Callable[..., Any]]:
    """Decorator that adds Stryx CLI to a function.

    Args:
        schema: Pydantic model class for config.
        recipes_dir: Directory for storing recipes (default: configs)
        runs_dir: Directory for storing run manifests (default: runs)

    Example:
        @stryx.cli(schema=Config)
        def main(cfg: Config):
            train(cfg)
    """
    recipes_path = Path(recipes_dir)
    runs_path = Path(runs_dir)

    def decorator(func: Callable[[T], Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # If called with ANY arguments, assume it's a direct function call.
            # We pass them through to let Python handle type checking/errors naturally.
            if args or kwargs:
                return func(*args, **kwargs)

            # Otherwise, parse CLI and dispatch
            return dispatch(
                schema=schema,
                configs_dir=recipes_path,
                runs_dir=runs_path,
                func=func,
                argv=sys.argv[1:],
            )

        # Attach metadata for introspection
        wrapper._stryx_schema = schema  # type: ignore
        wrapper._stryx_recipes_dir = recipes_path  # type: ignore

        return wrapper

    return decorator


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="stryx")
    # global options
    p.add_argument("--runs-dir", dest="runs_dir", type=Path)
    p.add_argument("--configs-dir", dest="configs_dir", type=Path)

    sub = p.add_subparsers(dest="cmd", required=False)
    p.set_defaults(handler=lambda ns: p.print_help())

    # list <configs|runs>
    p_list = sub.add_parser(
        "list",
        help="List experiments or execution runs",
        description="List and compare experiment configurations or review execution history.",
    )
    sub_list = p_list.add_subparsers(dest="what", required=False)
    p_list.set_defaults(handler=lambda ns: p_list.print_help())

    p_list_cfg = sub_list.add_parser("configs", help="List saved experiment recipes")
    p_list_cfg.set_defaults(handler=cmd_list_configs)

    p_list_runs = sub_list.add_parser("runs", help="List execution history")
    p_list_runs.add_argument("--status", default="any", choices=["any", "ok", "failed"])
    p_list_runs.set_defaults(handler=cmd_list_runs)

    p_new = sub.add_parser("new", help="Create a fresh experiment recipe")
    p_new.add_argument("recipe", nargs="?")
    p_new.add_argument("-m", "--message", help="Description of the experiment")
    p_new.add_argument("--force", action="store_true", help="Overwrite existing recipe")
    p_new.add_argument("overrides", nargs=argparse.REMAINDER)
    p_new.set_defaults(handler=cmd_new)

    # try [target] -- overrides...
    p_try = sub.add_parser(
        "try", help="Run an experimental variant (saved to scratches)"
    )
    p_try.add_argument("target", nargs="?")
    p_try.add_argument("--run-id", help="Explicitly set run id")
    p_try.add_argument("-m", "--message", help="Description for scratch metadata")
    p_try.add_argument("overrides", nargs=argparse.REMAINDER)
    p_try.set_defaults(handler=cmd_try)

    # run <recipe|path>
    p_run = sub.add_parser("run", help="Run an existing recipe exactly")
    p_run.add_argument("target")
    p_run.add_argument("--run-id", help="Explicitly set run id")
    p_run.set_defaults(handler=cmd_run)

    # fork <source> <name> -- overrides...
    p_fork = sub.add_parser("fork", help="Fork an existing recipe with modifications")
    p_fork.add_argument("source")
    p_fork.add_argument("name")
    p_fork.add_argument("-m", "--message", help="Description of the experiment")
    p_fork.add_argument(
        "--force", action="store_true", help="Overwrite existing recipe"
    )
    p_fork.add_argument("overrides", nargs=argparse.REMAINDER)
    p_fork.set_defaults(handler=cmd_fork)

    # edit <recipe>
    p_edit = sub.add_parser("edit", help="Edit a recipe interactively (TUI)")
    p_edit.add_argument("recipe")
    p_edit.set_defaults(handler=cmd_edit)

    # show [target] [overrides...]
    p_show = sub.add_parser(
        "show", help="Display configuration with source annotations"
    )
    p_show.add_argument("target", nargs="?")
    p_show.add_argument("overrides", nargs=argparse.REMAINDER)
    p_show.set_defaults(handler=cmd_show)

    # diff <recipe_a> [recipe_b]
    p_diff = sub.add_parser("diff", help="Compare two experiment recipes")
    p_diff.add_argument("recipe_a")
    p_diff.add_argument("recipe_b", nargs="?")
    p_diff.set_defaults(handler=cmd_diff)

    # schema
    p_schema = sub.add_parser("schema", help="Show the configuration schema")
    p_schema.add_argument("--json", action="store_true", help="Output schema as JSON")
    p_schema.set_defaults(handler=cmd_schema)

    return p


def normalize_overrides(tokens: list[str]) -> list[str]:
    """Remove the optional '--' separator if present in the captured tokens.

    argparse.REMAINDER captures everything after the command, including the
    double-dash separator often used to distinguish positional arguments
    from overrides.
    """
    return tokens[1:] if tokens[:1] == ["--"] else tokens


def dispatch(
    schema: type[BaseModel],
    configs_dir: Path,
    runs_dir: Path,
    func: Callable[[Any], Any],
    argv: list[str],
) -> Any:
    parser = build_parser()
    ns = parser.parse_args(argv)

    # Attach context to namespace
    ns.stryx_schema = schema
    ns.stryx_func = func

    # resolve effective dirs (use default if not in args)
    if ns.runs_dir:
        ns.runs_dir = ns.runs_dir.expanduser()
    else:
        ns.runs_dir = runs_dir

    if ns.configs_dir:
        ns.configs_dir = ns.configs_dir.expanduser()
    else:
        ns.configs_dir = configs_dir

    # normalize override lists if this command has them
    if hasattr(ns, "overrides"):
        ns.overrides = normalize_overrides(ns.overrides)

    return ns.handler(ns)
