"""Decorator-based CLI for Stryx."""

from __future__ import annotations

import functools
import sys
from pathlib import Path
from typing import Any, Callable, TypeVar

import typer
from pydantic import BaseModel

from stryx.utils import Ctx
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


def dispatch(
    schema: type[BaseModel],
    configs_dir: Path,
    runs_dir: Path,
    func: Callable[[Any], Any],
    argv: list[str],
) -> Any:
    """Parse CLI arguments and execute the corresponding command handler."""
    app = typer.Typer(
        name="stryx",
        add_completion=False,
        help="Experiment management CLI",
        no_args_is_help=True,
    )

    # Register commands
    app.command(name="new")(cmd_new)
    app.command(name="fork")(cmd_fork)
    app.command(name="run")(cmd_run)
    app.command(name="try")(cmd_try)
    
    # Subcommands for list (using a sub-app)
    list_app = typer.Typer(name="list", help="List experiments or execution runs")
    list_app.command(name="configs")(cmd_list_configs)
    list_app.command(name="runs")(cmd_list_runs)
    app.add_typer(list_app, name="list")

    app.command(name="edit")(cmd_edit)
    app.command(name="show")(cmd_show)
    app.command(name="diff")(cmd_diff)
    app.command(name="schema")(cmd_schema)

    @app.callback()
    def main(
        ctx: typer.Context,
        runs_dir_opt: Path = typer.Option(None, "--runs-dir", help="Override runs directory"),
        configs_dir_opt: Path = typer.Option(None, "--configs-dir", help="Override configs directory"),
    ):
        """
        Stryx CLI entry point.
        """
        final_runs = runs_dir_opt.expanduser() if runs_dir_opt else runs_dir
        final_configs = configs_dir_opt.expanduser() if configs_dir_opt else configs_dir
        
        ctx.obj = Ctx(
            schema=schema,
            configs_dir=final_configs,
            runs_dir=final_runs,
            func=func,
        )

    # Execute
    # We use standalone_mode=False to handle exceptions ourselves if needed,
    # but strictly speaking, standard Typer behavior (True) is usually what we want for CLI.
    # However, to avoid 'SystemExit: 0' showing up in traceback during some invocations, 
    # we can wrap it.
    try:
        app(argv)
    except SystemExit as e:
        # Re-raise to actually exit the process
        raise e