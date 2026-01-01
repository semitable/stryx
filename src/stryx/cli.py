"""Decorator-based CLI for Stryx."""

from __future__ import annotations

import functools
import sys
from pathlib import Path
from typing import Any, Callable, TypeVar, Optional, Annotated

import typer
from pydantic import BaseModel

from stryx.utils import Ctx
from stryx.commands import (
    run_exec,
    run_list,
    run_show,
    run_diff,
    recipe_fork,
    recipe_diff,
    recipe_edit,
    recipe_init,
    recipe_list,
    recipe_schema,
    recipe_show,
)

T = TypeVar("T", bound=BaseModel)


def cli(
    schema: type[T],
    recipes_dir: str | Path = "configs",
    runs_dir: str | Path = "runs",
) -> Callable[[Callable[[T], Any]], Callable[..., Any]]:
    """Decorator that adds Stryx CLI to a function."""
    recipes_path = Path(recipes_dir)
    runs_path = Path(runs_dir)

    def decorator(func: Callable[[T], Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            if args or kwargs:
                return func(*args, **kwargs)

            return dispatch(
                schema=schema,
                configs_dir=recipes_path,
                runs_dir=runs_path,
                func=func,
                argv=sys.argv[1:],
            )

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
    """Parse CLI arguments and execute."""
    app = typer.Typer(
        name="stryx",
        add_completion=False,
        help="Experiment management CLI",
        no_args_is_help=True,
        pretty_exceptions_enable=False,
    )

    # 1. Top-level run alias (Most common command)
    app.command(name="run", help="Execute an experiment (alias for 'runs exec').")(run_exec)

    # 2. Recipe Management Group
    recipe_app = typer.Typer(
        name="configs",
        help="Manage experiment configs (recipes).",
        no_args_is_help=True,
    )
    recipe_app.command(name="init")(recipe_init)
    recipe_app.command(name="fork")(recipe_fork)
    recipe_app.command(name="edit", hidden=True)(recipe_edit)
    recipe_app.command(name="show")(recipe_show)
    recipe_app.command(name="diff")(recipe_diff)
    recipe_app.command(name="list")(recipe_list)
    recipe_app.command(name="schema")(recipe_schema)
    
    app.add_typer(recipe_app)

    # 3. Run Management Group
    run_app = typer.Typer(
        name="runs",
        help="Execute and manage experiment runs.",
        no_args_is_help=True,
    )
    run_app.command(name="exec")(run_exec)
    run_app.command(name="list")(run_list)
    run_app.command(name="show")(run_show)
    run_app.command(name="diff")(run_diff)
    
    app.add_typer(run_app)

    @app.callback()
    def main(
        ctx: typer.Context,
        runs_dir_opt: Path = typer.Option(None, "--runs-dir", help="Override runs directory"),
        configs_dir_opt: Path = typer.Option(None, "--configs-dir", help="Override configs directory"),
    ):
        final_runs = runs_dir_opt.expanduser() if runs_dir_opt else runs_dir
        final_configs = configs_dir_opt.expanduser() if configs_dir_opt else configs_dir
        
        ctx.obj = Ctx(
            schema=schema,
            configs_dir=final_configs,
            runs_dir=final_runs,
            func=func,
        )

    try:
        app(argv)
    except SystemExit as e:
        raise e


# ============================================================================
# Standalone CLI (stryx command)
# ============================================================================

stryx_app = typer.Typer(
    name="stryx",
    help="Stryx management utility",
    add_completion=False,
    no_args_is_help=True,
)


@stryx_app.command(name="create-run-id")
def create_run_id_cmd(
    label: Annotated[
        Optional[str], 
        typer.Option(help="Optional label to include in the ID.")
    ] = None,
) -> None:
    """Generate a unique, timestamped run ID."""
    from stryx.run_id import _generate_local_id
    print(_generate_local_id(label))


@stryx_app.command(name="version")
def version_cmd() -> None:
    """Print the version of Stryx."""
    import importlib.metadata
    try:
        print(importlib.metadata.version("stryx"))
    except importlib.metadata.PackageNotFoundError:
        print("unknown")


def main() -> None:
    """Entry point for the 'stryx' command line tool."""
    stryx_app()
