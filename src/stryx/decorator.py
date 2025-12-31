"""Decorator-based CLI for Stryx."""

from __future__ import annotations

import functools
import sys
from pathlib import Path
from typing import Any, Callable, TypeVar

from pydantic import BaseModel

from .cli import dispatch
from .context import Ctx

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
            ctx = Ctx(
                schema=schema,
                configs_dir=recipes_path,
                runs_dir=runs_path,
                func=func,
            )
            return dispatch(ctx, sys.argv[1:])

        # Attach metadata for introspection
        wrapper._stryx_schema = schema  # type: ignore
        wrapper._stryx_recipes_dir = recipes_path  # type: ignore

        return wrapper

    return decorator