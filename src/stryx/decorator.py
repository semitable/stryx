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
            ctx = Ctx(
                schema=resolved_schema,
                configs_dir=recipes_path,
                runs_dir=runs_path,
                func=func
            )
            return dispatch(ctx, sys.argv[1:])

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