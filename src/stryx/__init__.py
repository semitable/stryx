"""Stryx - Typed configuration compiler for ML experiments.

Define your config schema in Python with Pydantic, compile frozen YAML recipes
with Hydra-style overrides, and run reproducible experiments.

Example:
    from pydantic import BaseModel
    import stryx

    class Config(BaseModel):
        lr: float = 1e-4
        batch_size: int = 32

    @stryx.cli(schema=Config)
    def main(cfg: Config):
        print(f"Training with lr={cfg.lr}")

    if __name__ == "__main__":
        main()

Usage:
    python train.py                      # Run with defaults
    python train.py lr=1e-3              # Run with overrides
    python train.py new my_exp lr=1e-3   # Save recipe
    python train.py run my_exp           # Run from recipe
    python train.py edit my_exp          # Edit in TUI
"""
from __future__ import annotations

import dataclasses
from typing import Any, Union, get_args, get_origin, get_type_hints

from pydantic import BaseModel, Field, create_model

__version__ = "0.1.0"


# =============================================================================
# Dataclass to Pydantic Conversion
# =============================================================================


def from_dataclass(
    dc: type,
    *,
    _cache: dict[type, type[BaseModel]] | None = None,
) -> type[BaseModel]:
    """Convert a dataclass to a Pydantic model.

    This allows using plain Python dataclasses with Stryx while getting
    all the benefits of Pydantic (validation, TUI support, etc.).

    Args:
        dc: A dataclass type to convert

    Returns:
        A Pydantic BaseModel class with the same fields

    Example:
        from dataclasses import dataclass

        @dataclass
        class Config:
            lr: float = 1e-4
            batch_size: int = 32

        @stryx.cli(schema=stryx.from_dataclass(Config))
        def main(cfg):
            print(cfg.lr)

    Notes:
        - Nested dataclasses are converted recursively
        - Optional[DataclassType] is handled correctly
        - For discriminated unions, use Pydantic directly
    """
    if not dataclasses.is_dataclass(dc):
        raise TypeError(f"Expected a dataclass, got {type(dc).__name__}")

    # Cache to handle recursive structures
    if _cache is None:
        _cache = {}

    if dc in _cache:
        return _cache[dc]

    # Get type hints (resolves forward references)
    try:
        hints = get_type_hints(dc)
    except Exception:
        # Fallback to __annotations__ if get_type_hints fails
        hints = getattr(dc, "__annotations__", {})

    field_definitions: dict[str, Any] = {}

    for f in dataclasses.fields(dc):
        field_type = hints.get(f.name, Any)

        # Convert nested dataclasses in the type
        field_type = _convert_type(field_type, _cache)

        # Handle default values
        if f.default is not dataclasses.MISSING:
            default = f.default
            # If default is a dataclass instance, convert it
            if dataclasses.is_dataclass(default) and not isinstance(default, type):
                converted_type = _convert_type(type(default), _cache)
                default = converted_type(**dataclasses.asdict(default))
            field_definitions[f.name] = (field_type, default)
        elif f.default_factory is not dataclasses.MISSING:
            # Wrap factory to convert dataclass instances
            factory = f.default_factory
            original_type = hints.get(f.name)

            if dataclasses.is_dataclass(original_type):
                converted = _convert_type(original_type, _cache)

                def make_factory(conv=converted, fact=factory):
                    def wrapped():
                        result = fact()
                        if dataclasses.is_dataclass(result):
                            return conv(**dataclasses.asdict(result))
                        return result

                    return wrapped

                field_definitions[f.name] = (
                    field_type,
                    Field(default_factory=make_factory()),
                )
            else:
                field_definitions[f.name] = (field_type, Field(default_factory=factory))
        else:
            # Required field (no default)
            field_definitions[f.name] = (field_type, ...)

    # Create the Pydantic model
    model = create_model(dc.__name__, **field_definitions)

    # Preserve module info for better error messages
    model.__module__ = dc.__module__

    # Store reference to original dataclass
    model._stryx_source_dataclass = dc  # type: ignore

    # Cache before returning (handles recursive refs)
    _cache[dc] = model

    return model


def _convert_type(
    field_type: Any,
    cache: dict[type, type[BaseModel]],
) -> Any:
    """Recursively convert dataclass types within a type annotation."""
    import types

    # Direct dataclass
    if dataclasses.is_dataclass(field_type) and isinstance(field_type, type):
        return from_dataclass(field_type, _cache=cache)

    origin = get_origin(field_type)
    args = get_args(field_type)

    if origin is None:
        return field_type

    # Handle Union[X, Y, ...] (typing.Union)
    if origin is Union:
        converted_args = tuple(_convert_type(arg, cache) for arg in args)
        return Union[converted_args]  # type: ignore

    # Handle X | Y (types.UnionType in Python 3.10+)
    if isinstance(field_type, types.UnionType):
        converted_args = tuple(_convert_type(arg, cache) for arg in args)
        # Rebuild using Union since UnionType isn't subscriptable
        return Union[converted_args]  # type: ignore

    # Handle List[X], Dict[K, V], etc.
    if args:
        converted_args = tuple(_convert_type(arg, cache) for arg in args)
        try:
            return origin[converted_args]  # type: ignore
        except TypeError:
            # Some origins aren't subscriptable, return as-is
            return field_type

    return field_type

# Core decorator API
from .decorator import cli

# TUI (for direct use)
from .tui import launch_tui, PydanticConfigTUI

# Schema introspection
from .schema import extract_fields, FieldInfo, SchemaIntrospector

# Config management
from .config import ConfigManager

# Utilities
from .utils import (
    FieldPath,
    path_to_str,
    parse_like_yaml,
    read_yaml,
    write_yaml,
    get_nested,
    set_nested,
    set_dotpath,
)

__all__ = [
    "__version__",
    # Core API
    "cli",
    "from_dataclass",
    # TUI
    "launch_tui",
    "PydanticConfigTUI",
    # Schema introspection
    "extract_fields",
    "FieldInfo",
    "SchemaIntrospector",
    # Config management
    "ConfigManager",
    # Utilities
    "FieldPath",
    "path_to_str",
    "parse_like_yaml",
    "read_yaml",
    "write_yaml",
    "get_nested",
    "set_nested",
    "set_dotpath",
]
