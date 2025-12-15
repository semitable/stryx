"""Stryx - Interactive config builder for Pydantic schemas."""

__version__ = "0.1.0"

from .recipes import recipe_cli, RecipeCmd
from .tui import launch_tui
from .schema import extract_fields, FieldInfo, SchemaIntrospector
from .config import ConfigManager
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
    # CLI entry points
    "recipe_cli",
    "RecipeCmd",
    "launch_tui",
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
