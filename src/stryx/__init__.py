from __future__ import annotations

from typing import TYPE_CHECKING, Any

# Config management
from .config import ConfigManager

# Core decorator API
from .cli import cli

# Lifecycle
from .lifecycle import current_run

# Schema introspection
from .schema import FieldInfo, SchemaIntrospector, extract_fields

# Utilities
from .utils import (
    FieldPath,
    get_nested,
    parse_smart,
    path_to_str,
    read_yaml,
    set_dotpath,
    set_nested,
    write_yaml,
)

if TYPE_CHECKING:
    from pathlib import Path

    run_path: Path

__all__ = [
    "cli",
    "ConfigManager",
    "current_run",
    "FieldInfo",
    "SchemaIntrospector",
    "extract_fields",
    "PydanticConfigTUI",
    "FieldPath",
    "get_nested",
    "parse_smart",
    "path_to_str",
    "read_yaml",
    "set_dotpath",
    "set_nested",
    "write_yaml",
    "run_path",
]


def __getattr__(name: str) -> Any:
    if name == "run_path":
        from .lifecycle import resolve_run_path

        return resolve_run_path()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")