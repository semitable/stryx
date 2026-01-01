from __future__ import annotations

from typing import TYPE_CHECKING, Any

# Core decorator API
from .cli import cli

# Lifecycle
from .lifecycle import current_run

if TYPE_CHECKING:
    from pathlib import Path

    run_path: Path

__all__ = [
    "cli",
    "current_run",
    "run_path",
]


def __getattr__(name: str) -> Any:
    if name == "run_path":
        from .lifecycle import resolve_run_path

        return resolve_run_path()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
