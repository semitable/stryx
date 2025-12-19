"""Shared utilities for Stryx."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# Type aliases
FieldPath = tuple[str | int, ...]


# ============================================================================
# YAML File I/O
# ============================================================================


def read_yaml(path: Path) -> Any:
    """Read and parse a YAML file.

    Args:
        path: Path to the YAML file

    Returns:
        Parsed YAML content, or empty dict if file is empty/null
    """
    content = path.read_text(encoding="utf-8")
    loaded = yaml.safe_load(content)
    return loaded if loaded is not None else {}


def write_yaml(path: Path, data: Any) -> None:
    """Atomically write data to a YAML file.

    Uses temp file + rename for atomic writes.

    Args:
        path: Destination path
        data: Data to serialize as YAML
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    txt = yaml.safe_dump(data, sort_keys=False, default_flow_style=False)
    tmp_path.write_text(txt, encoding="utf-8")
    tmp_path.replace(path)


# ============================================================================
# Nested Data Access
# ============================================================================


def get_nested(data: dict[str, Any], path: FieldPath) -> Any:
    """Get value at a nested path, returning None if path doesn't exist.

    Args:
        data: Root dictionary
        path: Tuple of keys/indices like ("user", "settings", "theme")

    Returns:
        Value at path, or None if not found
    """
    current = data
    try:
        for key in path:
            current = current[key]
        return current
    except (KeyError, IndexError, TypeError):
        return None


def set_nested(data: dict[str, Any], path: FieldPath, value: Any) -> None:
    """Set value at a nested path, creating intermediate dicts as needed.

    Args:
        data: Root dictionary to modify in-place
        path: Tuple of keys like ("user", "settings", "theme")
        value: Value to set
    """
    current = data
    for key in path[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    current[path[-1]] = value


def set_dotpath(data: dict[str, Any], dotpath: str, value: Any) -> None:
    """Set value using dot-separated path string.

    Args:
        data: Root dictionary to modify in-place
        dotpath: Dot-separated path like "user.settings.theme"
        value: Value to set
    """
    parts = tuple(dotpath.split("."))
    set_nested(data, parts, value)


def path_to_str(path: FieldPath) -> str:
    """Convert a field path to human-readable string.

    Examples:
        () -> "<root>"
        ("name",) -> "name"
        ("user", "age") -> "user.age"
        ("items", 0) -> "items[0]"
    """
    if not path:
        return "<root>"

    parts = []
    for component in path:
        if isinstance(component, int):
            parts.append(f"[{component}]")
        else:
            separator = "" if not parts else "."
            parts.append(f"{separator}{component}")
    return "".join(parts)


def parse_like_yaml(text: str, fallback: Any) -> Any:
    """Parse text as YAML, falling back to string or original value on error.

    Args:
        text: Text to parse
        fallback: Value to return if text is empty

    Returns:
        Parsed YAML value, or text string if parsing fails, or fallback if empty
    """
    stripped = text.strip()
    if not stripped:
        return fallback
    try:
        return yaml.safe_load(stripped)
    except Exception:
        return text


def flatten_config(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Recursively flatten a nested dictionary into dot-separated keys.

    Note: Lists are treated as atomic values and not recursed into.

    Args:
        data: The dictionary to flatten.
        prefix: Current key prefix.

    Returns:
        A flat dictionary where keys are dot-paths (e.g., "train.batch_size").
    """
    items = {}
    for k, v in data.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            items.update(flatten_config(v, key))
        else:
            items[key] = v
    return items


# ============================================================================
# Path Utilities
# ============================================================================


def get_next_sequential_name(
    directory: Path, prefix: str = "exp", extension: str = "yaml"
) -> str:
    """Find next sequential name (e.g., exp_001, exp_002) in a directory."""
    existing = list(directory.glob(f"{prefix}_*.{extension}"))
    numbers = []
    for p in existing:
        try:
            # exp_001.yaml -> 1
            num = int(p.stem.split("_")[1])
            numbers.append(num)
        except (IndexError, ValueError):
            continue

    next_num = max(numbers, default=0) + 1
    return f"{prefix}_{next_num:03d}"


def resolve_recipe_path(configs_dir: Path, name: str) -> Path:
    """Find recipe file by name (trying extensions, scratches, etc).

    Args:
        configs_dir: The root directory for configurations.
        name: The name or path of the recipe to find.

    Returns:
        Path object if found.

    Raises:
        FileNotFoundError: If the recipe cannot be found.
    """
    # 1. Exact path
    p = Path(name)
    if p.exists():
        return p

    # 2. Relative to configs_dir
    p = configs_dir / name
    if p.exists():
        return p

    # 3. Try extensions
    for ext in [".yaml", ".yml"]:
        p = configs_dir / f"{name}{ext}"
        if p.exists():
            return p

    # 4. Try scratches
    scratches = configs_dir / "scratches"
    if scratches.exists():
        p = scratches / name
        if p.exists():
            return p
        for ext in [".yaml", ".yml"]:
            p = scratches / f"{name}{ext}"
            if p.exists():
                return p

    raise FileNotFoundError(f"Source recipe not found: {name}")