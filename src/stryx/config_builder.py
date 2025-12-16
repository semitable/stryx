from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from .utils import read_yaml, set_dotpath

T = TypeVar("T", bound=BaseModel)


def build_config(schema: type[T], overrides: list[str]) -> T:
    """Build config from schema defaults + overrides."""
    try:
        base = schema()
        data = base.model_dump(mode="python")
    except ValidationError as e:
        raise SystemExit(f"Schema has required fields without defaults:\n{e}")

    for tok in overrides:
        apply_override(data, tok)

    return validate_or_die(schema, data, "building config")


def load_and_override(schema: type[T], path: Path, overrides: list[str]) -> T:
    """Load config from file, apply overrides, validate."""
    if not path.exists():
        raise SystemExit(f"Config not found: {path}")

    data = read_config_file(path)

    # Strip stryx metadata
    if isinstance(data, dict):
        data = {k: v for k, v in data.items() if not k.startswith("__")}

    for tok in overrides:
        apply_override(data, tok)

    return validate_or_die(schema, data, f"loading {path.name}")


def validate_or_die(schema: type[T], data: Any, context: str) -> T:
    """Validate data against schema or exit with formatted error."""
    try:
        return schema.model_validate(data)
    except ValidationError as e:
        errors = e.errors()
        lines = [f"Config validation failed ({context}):"]
        for err in errors[:5]:
            loc = ".".join(str(x) for x in err["loc"])
            lines.append(f"  {loc}: {err['msg']}")
        if len(errors) > 5:
            lines.append(f"  ... and {len(errors) - 5} more errors")
        raise SystemExit("\n".join(lines))


def apply_override(data: dict[str, Any], tok: str) -> None:
    """Apply a single key=value override."""
    if "=" not in tok:
        raise SystemExit(
            f"Invalid override: '{tok}'\n"
            f"Expected format: key=value (e.g., lr=1e-4, train.steps=1000)"
        )

    key, raw = tok.split("=", 1)
    key = key.strip()
    if not key:
        raise SystemExit(f"Invalid override: '{tok}' (empty key)")

    value = parse_value(raw.strip())
    set_dotpath(data, key, value)


def parse_value(s: str) -> Any:
    """Parse value string with smart type inference.

    Handles: null, true/false, numbers, quoted strings, JSON, raw strings.
    """
    if s == "":
        return ""

    low = s.lower()
    if low in ("null", "none"):
        return None
    if low == "true":
        return True
    if low == "false":
        return False

    # Quoted strings → strip quotes
    if len(s) >= 2:
        if (s[0] == '"' and s[-1] == '"') or (s[0] == "'" and s[-1] == "'"):
            return s[1:-1]

    # JSON arrays/objects
    if s and s[0] in "{[}":
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            return s

    # Numbers
    try:
        if "." in s or "e" in s.lower():
            return float(s)
        return int(s)
    except ValueError:
        return s


def read_config_file(path: Path) -> Any:
    """Read config from YAML or JSON."""
    suffix = path.suffix.lower()

    if suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))

    if suffix in (".yaml", ".yml"):
        return read_yaml(path)

    raise SystemExit(f"Unsupported format: {suffix} (use .yaml or .json)")
