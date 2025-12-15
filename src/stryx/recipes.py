from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generic, Optional, Sequence, Type, TypeVar

import yaml
from pydantic import BaseModel, ValidationError

from .utils import read_yaml, write_yaml, set_dotpath

T = TypeVar("T", bound=BaseModel)


def recipe_cli(
    *,
    schema: Type[T],
    recipes_dir: str | Path = "conf/recipes",
    argv: Optional[Sequence[str]] = None,
) -> "RecipeCmd[T]":
    """
    Entry point for recipe-based CLI.

    Usage:
      new <name> [overrides...]  - Create compiled recipe
      show <name>                - Display recipe contents
      run <name>                 - Load and return validated config
      --config <path>            - Load from explicit path
    """
    if argv is None:
        argv = sys.argv[1:]
    recipes_dir = Path(recipes_dir)

    if not argv:
        raise SystemExit(
            "Usage:\n"
            "  new <name> [overrides...]\n"
            "  show <name>\n"
            "  run <name>\n"
            "  --config <path>\n"
        )

    if argv[0] in ("--config", "-c"):
        if len(argv) < 2:
            raise SystemExit("--config requires a path")
        return RecipeCmd(
            kind="config",
            schema=schema,
            recipes_dir=recipes_dir,
            name=None,
            path=Path(argv[1]),
            overrides=[],
        )

    cmd = argv[0]
    if cmd not in ("new", "show", "run"):
        raise SystemExit(f"Unknown command '{cmd}'. Expected: new/show/run or --config PATH")

    if len(argv) < 2:
        raise SystemExit(f"'{cmd}' requires a name argument")

    name = argv[1]
    overrides = list(argv[2:])

    return RecipeCmd(
        kind=cmd,
        schema=schema,
        recipes_dir=recipes_dir,
        name=name,
        path=None,
        overrides=overrides,
    )


@dataclass
class RecipeCmd(Generic[T]):
    """Command object returned by recipe_cli with methods for each operation."""

    kind: str  # "new" | "show" | "run" | "config"
    schema: Type[T]
    recipes_dir: Path
    name: Optional[str]
    path: Optional[Path]
    overrides: list[str]

    def recipe_path(self) -> Path:
        """Get the path to the recipe file."""
        if self.kind == "config":
            assert self.path is not None
            return self.path
        assert self.name is not None
        return self.recipes_dir / f"{self.name}.yaml"

    def write(self) -> Path:
        """Create and write a compiled recipe (for 'new' command)."""
        if self.kind != "new":
            raise RuntimeError("write() only valid for kind='new'")

        self.recipes_dir.mkdir(parents=True, exist_ok=True)

        # Start with empty dict; schema defaults will be applied during validation
        data: dict[str, Any] = {}

        # Try initial validation to get defaults
        base_obj, _ = _try_validate(self.schema, data)
        if base_obj is not None:
            data = base_obj.model_dump(mode="python")

        # Apply overrides
        for tok in self.overrides:
            _apply_override_token(data, tok)

        cfg = self._validate_or_die(data, context="building recipe")

        # Write with metadata header
        out_path = self.recipe_path()
        payload: dict[str, Any] = {
            "__stryx__": {
                "schema": f"{self.schema.__module__}:{self.schema.__name__}",
                "built_at": datetime.now(tz=timezone.utc).isoformat(),
                "overrides": list(self.overrides),
            }
        }
        payload.update(cfg.model_dump(mode="python"))

        write_yaml(out_path, payload)
        return out_path

    def text(self) -> str:
        """Read and return recipe file contents (for 'show' command)."""
        if self.kind != "show":
            raise RuntimeError("text() only valid for kind='show'")
        return self.recipe_path().read_text(encoding="utf-8")

    def load(self) -> T:
        """Load and validate recipe (for 'run'/'config' commands)."""
        if self.kind not in ("run", "config"):
            raise RuntimeError("load() only valid for kind='run' or kind='config'")

        path = self.recipe_path()
        data = _read_yaml_or_json(path)

        # Strip metadata if present
        if isinstance(data, dict) and "__stryx__" in data:
            data = dict(data)
            data.pop("__stryx__", None)

        return self._validate_or_die(data, context=f"loading config from {path}")

    def _validate_or_die(self, data: Any, *, context: str) -> T:
        """Validate data against schema or exit with error message."""
        obj, err = _try_validate(self.schema, data)
        if obj is not None:
            return obj
        assert err is not None
        raise SystemExit(f"Config validation failed while {context}:\n\n{err}\n")


def _apply_override_token(root: dict[str, Any], tok: str) -> None:
    """Apply Hydra-style override: key=value or a.b.c=value."""
    if "=" not in tok:
        raise SystemExit(f"Invalid override '{tok}'. Expected form key=value or a.b.c=value")

    key, raw = tok.split("=", 1)
    key = key.strip()
    if not key:
        raise SystemExit(f"Invalid override '{tok}': empty key")

    value = _parse_value(raw.strip())
    set_dotpath(root, key, value)


def _parse_value(s: str) -> Any:
    """
    Parse override value with smart type inference:
    - null/none/true/false
    - numbers (int/float)
    - quoted strings
    - JSON literals ({...} or [...])
    - fallback to raw string
    """
    if s == "":
        return ""

    low = s.lower()
    if low in ("null", "none"):
        return None
    if low in ("true", "false"):
        return low == "true"

    # Quoted strings
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        return s[1:-1]

    # JSON literals
    if s[0] in ("{", "["):
        try:
            return json.loads(s)
        except Exception:
            return s

    # Numbers
    try:
        if any(c in s for c in (".", "e", "E")):
            return float(s)
        return int(s)
    except Exception:
        return s


def _try_validate(schema: Type[T], data: Any) -> tuple[Optional[T], Optional[ValidationError]]:
    """Attempt validation, returning (object, None) on success or (None, error) on failure."""
    try:
        return schema.model_validate(data), None
    except ValidationError as e:
        return None, e


def _read_yaml_or_json(path: Path) -> Any:
    """Read and parse YAML or JSON config file."""
    suf = path.suffix.lower()

    if suf == ".json":
        return json.loads(path.read_text(encoding="utf-8"))

    if suf in (".yaml", ".yml"):
        return read_yaml(path)

    raise SystemExit(f"Unsupported config extension: {suf} (expected .yaml/.yml/.json)")
