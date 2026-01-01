"Core configuration management for Pydantic schemas."

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Type, TypeVar

from pydantic import BaseModel, ValidationError

from .utils import FieldPath, read_yaml, write_yaml, get_nested, set_nested, set_dotpath, parse_smart
from .schema import SchemaIntrospector, is_discriminated_union, unwrap_optional, get_union_members

T = TypeVar("T", bound=BaseModel)


class ConfigManager:
    """Manages Pydantic-based configuration data with validation and persistence."""

    def __init__(self, schema: Type[BaseModel], file_path: Path):
        self.schema = schema
        self.file_path = file_path
        self.introspector = SchemaIntrospector(schema)
        self.metadata: dict[str, Any] = {}
        self.data = self._load_or_init()

    def _load_or_init(self) -> dict[str, Any]:
        """Load config from file, or initialize with schema defaults."""
        if self.file_path.exists():
            data = read_yaml(self.file_path)
            # Extract metadata if present
            if isinstance(data, dict) and "__stryx__" in data:
                self.metadata = data.pop("__stryx__")
        else:
            # Initialize with defaults from schema for new configs
            data = self._build_defaults_dict()

        # Always try to validate and fill in any remaining defaults
        try:
            obj = self.schema.model_validate(data)
            return obj.model_dump(mode="python")
        except ValidationError:
            # If validation fails, return what we have so user can fix it
            # (This can happen if there are required fields without defaults)
            return data

    def _build_defaults_dict(self) -> dict[str, Any]:
        """Build a dictionary with all default values from the schema.

        This properly handles discriminated unions by instantiating the first variant.
        """
        from pydantic_core import PydanticUndefined

        result = {}

        for field_name, field_info in self.schema.model_fields.items():
            # Unwrap Optional if present
            field_type, _ = unwrap_optional(field_info.annotation)

            # Handle discriminated unions first
            if is_discriminated_union(field_info):
                # Get union members (excluding None)
                non_none_members = get_union_members(field_type)

                if non_none_members:
                    first_variant = non_none_members[0]
                    if isinstance(first_variant, type) and issubclass(first_variant, BaseModel):
                        result[field_name] = first_variant().model_dump(mode="python")
                        continue

            # Use explicit defaults
            if field_info.default is not PydanticUndefined:
                result[field_name] = field_info.default
            elif field_info.default_factory is not None:
                result[field_name] = field_info.default_factory()
            # For nested BaseModels, instantiate with defaults
            elif isinstance(field_type, type) and issubclass(field_type, BaseModel):
                result[field_name] = field_type().model_dump(mode="python")

        return result

    def validate(self) -> tuple[bool, str]:
        """Validate current data against schema.

        Returns:
            (is_valid, error_message)
        """
        try:
            self.schema.model_validate(self.data)
            return True, ""
        except ValidationError as e:
            # Format errors in a readable way
            errors = e.errors()
            error_lines = []
            for err in errors[:3]:  # Show first 3 errors
                loc = ".".join(str(x) for x in err["loc"])
                msg = err["msg"]
                error_lines.append(f"{loc}: {msg}")

            if len(errors) > 3:
                error_lines.append(f"... and {len(errors) - 3} more errors")

            return False, "; ".join(error_lines)

    def save(self) -> None:
        """Atomically save data to file with validation."""
        # Validate before saving
        is_valid, error = self.validate()
        if not is_valid:
            raise ValueError(error)

        to_save = dict(self.data)
        if self.metadata:
            # Put metadata at the top
            to_save = {"__stryx__": self.metadata, **to_save}

        write_yaml(self.file_path, to_save)

    def get_at(self, path: FieldPath) -> Any:
        """Get value at the given path, returning None if path doesn't exist."""
        return get_nested(self.data, path)

    def set_at(self, path: FieldPath, value: Any) -> None:
        """Set value at the given path, creating intermediate dicts as needed."""
        set_nested(self.data, path, value)

    def flatten_scalars(self) -> list[tuple[FieldPath, Any]]:
        """Recursively flatten nested structure to (path, value) pairs for all scalars."""
        results = []

        def recurse(obj: Any, path: FieldPath) -> None:
            if isinstance(obj, dict):
                for key, val in obj.items():
                    recurse(val, path + (str(key),))
            elif isinstance(obj, list):
                for idx, val in enumerate(obj):
                    recurse(val, path + (idx,))
            else:
                results.append((path, obj))

        recurse(self.data, tuple())
        return results

    # ============================================================================
    # Schema Introspection (delegated to SchemaIntrospector)
    # ============================================================================

    def get_field_info_for_path(self, path: FieldPath):
        """Get schema information for a field path.

        Delegates to SchemaIntrospector.
        """
        return self.introspector.get_field_info_for_path(path)

    def get_union_variants(self, path: FieldPath):
        """Get available variants for a discriminated union field.

        Delegates to SchemaIntrospector.
        """
        return self.introspector.get_union_variants(path)

    def is_discriminator_field(self, path: FieldPath):
        """Check if a field is a discriminator field for a union.

        Delegates to SchemaIntrospector.
        """
        return self.introspector.is_discriminator_field(path)


# ============================================================================
# CLI Helpers (Stateless)
# ============================================================================


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

    value = parse_smart(raw.strip())
    set_dotpath(data, key, value)


def read_config_file(path: Path) -> Any:
    """Read config from YAML or JSON."""
    suffix = path.suffix.lower()

    if suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))

    if suffix in (".yaml", ".yml"):
        return read_yaml(path)

    raise SystemExit(f"Unsupported format: {suffix} (use .yaml or .json)")


def save_recipe(
    path: Path,
    cfg_data: dict[str, Any],
    schema_cls: type,
    overrides: list[str],
    kind: str = "canonical",  # "canonical" or "scratch"
    source: str | None = None,
    description: str | None = None,
    force: bool = False,
) -> None:
    """Construct metadata and write recipe to file.

    Args:
        path: Destination path.
        cfg_data: Dictionary dump of the configuration.
        schema_cls: The schema class (for metadata).
        overrides: List of overrides applied.
        kind: Recipe type ("canonical" or "scratch").
        source: Optional source lineage string.
        description: Optional description.
        force: If True, overwrite existing file.

    Raises:
        FileExistsError: If path exists and force is False.
    """
    if path.exists() and not force:
        raise FileExistsError(f"Recipe '{path}' already exists.")

    meta = {
        "schema": f"{schema_cls.__module__}:{schema_cls.__name__}",
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
        "type": kind,
        "overrides": overrides,
    }
    if source:
        meta["from"] = source
    if description:
        meta["description"] = description

    payload = {"__stryx__": meta, **cfg_data}

    write_yaml(path, payload)