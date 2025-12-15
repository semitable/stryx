"""Core configuration management for Pydantic schemas."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Type

from pydantic import BaseModel, ValidationError

from .utils import FieldPath, read_yaml, write_yaml, get_nested, set_nested
from .schema import SchemaIntrospector, is_discriminated_union, unwrap_optional


class ConfigManager:
    """Manages Pydantic-based configuration data with validation and persistence."""

    def __init__(self, schema: Type[BaseModel], file_path: Path):
        self.schema = schema
        self.file_path = file_path
        self.introspector = SchemaIntrospector(schema)
        self.data = self._load_or_init()

    def _load_or_init(self) -> dict[str, Any]:
        """Load config from file, or initialize with schema defaults."""
        if self.file_path.exists():
            data = read_yaml(self.file_path)
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
        from typing import get_args

        result = {}

        for field_name, field_info in self.schema.model_fields.items():
            # Unwrap Optional if present
            field_type, _ = unwrap_optional(field_info.annotation)

            # Handle discriminated unions first
            if is_discriminated_union(field_info):
                # Get union members (excluding None)
                members = get_args(field_type)
                non_none_members = [m for m in members if m is not type(None)]

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

        write_yaml(self.file_path, self.data)

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
