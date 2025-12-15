from __future__ import annotations

import typing
from dataclasses import dataclass
from typing import Any, Type, Union, get_args, get_origin

from pydantic import BaseModel
from pydantic.fields import FieldInfo as PydanticFieldInfo
from pydantic_core import PydanticUndefined

from .utils import FieldPath


# ============================================================================
# Type Helper Functions (DRY)
# ============================================================================

def is_union_type(field_type: Any) -> bool:
    """Check if a type is a Union type."""
    origin = get_origin(field_type)
    return origin is Union or (origin and getattr(origin, "__origin__", None) is Union)


def unwrap_optional(field_type: Any) -> tuple[Any, bool]:
    """Unwrap Optional[X] to get X.

    Args:
        field_type: Type to unwrap

    Returns:
        (unwrapped_type, was_optional)
    """
    if not is_union_type(field_type):
        return field_type, False

    args = get_args(field_type)
    if type(None) not in args:
        return field_type, False

    # It's Optional - extract non-None types
    inner_types = [t for t in args if t is not type(None)]
    if not inner_types:
        return field_type, True

    # Return first non-None type
    return inner_types[0], True


def is_discriminated_union(field_info: PydanticFieldInfo) -> bool:
    """Check if a field is a discriminated union."""
    if not is_union_type(field_info.annotation):
        return False
    return field_info.discriminator is not None


def get_union_members(field_type: Any) -> list[Type[BaseModel] | type(None)]:
    """Get all members of a Union type, excluding None if present.

    Returns list of BaseModel types (or None type).
    """
    if not is_union_type(field_type):
        return []

    args = get_args(field_type)
    return list(args)


# ============================================================================
# Field Extraction
# ============================================================================

@dataclass
class FieldInfo:
    """Represents a flattened config field for UI display."""

    path: str  # "train.batch_size" or "optim.lr"
    type_str: str  # "int", "float", "Literal['adamw', 'sgd']"
    default: Any  # 128, 0.0003, None
    required: bool  # True if no default
    description: str | None  # from Field(description=...)


def extract_fields(schema: Type[BaseModel], prefix: str = "") -> list[FieldInfo]:
    """
    Recursively walk a Pydantic schema and extract all fields as flat paths.

    Args:
        schema: Pydantic BaseModel class
        prefix: Current path prefix for nested fields

    Returns:
        List of FieldInfo objects with flattened paths
    """
    fields: list[FieldInfo] = []

    for field_name, field_info in schema.model_fields.items():
        full_path = f"{prefix}.{field_name}" if prefix else field_name

        # Get the type annotation
        field_type = field_info.annotation
        type_str = _format_type(field_type)

        # Get default value
        default_value = field_info.default
        if default_value is PydanticFieldInfo:
            default_value = None

        # Check if required
        is_required = field_info.is_required()

        # Get description
        description = field_info.description

        # Check if it's a nested BaseModel
        origin = get_origin(field_type)
        args = get_args(field_type)

        # Handle Optional[X] -> extract X
        if origin is type(None) or (origin and type(None) in args):
            # It's Optional, unwrap it
            inner_types = [t for t in args if t is not type(None)]
            if inner_types:
                field_type = inner_types[0]

        # If it's a BaseModel subclass, recurse
        if isinstance(field_type, type) and issubclass(field_type, BaseModel):
            # Add the nested model's fields
            nested_fields = extract_fields(field_type, prefix=full_path)
            fields.extend(nested_fields)
        # Handle Union types (discriminated unions)
        elif origin is Union or (
            origin is not None and getattr(origin, "__origin__", None) is Union
        ):
            # Check if this is a discriminated union by looking for discriminator in field metadata
            discriminator = field_info.discriminator

            if discriminator:
                # This is a discriminated union - extract fields from the default variant
                # Get the union members (exclude None)
                union_members = [arg for arg in args if arg is not type(None)]

                # Try to use the first union member as default, or use default_value to determine
                if (
                    default_value is not None
                    and default_value is not PydanticUndefined
                    and hasattr(default_value, "__class__")
                ):
                    # default_value is an instance, use its type
                    default_variant = default_value.__class__
                elif union_members:
                    # Use first union member
                    default_variant = union_members[0]
                else:
                    # No default, just add as regular field
                    fields.append(
                        FieldInfo(
                            path=full_path,
                            type_str=type_str,
                            default=default_value,
                            required=is_required,
                            description=description,
                        )
                    )
                    continue

                # Recurse into the default variant
                if isinstance(default_variant, type) and issubclass(
                    default_variant, BaseModel
                ):
                    nested_fields = extract_fields(default_variant, prefix=full_path)
                    fields.extend(nested_fields)
            else:
                # Non-discriminated union, just add as regular field
                fields.append(
                    FieldInfo(
                        path=full_path,
                        type_str=type_str,
                        default=default_value,
                        required=is_required,
                        description=description,
                    )
                )
        else:
            # Regular field - add it
            fields.append(
                FieldInfo(
                    path=full_path,
                    type_str=type_str,
                    default=default_value,
                    required=is_required,
                    description=description,
                )
            )

    return fields


def _format_type(typ: Any) -> str:
    """Format a type annotation as a readable string."""
    # Handle None
    if typ is type(None):
        return "None"

    # Handle basic types
    if typ in (int, float, str, bool):
        return typ.__name__

    # Handle generic types
    origin = get_origin(typ)
    args = get_args(typ)

    if origin is None:
        # Not a generic, just use __name__ or repr
        return getattr(typ, "__name__", repr(typ))

    # Handle Optional[X]
    if origin is type(None) or (args and type(None) in args):
        inner = [_format_type(t) for t in args if t is not type(None)]
        return f"Optional[{', '.join(inner)}]"

    # Handle Union[X, Y, ...]
    if hasattr(origin, "__name__") and origin.__name__ == "UnionType":
        formatted_args = [_format_type(arg) for arg in args]
        return " | ".join(formatted_args)

    # Handle Literal
    if hasattr(origin, "__name__") and origin.__name__ == "Literal":
        values = ", ".join(repr(v) for v in args)
        return f"Literal[{values}]"

    # Generic fallback
    origin_name = getattr(origin, "__name__", repr(origin))
    if args:
        args_str = ", ".join(_format_type(arg) for arg in args)
        return f"{origin_name}[{args_str}]"

    return origin_name


# ============================================================================
# Schema Introspection
# ============================================================================

class SchemaIntrospector:
    """Provides schema introspection capabilities for Pydantic models.

    This class handles all schema queries - it doesn't manage data, only schema structure.
    """

    def __init__(self, schema: Type[BaseModel]):
        self.schema = schema

    def get_field_info_for_path(self, path: FieldPath) -> tuple[Type[BaseModel] | None, PydanticFieldInfo | None]:
        """Get schema information for a field path.

        Args:
            path: Tuple path like ("train", "batch_size")

        Returns:
            (parent_schema, field_info) if found, (None, None) otherwise
        """
        current_schema = self.schema
        path_list = list(path)

        for i, key in enumerate(path_list):
            if not hasattr(current_schema, 'model_fields'):
                return None, None

            field_info = current_schema.model_fields.get(key)
            if field_info is None:
                return None, None

            # Last key - return this field info
            if i == len(path_list) - 1:
                return current_schema, field_info

            # Navigate deeper
            field_type = field_info.annotation

            # Handle Optional
            field_type, _ = unwrap_optional(field_type)

            # Check if it's a BaseModel to continue traversal
            if isinstance(field_type, type) and issubclass(field_type, BaseModel):
                current_schema = field_type
            else:
                return None, None

        return None, None

    def get_union_variants(self, path: FieldPath) -> list[tuple[str, Type[BaseModel] | None]]:
        """Get available variants for a discriminated union field.

        Args:
            path: Path to the union field

        Returns:
            List of (variant_name, variant_type) tuples. Empty list if not a union.
            variant_type is None for the "None" option.
        """
        parent_schema, field_info = self.get_field_info_for_path(path)
        if not field_info:
            return []

        # Check if it's a discriminated union
        if not is_discriminated_union(field_info):
            return []

        # Build list of variants
        variants = []
        members = get_union_members(field_info.annotation)

        for member_type in members:
            if member_type is type(None):
                variants.append(("None", None))
            elif isinstance(member_type, type) and issubclass(member_type, BaseModel):
                # Use the class name as display name
                variants.append((member_type.__name__, member_type))

        return variants

    def is_discriminator_field(self, path: FieldPath) -> tuple[bool, FieldPath | None]:
        """Check if a field is a discriminator field for a union.

        Args:
            path: Path to check (e.g., ("sched", "kind"))

        Returns:
            (is_discriminator, parent_path) - parent_path is the union field path if True
        """
        if len(path) < 2:
            return False, None

        # Get parent path (everything except last component)
        parent_path = path[:-1]
        field_name = path[-1]

        # Get field info for the parent
        parent_schema, parent_field_info = self.get_field_info_for_path(parent_path)
        if not parent_field_info:
            return False, None

        # Check if parent has a discriminator and if it matches this field
        discriminator = parent_field_info.discriminator
        if discriminator and discriminator == field_name:
            return True, parent_path

        return False, None
