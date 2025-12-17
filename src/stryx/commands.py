from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypeVar

import petname
from pydantic import BaseModel, ValidationError

from .cli_parser import ParsedArgs
from .config_builder import (
    apply_override,
    build_config,
    load_and_override,
    read_config_file,
    validate_or_die,
)
from .schema import FieldInfo, extract_fields
from .utils import read_yaml, write_yaml

T = TypeVar("T", bound=BaseModel)

# Sentinel for "value not found"
_NOT_FOUND = object()


def cmd_try(schema: type[T], recipes_dir: Path, args: ParsedArgs) -> Path:
    """Handle: try [source] [overrides...] - create scratchpad run."""
    # Resolve config
    from_path: Path | None = None
    if args.from_recipe:
        # Check if source is a scratch (contains /) or canonical
        name = args.from_recipe
        if not name.endswith(".yaml"):
            name += ".yaml"
        
        if "/" in name:
             from_path = recipes_dir / name
        else:
             from_path = recipes_dir / name
             
        if not from_path.exists():
            # If not found, check if it was a scratch name without prefix
            scratch_path = recipes_dir / "scratches" / name
            if scratch_path.exists():
                from_path = scratch_path
            else:
                raise SystemExit(f"Source recipe not found: {args.from_recipe}")
        
        cfg = load_and_override(schema, from_path, args.overrides)
    else:
        cfg = build_config(schema, args.overrides)

    # Generate scratch name
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    # petname generate(2) gives "adjective-animal"
    name = f"{timestamp}_{petname.generate(2)}"
    
    scratches_dir = recipes_dir / "scratches"
    scratches_dir.mkdir(parents=True, exist_ok=True)
    out_path = scratches_dir / f"{name}.yaml"

    # Prepare payload
    payload: dict[str, Any] = {
        "__stryx__": {
            "schema": f"{schema.__module__}:{schema.__name__}",
            "created_at": datetime.now(tz=timezone.utc).isoformat(),
            "type": "scratch",
        }
    }
    if args.from_recipe:
        payload["__stryx__"]["from"] = args.from_recipe
    if args.overrides:
        payload["__stryx__"]["overrides"] = args.overrides
    payload.update(cfg.model_dump(mode="python"))

    write_yaml(out_path, payload)
    print(f"Running scratch: scratches/{name}.yaml")
    return out_path


def cmd_fork(schema: type[T], recipes_dir: Path, args: ParsedArgs) -> Path:
    """Handle: fork <source> [name] [overrides...]"""
    from filelock import FileLock

    # Resolve config
    from_path: Path | None = None
    
    if not args.from_recipe or args.from_recipe == "defaults":
        # Fork from defaults
        cfg = build_config(schema, args.overrides)
    else:
        # Fork from existing
        name = args.from_recipe
        if not name.endswith(".yaml"):
            name += ".yaml"
            
        if "/" in name:
             from_path = recipes_dir / name
        else:
             from_path = recipes_dir / name
             
        if not from_path.exists():
             # Check scratches
             scratch_path = recipes_dir / "scratches" / name
             if scratch_path.exists():
                 from_path = scratch_path
             else:
                 raise SystemExit(f"Source recipe not found: {args.from_recipe}")
                 
        cfg = load_and_override(schema, from_path, args.overrides)

    # Prepare payload
    payload: dict[str, Any] = {
        "__stryx__": {
            "schema": f"{schema.__module__}:{schema.__name__}",
            "created_at": datetime.now(tz=timezone.utc).isoformat(),
            "type": "canonical",
        }
    }
    if args.from_recipe:
        payload["__stryx__"]["from"] = args.from_recipe
    if args.overrides:
        payload["__stryx__"]["overrides"] = args.overrides
    payload.update(cfg.model_dump(mode="python"))

    # Create directory
    recipes_dir.mkdir(parents=True, exist_ok=True)

    # If name provided, just write directly
    if args.recipe:
        out_path = recipes_dir / f"{args.recipe}.yaml"
        write_yaml(out_path, payload)
        print(f"Forked to: {out_path}")
        return out_path

    # Auto-generate sequential name with locking
    lock_path = recipes_dir / ".stryx.lock"
    with FileLock(lock_path):
        name = _next_sequential_name(recipes_dir)
        out_path = recipes_dir / f"{name}.yaml"
        write_yaml(out_path, payload)

    print(f"Forked to: {out_path}")
    return out_path


def _next_sequential_name(recipes_dir: Path) -> str:
    """Find next sequential experiment name (exp_001, exp_002, etc.).

    Must be called while holding the directory lock.
    """
    existing = list(recipes_dir.glob("exp_*.yaml"))
    numbers = []
    for p in existing:
        try:
            # exp_001.yaml → 1
            num = int(p.stem.split("_")[1])
            numbers.append(num)
        except (IndexError, ValueError):
            continue

    next_num = max(numbers, default=0) + 1
    return f"exp_{next_num:03d}"


def cmd_edit(schema: type[T], recipes_dir: Path, name: str) -> None:
    """Handle: edit <name> - launch TUI editor."""
    from .tui import PydanticConfigTUI

    recipe_path = recipes_dir / f"{name}.yaml"

    if not recipe_path.exists():
        # Offer to create it
        raise SystemExit(
            f"Recipe not found: {recipe_path}\n"
            f"Create it first with: new {name}"
        )

    tui = PydanticConfigTUI(schema, recipe_path)
    tui.run()


def cmd_show(schema: type[T], recipes_dir: Path, args: ParsedArgs) -> None:
    """Handle: show [recipe] [--config path] [overrides...] 

    Pretty-prints the config with source annotations showing where each value
    comes from: (default), (recipe), or (override ← previous_value).
    """
    # Get schema defaults
    try:
        defaults_instance = schema()
        schema_defaults = defaults_instance.model_dump(mode="python")
    except ValidationError as e:
        raise SystemExit(f"Schema has required fields without defaults:\n{e}")

    # Determine source file
    recipe_path = recipes_dir / f"{args.recipe}.yaml" if args.recipe else None
    source_file: Path | None = args.config_path or recipe_path
    source_name = "defaults"
    recipe_data: dict[str, Any] | None = None

    if source_file:
        if not source_file.exists():
            raise SystemExit(f"Config not found: {source_file}")
        recipe_data = read_config_file(source_file)
        # Strip metadata
        if isinstance(recipe_data, dict):
            recipe_data = {k: v for k, v in recipe_data.items() if not k.startswith("__")}
        source_name = source_file.stem if recipe_path else source_file.name

    # Build the config data (before validation, to track sources)
    if recipe_data is not None:
        data = dict(recipe_data)
    else:
        data = dict(schema_defaults)

    # Track override paths and their previous values
    override_info: dict[str, Any] = {}  # path → previous value
    for tok in args.overrides:
        key, _ = tok.split("=", 1)
        key = key.strip()
        # Get previous value before override
        prev = _get_nested(data, key.split("."))
        override_info[key] = prev
        apply_override(data, tok)

    # Validate
    cfg = validate_or_die(schema, data, "show")
    final_data = cfg.model_dump(mode="python")

    # Print header
    header_parts = ["Config"]
    if source_name != "defaults":
        header_parts.append(f"recipe: {source_name}")
    if args.overrides:
        header_parts.append(f"{len(args.overrides)} override{'s' if len(args.overrides) > 1 else ''}")
    if len(header_parts) > 1:
        print(f"{header_parts[0]} ({', '.join(header_parts[1:])})")
    else:
        print(header_parts[0])
    print("=" * 60)

    # Print config with sources
    _print_with_sources(
        final_data,
        schema_defaults,
        recipe_data,
        override_info,
        prefix="",
        indent=0,
    )


def cmd_list(recipes_dir: Path) -> None:
    """Handle: list - show all recipes in the recipes directory."""
    if not recipes_dir.exists():
        print(f"No recipes directory: {recipes_dir}")
        return

    # Canonical
    print("Experiments:")
    print("-" * 60)
    canonicals = sorted(recipes_dir.glob("*.yaml")) + sorted(recipes_dir.glob("*.yml"))
    if canonicals:
        _print_recipe_list(canonicals)
    else:
        print("  (none)")
    print()

    # Scratches
    scratches_dir = recipes_dir / "scratches"
    if scratches_dir.exists():
        scratches = sorted(scratches_dir.glob("*.yaml"), reverse=True)  # Newest first
        if scratches:
            print("Scratches:")
            print("-" * 60)
            _print_recipe_list(scratches)


def _print_recipe_list(recipes: list[Path]) -> None:
    for recipe_path in recipes:
        name = recipe_path.stem

        # Try to get metadata
        try:
            data = read_yaml(recipe_path)
            meta = data.get("__stryx__", {}) if isinstance(data, dict) else {}
            created = meta.get("created_at", "")
            if created:
                # Parse and format date
                try:
                    dt = datetime.fromisoformat(created)
                    date_str = dt.strftime("%Y-%m-%d %H:%M")
                except ValueError:
                    date_str = created[:16]
            else:
                date_str = ""

            # Show if it was derived from another recipe
            from_recipe = meta.get("from", "")
            if from_recipe:
                name = f"{name} (from: {from_recipe})"

        except Exception:
            date_str = ""

        if date_str:
            print(f"  {name:<40} {date_str}")
        else:
            print(f"  {name}")


def cmd_schema(schema: type[T]) -> None:
    """Handle: schema - print the configuration schema."""
    print(f"Schema: {schema.__module__}:{schema.__name__}")

    fields = extract_fields(schema)
    if fields:
        print("Fields:")
        groups: dict[str, list[FieldInfo]] = {}
        group_order: list[str] = []

        # Bucket fields by their first path segment
        for field in fields:
            path = field.path
            group = path.split(".", 1)[0]
            if group not in groups:
                groups[group] = []
                group_order.append(group)
            groups[group].append(field)

        grouped: list[tuple[str, FieldInfo | None, list[FieldInfo]]] = []
        for group in group_order:
            entries = groups[group]
            parent = next((e for e in entries if e.path == group), None)
            children = [e for e in entries if e.path != group]
            grouped.append((group, parent, children))

        for idx, (group, parent, children) in enumerate(grouped):
            has_children = bool(children)
            # Parent line (type/default for the group itself)
            if parent:
                for line in _format_field_lines(
                    indent="  ",
                    label=group,
                    type_name=parent.type_str,
                    default_str=parent.default_str,
                    description=parent.description,
                ):
                    print(line)
            else:
                print(f"  {group}:")

            # Child lines (strip the group prefix for readability)
            for child in children:
                child_path = child.path
                label = child_path[len(group) + 1 :] if child_path.startswith(f"{group}.") else child_path
                for line in _format_field_lines(
                    indent="    ",
                    label=label,
                    type_name=child.type_str,
                    default_str=child.default_str,
                    description=child.description,
                ):
                    print(line)

            # Separate blocks only when nested sections are involved
            if idx != len(grouped) - 1:
                next_has_children = bool(grouped[idx + 1][2])
                if has_children or next_has_children:
                    print()
        print()


def _format_field_lines(
    indent: str,
    label: str,
    type_name: str,
    default_str: str,
    description: str | None,
) -> list[str]:
    """Format a field line (with optional description) for help output."""
    line = f"{indent}{label}: {type_name}"
    if default_str:
        line += f" = {default_str}"

    if description:
        if len(line) < 40:
            padding = max(1, 42 - len(line))
            return [f"{line}{' ' * padding}# {description}"]
        return [line, f"{indent}  # {description}"]

    return [line]


def _get_nested(data: dict[str, Any], path: list[str]) -> Any:
    """Get nested value from dict, returns _NOT_FOUND if not present."""
    current = data
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return _NOT_FOUND
        current = current[key]
    return current


def _get_source(
    path: str,
    value: Any,
    defaults: dict[str, Any],
    recipe: dict[str, Any] | None,
    override_info: dict[str, Any],
) -> str:
    """Determine the source of a config value and format the annotation."""
    # Check if it was a CLI override
    if path in override_info:
        prev = override_info[path]
        if prev is _NOT_FOUND:
            return "override (new)"
        else:
            prev_str = _format_value(prev)
            return f"override ← {prev_str}"

    # Get default value for comparison
    parts = path.split(".")
    default_val = _get_nested(defaults, parts)

    # Check if it's in recipe AND different from default
    if recipe is not None:
        recipe_val = _get_nested(recipe, parts)
        if recipe_val is not _NOT_FOUND:
            # If recipe value differs from default, it's a recipe customization
            if recipe_val != default_val:
                return "recipe"
            # Otherwise, it matches default (just happens to be in recipe too)

    return "default"


def _format_value(value: Any) -> str:
    """Format a value for display."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return f'"{value}"'
    if isinstance(value, float):
        # Use scientific notation for very small/large numbers
        if value != 0 and (abs(value) < 0.001 or abs(value) >= 10000):
            return f"{value:.2e}"
        return str(value)
    return str(value)


def _print_with_sources(
    final: dict[str, Any],
    defaults: dict[str, Any],
    recipe: dict[str, Any] | None,
    override_info: dict[str, Any],
    prefix: str,
    indent: int,
) -> None:
    """Recursively print config dict with source annotations."""
    pad = "  " * indent

    for key, value in final.items():
        path = f"{prefix}.{key}" if prefix else key

        if isinstance(value, dict):
            print(f"{pad}{key}:")
            _print_with_sources(value, defaults, recipe, override_info, path, indent + 1)
        else:
            # Get source annotation
            source = _get_source(path, value, defaults, recipe, override_info)

            # Format value
            val_str = _format_value(value)

            # Calculate padding for alignment
            left_part = f"{pad}{key}: {val_str}"
            # Align source annotations
            padding = max(1, 45 - len(left_part))
            print(f"{left_part}{' ' * padding}({source})")
