"""Decorator-based CLI for Stryx.

The @stryx.cli decorator transforms a function into a full CLI:

    @stryx.cli(schema=Config)
    def main(cfg: Config):
        train(cfg)

    if __name__ == "__main__":
        main()

Commands:
    train.py                                 Run with schema defaults
    train.py lr=1e-4 batch=32                Run with overrides
    train.py new [name] [overrides...]       Save recipe (auto-names if no name)
    train.py new [name] --from <src> [ov]    Copy + modify recipe
    train.py run <name|path> [overrides...]  Run from recipe or file path
    train.py edit <name>                     Edit recipe in TUI
    train.py show [name|path] [overrides...] Show config with sources
    train.py list                            List all recipes
"""
from __future__ import annotations

import functools
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal, TypeVar, Union

from pydantic import BaseModel, ValidationError

from .utils import read_yaml, write_yaml, set_dotpath

T = TypeVar("T", bound=BaseModel)

# Reserved command names
COMMANDS = frozenset({"new", "run", "edit", "show", "list"})

# Sentinel for "value not found"
_NOT_FOUND = object()


def _is_path(arg: str) -> bool:
    """Check if argument looks like a file path rather than a recipe name.

    Returns True if arg contains path separators or has a YAML/JSON extension.
    """
    return "/" in arg or "\\" in arg or arg.endswith((".yaml", ".yml", ".json"))


# =============================================================================
# CLI Argument Parsing
# =============================================================================

Command = Literal["run", "new", "edit", "show", "list", "help"]


@dataclass
class ParsedArgs:
    """Parsed CLI arguments."""

    command: Command
    recipe: str | None = None
    config_path: Path | None = None
    from_recipe: str | None = None
    overrides: list[str] = field(default_factory=list)


def _parse_argv(argv: list[str]) -> ParsedArgs:
    """Parse CLI arguments into structured form.

    Handles all argument parsing in one place for consistency.
    """
    # No args → run with defaults
    if not argv:
        return ParsedArgs(command="run")

    first = argv[0]

    # --help / -h
    if first in ("--help", "-h"):
        return ParsedArgs(command="help")

    # list (no additional args)
    if first == "list":
        return ParsedArgs(command="list")

    # new [name] [--from <src>] [overrides...]
    # Name is optional - if first arg contains '=', it's an override and we auto-generate
    if first == "new":
        name: str | None = None
        from_recipe: str | None = None
        overrides: list[str] = []

        i = 1
        # Check if first arg is a name or an override
        if i < len(argv) and "=" not in argv[i] and argv[i] != "--from":
            name = argv[i]
            if name in COMMANDS:
                raise SystemExit(f"Cannot use reserved name '{name}' for recipe")
            i += 1

        while i < len(argv):
            if argv[i] == "--from":
                if i + 1 >= len(argv):
                    raise SystemExit("--from requires a source recipe name")
                from_recipe = argv[i + 1]
                i += 2
            else:
                overrides.append(argv[i])
                i += 1

        return ParsedArgs(
            command="new",
            recipe=name,  # None means auto-generate
            from_recipe=from_recipe,
            overrides=overrides,
        )

    # run <name|path> [overrides...]
    if first == "run":
        if len(argv) < 2:
            raise SystemExit("'run' requires a recipe name or path")
        target = argv[1]
        if _is_path(target):
            return ParsedArgs(
                command="run",
                config_path=Path(target),
                overrides=argv[2:],
            )
        return ParsedArgs(
            command="run",
            recipe=target,
            overrides=argv[2:],
        )

    # edit <name>
    if first == "edit":
        if len(argv) < 2:
            raise SystemExit("'edit' requires a recipe name")
        return ParsedArgs(command="edit", recipe=argv[1])

    # show [name|path] [overrides...]
    if first == "show":
        recipe: str | None = None
        config_path: Path | None = None
        overrides = []

        i = 1
        while i < len(argv):
            arg = argv[i]
            if "=" in arg:
                overrides.append(arg)
            elif recipe is None and config_path is None:
                # First non-override arg is the target
                if _is_path(arg):
                    config_path = Path(arg)
                else:
                    recipe = arg
            else:
                raise SystemExit(f"Unexpected argument: {arg}")
            i += 1

        return ParsedArgs(
            command="show",
            recipe=recipe,
            config_path=config_path,
            overrides=overrides,
        )

    # Default: treat all args as overrides → run
    if "=" not in first:
        raise SystemExit(
            f"Unknown command '{first}'.\n"
            f"Did you mean: run {first}  (to run a recipe)\n"
            f"Or use overrides like: {first}=value"
        )

    return ParsedArgs(command="run", overrides=argv)


def cli(
    schema: type[T] | None = None,
    recipes_dir: str | Path = "configs",
) -> Callable[[Callable[[T], Any]], Callable[[], Any]]:
    """Decorator that adds Stryx CLI to a function.

    Args:
        schema: Pydantic model class for config. If None, inferred from type hints.
        recipes_dir: Directory for storing recipes (default: configs)

    Example:
        @stryx.cli(schema=Config)
        def main(cfg: Config):
            train(cfg)

        # Or with type hint inference:
        @stryx.cli()
        def main(cfg: Config):
            train(cfg)
    """
    recipes_path = Path(recipes_dir)

    def decorator(func: Callable[[T], Any]) -> Callable[[], Any]:
        nonlocal schema

        # Infer schema from type hints if not provided
        if schema is None:
            inferred = _infer_schema(func)
            if inferred is None:
                raise TypeError(
                    f"Could not infer schema for {func.__name__}. "
                    "Either provide schema= argument or add type hint like: def main(cfg: Config)"
                )
            resolved_schema = inferred
        else:
            resolved_schema = schema

        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # If called with a config directly, just run the function
            if args and isinstance(args[0], resolved_schema):
                return func(*args, **kwargs)
            if kwargs:
                return func(*args, **kwargs)

            # Otherwise, parse CLI and dispatch
            argv = sys.argv[1:]
            return _dispatch(func, resolved_schema, recipes_path, argv)

        # Attach metadata for introspection
        wrapper._stryx_schema = resolved_schema  # type: ignore
        wrapper._stryx_recipes_dir = recipes_path  # type: ignore

        return wrapper

    return decorator


def _infer_schema(func: Callable) -> type[T] | None:
    """Infer schema type from function's type hints."""
    hints = getattr(func, "__annotations__", {})
    for param_type in hints.values():
        if isinstance(param_type, type) and issubclass(param_type, BaseModel):
            return param_type
    return None


def _dispatch(
    func: Callable[[T], Any],
    schema: type[T],
    recipes_dir: Path,
    argv: list[str],
) -> Any:
    """Parse argv and dispatch to appropriate handler."""
    args = _parse_argv(argv)

    if args.command == "help":
        _print_help(schema)
        return

    if args.command == "list":
        return _cmd_list(recipes_dir)

    if args.command == "new":
        return _cmd_new(schema, recipes_dir, args)

    if args.command == "edit":
        return _cmd_edit(schema, recipes_dir, args.recipe)

    if args.command == "show":
        return _cmd_show(schema, recipes_dir, args)

    # command == "run"
    if args.config_path:
        cfg = _load_and_override(schema, args.config_path, args.overrides)
    elif args.recipe:
        recipe_path = recipes_dir / f"{args.recipe}.yaml"
        cfg = _load_and_override(schema, recipe_path, args.overrides)
    else:
        cfg = _build_config(schema, args.overrides)

    return func(cfg)


# =============================================================================
# Command Handlers
# =============================================================================

def _cmd_new(schema: type[T], recipes_dir: Path, args: ParsedArgs) -> Path:
    """Handle: new [name] [--from <src>] [overrides...]

    If name is not provided, generates sequential name (exp_001, exp_002, etc.)
    with file locking to prevent race conditions.
    """
    from filelock import FileLock

    # Build config first (before locking, since this doesn't need the lock)
    from_path: Path | None = None
    if args.from_recipe:
        from_path = recipes_dir / f"{args.from_recipe}.yaml"
        if not from_path.exists():
            raise SystemExit(f"Source recipe not found: {from_path}")
        cfg = _load_and_override(schema, from_path, args.overrides)
    else:
        cfg = _build_config(schema, args.overrides)

    # Prepare payload
    payload: dict[str, Any] = {
        "__stryx__": {
            "schema": f"{schema.__module__}:{schema.__name__}",
            "created_at": datetime.now(tz=timezone.utc).isoformat(),
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
        print(f"Recipe saved: {out_path}")
        return out_path

    # Auto-generate sequential name with locking
    lock_path = recipes_dir / ".stryx.lock"
    with FileLock(lock_path):
        name = _next_sequential_name(recipes_dir)
        out_path = recipes_dir / f"{name}.yaml"
        write_yaml(out_path, payload)

    print(f"Recipe saved: {out_path}")
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


def _cmd_edit(schema: type[T], recipes_dir: Path, name: str) -> None:
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


def _cmd_show(schema: type[T], recipes_dir: Path, args: ParsedArgs) -> None:
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
        recipe_data = _read_config_file(source_file)
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
    override_info: dict[str, Any] = {}  # path -> previous value
    for tok in args.overrides:
        key, _ = tok.split("=", 1)
        key = key.strip()
        # Get previous value before override
        prev = _get_nested(data, key.split("."))
        override_info[key] = prev
        _apply_override(data, tok)

    # Validate
    cfg = _validate_or_die(schema, data, "show")
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


def _cmd_list(recipes_dir: Path) -> None:
    """Handle: list - show all recipes in the recipes directory."""
    if not recipes_dir.exists():
        print(f"No recipes directory: {recipes_dir}")
        return

    recipes = sorted(recipes_dir.glob("*.yaml")) + sorted(recipes_dir.glob("*.yml"))

    if not recipes:
        print(f"No recipes in {recipes_dir}/")
        return

    print(f"Recipes ({recipes_dir}/):")
    print("-" * 40)

    for recipe_path in recipes:
        name = recipe_path.stem

        # Try to get metadata
        try:
            data = read_yaml(recipe_path)
            meta = data.get("__stryx__", {}) if isinstance(data, dict) else {}
            created = meta.get("created_at", "")
            if created:
                # Parse and format date
                from datetime import datetime
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
            print(f"  {name:<30} {date_str}")
        else:
            print(f"  {name}")


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


# =============================================================================
# Config Building
# =============================================================================

def _build_config(schema: type[T], overrides: list[str]) -> T:
    """Build config from schema defaults + overrides."""
    try:
        base = schema()
        data = base.model_dump(mode="python")
    except ValidationError as e:
        raise SystemExit(f"Schema has required fields without defaults:\n{e}")

    for tok in overrides:
        _apply_override(data, tok)

    return _validate_or_die(schema, data, "building config")


def _load_and_override(schema: type[T], path: Path, overrides: list[str]) -> T:
    """Load config from file, apply overrides, validate."""
    if not path.exists():
        raise SystemExit(f"Config not found: {path}")

    data = _read_config_file(path)

    # Strip stryx metadata
    if isinstance(data, dict):
        data = {k: v for k, v in data.items() if not k.startswith("__")}

    for tok in overrides:
        _apply_override(data, tok)

    return _validate_or_die(schema, data, f"loading {path.name}")


def _validate_or_die(schema: type[T], data: Any, context: str) -> T:
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


# =============================================================================
# Override Parsing
# =============================================================================

def _apply_override(data: dict[str, Any], tok: str) -> None:
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

    value = _parse_value(raw.strip())
    set_dotpath(data, key, value)


def _parse_value(s: str) -> Any:
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
    if s and s[0] in "{[":
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


# =============================================================================
# File I/O
# =============================================================================

def _read_config_file(path: Path) -> Any:
    """Read config from YAML or JSON."""
    suffix = path.suffix.lower()

    if suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))

    if suffix in (".yaml", ".yml"):
        return read_yaml(path)

    raise SystemExit(f"Unsupported format: {suffix} (use .yaml or .json)")


# =============================================================================
# Help
# =============================================================================

def _get_type_name(annotation: Any) -> str:
    """Get a readable type name from a type annotation."""
    import types
    from typing import Union, get_args, get_origin

    origin = get_origin(annotation)

    # Handle Optional[X] (Union[X, None])
    if origin is Union:
        args = get_args(annotation)
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1 and type(None) in args:
            return f"{_get_type_name(non_none[0])}?"
        # Union of multiple types
        return " | ".join(_get_type_name(a) for a in args if a is not type(None))

    # Handle X | Y (Python 3.10+ union)
    if isinstance(annotation, types.UnionType):
        args = get_args(annotation)
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1 and type(None) in args:
            return f"{_get_type_name(non_none[0])}?"
        return " | ".join(_get_type_name(a) for a in args if a is not type(None))

    # Handle list[X], dict[K, V], etc.
    if origin is list:
        args = get_args(annotation)
        if args:
            return f"list[{_get_type_name(args[0])}]"
        return "list"

    if origin is dict:
        args = get_args(annotation)
        if len(args) == 2:
            return f"dict[{_get_type_name(args[0])}, {_get_type_name(args[1])}]"
        return "dict"

    # Handle Literal["a", "b"]
    if origin is Literal:
        args = get_args(annotation)
        return " | ".join(repr(a) for a in args)

    # BaseModel subclass
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation.__name__

    # Simple types
    if hasattr(annotation, "__name__"):
        return annotation.__name__

    return str(annotation)


def _format_default(value: Any) -> str:
    """Format a default value for display."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        if len(value) > 30:
            return f'"{value[:27]}..."'
        return f'"{value}"'
    if isinstance(value, float):
        if value != 0 and (abs(value) < 0.001 or abs(value) >= 10000):
            return f"{value:.2e}"
        return str(value)
    if isinstance(value, (list, dict)):
        s = str(value)
        if len(s) > 30:
            return s[:27] + "..."
        return s
    return str(value)


def _extract_schema_fields(
    schema: type[BaseModel],
    prefix: str = "",
) -> list[tuple[str, str, str, str | None]]:
    """Extract fields from schema recursively.

    Returns list of (path, type_name, default_str, description).
    """
    from pydantic.fields import FieldInfo

    fields = []

    for name, field_info in schema.model_fields.items():
        path = f"{prefix}.{name}" if prefix else name
        annotation = field_info.annotation
        description = field_info.description

        # Get default value
        default = field_info.default
        if default is None and field_info.default_factory is not None:
            # Try to get value from factory
            try:
                default = field_info.default_factory()
            except Exception:
                default = None

        # Check if this is a nested BaseModel
        inner_type = annotation
        # Unwrap Optional
        origin = getattr(annotation, "__origin__", None)
        if origin is Union:
            from typing import get_args
            args = get_args(annotation)
            non_none = [a for a in args if a is not type(None)]
            if len(non_none) == 1:
                inner_type = non_none[0]

        if isinstance(inner_type, type) and issubclass(inner_type, BaseModel):
            # Nested model - recurse
            nested_fields = _extract_schema_fields(inner_type, prefix=path)
            fields.extend(nested_fields)
        else:
            # Leaf field
            type_name = _get_type_name(annotation)
            default_str = _format_default(default) if default is not None else ""
            fields.append((path, type_name, default_str, description))

    return fields


def _print_help(schema: type[BaseModel]) -> None:
    """Print CLI help message."""
    prog = Path(sys.argv[0]).name
    print(f"""Stryx - Typed Configuration Compiler

Usage:
  {prog}                                 Run with defaults
  {prog} <key>=<value> ...               Run with overrides
  {prog} new [name] [overrides...]       Create recipe (auto-names if no name)
  {prog} new [name] --from <src> [ov]    Copy + modify recipe
  {prog} run <name|path> [overrides...]  Run from recipe or file
  {prog} edit <name>                     Edit recipe (TUI)
  {prog} show [name|path] [overrides...] Show config with sources
  {prog} list                            List all recipes

Examples:
  {prog} lr=1e-4 train.steps=1000
  {prog} new my_exp lr=1e-4
  {prog} run my_exp
  {prog} show my_exp lr=1e-5             # See where values come from

Schema: {schema.__module__}:{schema.__name__}
""")

    # Print schema fields
    fields = _extract_schema_fields(schema)
    if fields:
        print("Fields:")
        for path, type_name, default_str, description in fields:
            # Build the field line
            if default_str:
                line = f"  {path}: {type_name} = {default_str}"
            else:
                line = f"  {path}: {type_name}"

            # Add description if present
            if description:
                # Align description or put on same line if short
                if len(line) < 40:
                    padding = " " * (42 - len(line))
                    print(f"{line}{padding}# {description}")
                else:
                    print(line)
                    print(f"      # {description}")
            else:
                print(line)
        print()
