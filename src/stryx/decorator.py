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
    train.py new <name> [overrides...]       Save recipe
    train.py new <name> --from <src> [ov]    Copy + modify recipe
    train.py run <name> [overrides...]       Run from recipe
    train.py edit <name>                     Edit recipe in TUI
    train.py --config <path> [overrides...]  Run from explicit path
"""
from __future__ import annotations

import functools
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, TypeVar

from pydantic import BaseModel, ValidationError

from .utils import read_yaml, write_yaml, set_dotpath

T = TypeVar("T", bound=BaseModel)

# Reserved command names
COMMANDS = frozenset({"new", "run", "edit", "show", "list"})

# Sentinel for "value not found"
_NOT_FOUND = object()


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

    # No args → run with defaults
    if not argv:
        cfg = _build_config(schema, [])
        return func(cfg)

    first = argv[0]

    # --help / -h
    if first in ("--help", "-h"):
        _print_help(schema)
        return

    # --config <path> [overrides...]
    if first in ("--config", "-c"):
        if len(argv) < 2:
            raise SystemExit("--config requires a path")
        cfg = _load_and_override(schema, Path(argv[1]), argv[2:])
        return func(cfg)

    # new <name> [--from <src>] [overrides...]
    if first == "new":
        return _cmd_new(schema, recipes_dir, argv[1:])

    # run <name> [overrides...]
    if first == "run":
        if len(argv) < 2:
            raise SystemExit("'run' requires a recipe name")
        recipe_path = recipes_dir / f"{argv[1]}.yaml"
        cfg = _load_and_override(schema, recipe_path, argv[2:])
        return func(cfg)

    # edit <name>
    if first == "edit":
        if len(argv) < 2:
            raise SystemExit("'edit' requires a recipe name")
        return _cmd_edit(schema, recipes_dir, argv[1])

    # show [recipe] [--config path] [overrides...]
    if first == "show":
        return _cmd_show(schema, recipes_dir, argv[1:])

    # list
    if first == "list":
        return _cmd_list(recipes_dir)

    # Default: treat all args as overrides → run
    # Check that args look like overrides (contain '=')
    if "=" not in first:
        raise SystemExit(
            f"Unknown command '{first}'.\n"
            f"Did you mean: run {first}  (to run a recipe)\n"
            f"Or use overrides like: {first}=value"
        )

    cfg = _build_config(schema, argv)
    return func(cfg)


# =============================================================================
# Command Handlers
# =============================================================================

def _cmd_new(schema: type[T], recipes_dir: Path, argv: list[str]) -> Path:
    """Handle: new <name> [--from <src>] [overrides...]"""
    if not argv:
        raise SystemExit("'new' requires a recipe name")

    name = argv[0]
    rest = argv[1:]

    # Check for reserved names
    if name in COMMANDS:
        raise SystemExit(f"Cannot use reserved name '{name}' for recipe")

    # Parse --from flag
    from_recipe: Path | None = None
    overrides: list[str] = []

    i = 0
    while i < len(rest):
        if rest[i] == "--from":
            if i + 1 >= len(rest):
                raise SystemExit("--from requires a source recipe name")
            from_recipe = recipes_dir / f"{rest[i + 1]}.yaml"
            i += 2
        else:
            overrides.append(rest[i])
            i += 1

    # Build config
    if from_recipe:
        if not from_recipe.exists():
            raise SystemExit(f"Source recipe not found: {from_recipe}")
        cfg = _load_and_override(schema, from_recipe, overrides)
    else:
        cfg = _build_config(schema, overrides)

    # Write recipe
    recipes_dir.mkdir(parents=True, exist_ok=True)
    out_path = recipes_dir / f"{name}.yaml"

    payload: dict[str, Any] = {
        "__stryx__": {
            "schema": f"{schema.__module__}:{schema.__name__}",
            "created_at": datetime.now(tz=timezone.utc).isoformat(),
        }
    }
    if from_recipe:
        payload["__stryx__"]["from"] = from_recipe.stem
    if overrides:
        payload["__stryx__"]["overrides"] = overrides

    payload.update(cfg.model_dump(mode="python"))

    write_yaml(out_path, payload)
    print(f"Recipe saved: {out_path}")
    return out_path


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


def _cmd_show(schema: type[T], recipes_dir: Path, argv: list[str]) -> None:
    """Handle: show [recipe] [--config path] [overrides...]

    Pretty-prints the config with source annotations showing where each value
    comes from: (default), (recipe), or (override ← previous_value).
    """
    # Parse arguments
    recipe_path: Path | None = None
    config_path: Path | None = None
    overrides: list[str] = []

    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg in ("--config", "-c"):
            if i + 1 >= len(argv):
                raise SystemExit("--config requires a path")
            config_path = Path(argv[i + 1])
            i += 2
        elif "=" in arg:
            overrides.append(arg)
            i += 1
        else:
            # Assume it's a recipe name
            recipe_path = recipes_dir / f"{arg}.yaml"
            i += 1

    # Get schema defaults
    try:
        defaults_instance = schema()
        schema_defaults = defaults_instance.model_dump(mode="python")
    except ValidationError as e:
        raise SystemExit(f"Schema has required fields without defaults:\n{e}")

    # Determine source file
    source_file: Path | None = config_path or recipe_path
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
    for tok in overrides:
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
    if overrides:
        header_parts.append(f"{len(overrides)} override{'s' if len(overrides) > 1 else ''}")
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

def _print_help(schema: type[BaseModel]) -> None:
    """Print CLI help message."""
    prog = Path(sys.argv[0]).name
    print(f"""Stryx - Typed Configuration Compiler

Usage:
  {prog}                                 Run with defaults
  {prog} <key>=<value> ...               Run with overrides
  {prog} new <name> [overrides...]       Create recipe
  {prog} new <name> --from <src> [ov]    Copy + modify recipe
  {prog} run <name> [overrides...]       Run from recipe
  {prog} edit <name>                     Edit recipe (TUI)
  {prog} show [name] [overrides...]      Show config with sources
  {prog} list                            List all recipes
  {prog} --config <path> [overrides...]  Run from file

Examples:
  {prog} lr=1e-4 train.steps=1000
  {prog} new my_exp lr=1e-4
  {prog} new my_exp_v2 --from my_exp batch_size=64
  {prog} run my_exp
  {prog} edit my_exp
  {prog} show my_exp lr=1e-5            # See where values come from

Schema: {schema.__module__}:{schema.__name__}
""")
