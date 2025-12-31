import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, Optional

import typer
from filelock import FileLock
import petname

from stryx.config_builder import (
    apply_override,
    build_config,
    load_and_override,
    read_config_file,
    validate_or_die,
)
from stryx.lifecycle import RunContext, get_rank, record_run_manifest
from stryx.run_id import derive_run_id
from stryx.schema import FieldInfo, extract_fields
from stryx.utils import (
    Ctx,
    flatten_config,
    get_next_sequential_name,
    read_yaml,
    resolve_recipe_path,
    save_recipe,
)


def cmd_new(
    ctx: typer.Context,
    recipe: Annotated[
        Optional[str],
        typer.Argument(
            metavar="recipe",
            help="Optional name for the recipe. If omitted, an auto-incrementing name like 'exp_001' is used.",
        ),
    ] = None,
    overrides: Annotated[
        Optional[list[str]],
        typer.Argument(
            metavar="overrides",
            help="Configuration overrides in 'key=value' format.",
        ),
    ] = None,
    message: Annotated[
        Optional[str],
        typer.Option("--message", "-m", help="Short description to store in the recipe metadata."),
    ] = None,
    force: Annotated[
        bool,
        typer.Option("--force", help="Overwrite the recipe if it already exists."),
    ] = False,
) -> Path:
    """Create a fresh experiment recipe from defaults."""
    c: Ctx = ctx.obj
    overrides = overrides or []

    # Handle case where recipe name is omitted but overrides are provided
    # e.g. `stryx new optim.lr=1` -> recipe="optim.lr=1", overrides=[]
    if recipe and "=" in recipe:
        overrides = [recipe] + overrides
        recipe = None

    cfg = build_config(c.schema, overrides)
    cfg_data = cfg.model_dump(mode="python")

    # Create directory
    c.configs_dir.mkdir(parents=True, exist_ok=True)

    try:
        if recipe:
            name = recipe
            if "." not in name:
                name = f"{name}.yaml"
            out_path = c.configs_dir / name

            save_recipe(
                path=out_path,
                cfg_data=cfg_data,
                schema_cls=c.schema,
                overrides=overrides,
                description=message,
                force=force,
                kind="canonical",
            )
        else:
            # Auto-generate name with lock
            lock_path = c.configs_dir / ".stryx.lock"
            with FileLock(lock_path):
                name = get_next_sequential_name(c.configs_dir)
                out_path = c.configs_dir / f"{name}.yaml"

                save_recipe(
                    path=out_path,
                    cfg_data=cfg_data,
                    schema_cls=c.schema,
                    overrides=overrides,
                    description=message,
                    force=False,
                    kind="canonical",
                )

    except FileExistsError as e:
        print(f"Error: {e} Use --force to overwrite.")
        raise typer.Exit(code=1)

    print(f"Created recipe: {out_path}")
    return out_path


def cmd_fork(
    ctx: typer.Context,
    source: Annotated[str, typer.Argument(metavar="source", help="Source recipe name or file path.")],
    name: Annotated[str, typer.Argument(metavar="name", help="Name for the new forked recipe.")],
    overrides: Annotated[
        Optional[list[str]],
        typer.Argument(metavar="overrides", help="Configuration overrides in 'key=value' format."),
    ] = None,
    message: Annotated[
        Optional[str],
        typer.Option("--message", "-m", help="Short description for the new recipe."),
    ] = None,
    force: Annotated[
        bool,
        typer.Option("--force", help="Overwrite the destination recipe if it exists."),
    ] = False,
) -> Path:
    """Fork an existing recipe and apply modifications."""
    c: Ctx = ctx.obj
    overrides = overrides or []

    # Resolve and Load Source
    try:
        from_path = resolve_recipe_path(c.configs_dir, source)
    except FileNotFoundError:
        print(f"Source recipe not found: {source}")
        raise typer.Exit(code=1)

    cfg = load_and_override(c.schema, from_path, overrides)
    cfg_data = cfg.model_dump(mode="python")

    # Determine output path
    if "." not in name:
        name = f"{name}.yaml"
    out_path = c.configs_dir / name

    c.configs_dir.mkdir(parents=True, exist_ok=True)

    try:
        save_recipe(
            path=out_path,
            cfg_data=cfg_data,
            schema_cls=c.schema,
            overrides=overrides,
            description=message,
            force=force,
            kind="canonical",
            source=str(source),
        )
    except FileExistsError as e:
        print(f"Error: {e} Use --force to overwrite.")
        raise typer.Exit(code=1)

    print(f"Forked recipe: {out_path}")
    return out_path


def cmd_run(
    ctx: typer.Context,
    target: Annotated[str, typer.Argument(metavar="recipe", help="Recipe name or path to execute.")],
    run_id: Annotated[
        Optional[str],
        typer.Option("--run-id", help="Manually specify a unique ID for this run."),
    ] = None,
) -> Any:
    """Execute a specific experiment recipe exactly as defined."""
    c: Ctx = ctx.obj
    try:
        path = resolve_recipe_path(c.configs_dir, target)
    except FileNotFoundError:
        print(f"Recipe not found: {target}")
        raise typer.Exit(code=1)

    # Load config (run is strict, no overrides)
    cfg = load_and_override(c.schema, path, [])

    return _execute(
        c,
        cfg,
        source={"kind": "file", "path": str(path), "name": path.stem},
        overrides=[],
        run_id_override=run_id,
    )


def cmd_try(
    ctx: typer.Context,
    target: Annotated[
        Optional[str],
        typer.Argument(metavar="recipe", help="Optional base recipe to start from."),
    ] = None,
    overrides: Annotated[
        Optional[list[str]],
        typer.Argument(metavar="overrides", help="Configuration overrides in 'key=value' format."),
    ] = None,
    message: Annotated[
        Optional[str],
        typer.Option("--message", "-m", help="Short description for the scratch metadata."),
    ] = None,
    run_id: Annotated[
        Optional[str],
        typer.Option("--run-id", help="Manually specify a unique ID for this run."),
    ] = None,
) -> Any:
    """Run an experiment variant without saving a permanent recipe (saved to scratches)."""
    c: Ctx = ctx.obj
    overrides = overrides or []

    # If target token looks like an override (contains '='), shift it
    if target and "=" in target:
        overrides = [target] + overrides
        target = None

    # Resolve source
    if target:
        try:
            from_path = resolve_recipe_path(c.configs_dir, target)
            cfg = load_and_override(c.schema, from_path, overrides)
            lineage = target
            name_label = from_path.stem
        except FileNotFoundError:
            print(f"Source recipe not found: {target}")
            raise typer.Exit(code=1)
    else:
        cfg = build_config(c.schema, overrides)
        lineage = None
        name_label = "defaults"

    # Generate scratch name
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    name = f"{timestamp}_{petname.generate(2)}"

    out_dir = c.configs_dir / "scratches"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}.yaml"

    # Save scratch
    save_recipe(
        path=out_path,
        cfg_data=cfg.model_dump(mode="python"),
        schema_cls=c.schema,
        overrides=overrides,
        kind="scratch",
        source=lineage,
        description=message,
        force=False,
    )

    print(f"Running scratch: scratches/{name}.yaml")

    return _execute(
        c,
        cfg,
        source={"kind": "scratch", "path": str(out_path), "name": name_label},
        overrides=overrides,
        run_id_override=run_id,
    )


def _execute(
    c: Ctx,
    cfg: Any,
    source: dict[str, Any],
    overrides: list[str],
    run_id_override: str | None = None,
) -> Any:
    """Orchestrate the execution lifecycle."""
    # 1. Derive Run ID
    run_id = derive_run_id(
        label=source.get("name") or "run", run_id_override=run_id_override
    )

    # 2. Setup Manifest (only on rank 0)
    rank = get_rank()
    if rank == 0:
        record_run_manifest(c, cfg, run_id, source, overrides)

    manifest_path = c.runs_dir / run_id / "manifest.yaml"

    # 3. Execute User Function
    with RunContext(manifest_path, rank) as run_ctx:
        result = c.func(cfg)
        run_ctx.record_result(result)
        return result


def cmd_list_configs(ctx: typer.Context) -> None:
    """List all saved experiment recipes and scratches."""
    c: Ctx = ctx.obj
    if not c.configs_dir.exists():
        print(f"No recipes found in {c.configs_dir}")
        return

    # Collect all recipes
    canonicals = sorted(c.configs_dir.glob("*.yaml")) + sorted(
        c.configs_dir.glob("*.yml")
    )

    scratches_dir = c.configs_dir / "scratches"
    scratches = []
    if scratches_dir.exists():
        scratches = sorted(scratches_dir.glob("*.yaml"), reverse=True)

    all_recipes = canonicals + scratches
    if not all_recipes:
        print("No recipes found.")
        return

    rows = []
    all_keys = set()

    for p in all_recipes:
        try:
            data = read_yaml(p)
            meta = data.get("__stryx__", {}) if isinstance(data, dict) else {}
            created = meta.get("created_at", "")
            if created:
                created = created[:16].replace("T", " ")  # Simplified ISO format

            # Clean data for interesting columns
            if isinstance(data, dict):
                clean = {k: v for k, v in data.items() if not k.startswith("__")}
            else:
                clean = {}

            flat = flatten_config(clean)

            is_scratch = "scratches" in p.parts
            name = f"scratches/{p.stem}" if is_scratch else p.stem

            row = {"Name": name, "Created": created, **flat}
            rows.append(row)
            all_keys.update(flat.keys())
        except Exception:
            continue

    _print_smart_table(rows, ["Name", "Created"], all_keys)


def cmd_list_runs(
    ctx: typer.Context,
    status: Annotated[
        str,
        typer.Option(help="Filter runs by status (any, ok, failed)."),
    ] = "any",
) -> None:
    """List execution history and run statuses."""
    c: Ctx = ctx.obj
    if not c.runs_dir.exists():
        print(f"No runs found in {c.runs_dir}")
        return

    rows = []
    all_keys = set()

    for p in c.runs_dir.iterdir():
        if not p.is_dir():
            continue
        manifest_path = p / "manifest.yaml"
        if not manifest_path.exists():
            continue

        try:
            data = read_yaml(manifest_path)

            # Extract key info
            run_id = data.get("run_id", p.name)
            run_status = data.get("status", "UNKNOWN")
            
            # Filter
            if status != "any":
                if status == "ok" and run_status != "COMPLETED":
                    continue
                if status == "failed" and run_status != "FAILED":
                    continue
            
            created = data.get("created_at", "")[:16].replace("T", " ")

            # Config subset?
            config = data.get("config", {})
            flat_cfg = flatten_config(config)

            row = {"Run ID": run_id, "Status": run_status, "Created": created, **flat_cfg}
            rows.append(row)
            all_keys.update(flat_cfg.keys())
        except Exception:
            continue

    # Sort by created desc
    rows.sort(key=lambda x: x.get("Created", ""), reverse=True)

    _print_smart_table(rows, ["Run ID", "Status", "Created"], all_keys)


def _print_smart_table(
    rows: list[dict], fixed_cols: list[str], potential_cols: set[str]
) -> None:
    """Print a table with fixed columns + interesting variant columns."""
    if not rows:
        print("No items.")
        return

    # Identify interesting columns (variance > 1)
    interesting_keys = []
    for key in sorted(potential_cols):
        values = set()
        for row in rows:
            val = str(row.get(key, ""))
            values.add(val)
        if len(values) > 1:
            interesting_keys.append(key)

    columns = fixed_cols + interesting_keys

    # Calculate widths
    widths = {col: len(col) for col in columns}
    for row in rows:
        for col in columns:
            val = str(row.get(col, ""))
            widths[col] = max(widths[col], len(val))

    # Header
    header = "  ".join(f"{col:<{widths[col]}}" for col in columns)
    print(header)
    print("-" * len(header))

    for row in rows:
        line = "  ".join(f"{str(row.get(col, '')):<{widths[col]}}" for col in columns)
        print(line)


def cmd_edit(
    ctx: typer.Context,
    recipe: Annotated[str, typer.Argument(metavar="recipe", help="Name of the recipe to edit.")],
) -> None:
    """Open the interactive TUI editor for a recipe."""
    from stryx.tui import PydanticConfigTUI
    c: Ctx = ctx.obj

    try:
        recipe_path = resolve_recipe_path(c.configs_dir, recipe)
    except FileNotFoundError:
        print(f"Recipe not found: {recipe}\nCreate it first with: new {recipe}")
        raise typer.Exit(code=1)

    tui = PydanticConfigTUI(c.schema, recipe_path)
    tui.run()


def cmd_show(
    ctx: typer.Context,
    target: Annotated[
        Optional[str],
        typer.Argument(metavar="recipe", help="Recipe to display (defaults to schema defaults)."),
    ] = None,
    overrides: Annotated[
        Optional[list[str]],
        typer.Argument(metavar="overrides", help="Temporary overrides to apply before displaying."),
    ] = None,
) -> None:
    """Show the resolved configuration with source annotations (default vs recipe vs override)."""
    c: Ctx = ctx.obj
    overrides = overrides or []

    # Handle case where target is omitted but overrides are provided
    if target and "=" in target:
        overrides = [target] + overrides
        target = None

    # Get schema defaults
    try:
        defaults_instance = c.schema()
        schema_defaults = defaults_instance.model_dump(mode="python")
    except Exception as e:
        print(f"Schema has required fields without defaults:\n{e}")
        raise typer.Exit(code=1)

    # Determine source file
    source_name = "defaults"
    recipe_data: dict[str, Any] | None = None

    if target:
        try:
            path = resolve_recipe_path(c.configs_dir, target)
            recipe_data = read_config_file(path)
            # Strip metadata
            if isinstance(recipe_data, dict):
                recipe_data = {
                    k: v for k, v in recipe_data.items() if not k.startswith("__")
                }
            source_name = path.stem
        except FileNotFoundError:
            print(f"Config not found: {target}")
            raise typer.Exit(code=1)

    # Build the config data (before validation, to track sources)
    if recipe_data is not None:
        data = dict(recipe_data)
    else:
        data = dict(schema_defaults)

    # Track override paths and their previous values
    override_info: dict[str, Any] = {}  # path → previous value
    for tok in overrides:
        try:
            key, _ = tok.split("=", 1)
            key = key.strip()
            # Get previous value before override
            prev = _get_nested(data, key.split("."))
            override_info[key] = prev
            apply_override(data, tok)
        except ValueError:
            print(f"Invalid override format: {tok}")
            raise typer.Exit(code=1)

    # Validate
    cfg = validate_or_die(c.schema, data, "show")
    final_data = cfg.model_dump(mode="python")

    # Print header
    header_parts = ["Config"]
    if source_name != "defaults":
        header_parts.append(f"recipe: {source_name}")
    if overrides:
        header_parts.append(
            f"{len(overrides)} override{'s' if len(overrides) > 1 else ''}"
        )
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


def cmd_diff(
    ctx: typer.Context,
    recipe_a: Annotated[str, typer.Argument(metavar="recipe_a", help="First recipe to compare.")],
    recipe_b: Annotated[
        Optional[str],
        typer.Argument(metavar="recipe_b", help="Second recipe to compare (defaults to schema defaults)."),
    ] = None,
) -> None:
    """Compare two experiment recipes and highlight differences."""
    c: Ctx = ctx.obj

    # Load both configs
    try:
        path_a = resolve_recipe_path(c.configs_dir, recipe_a)
        cfg_a = read_config_file(path_a)
    except FileNotFoundError:
        print(f"Recipe not found: {recipe_a}")
        raise typer.Exit(code=1)

    if recipe_b:
        try:
            path_b = resolve_recipe_path(c.configs_dir, recipe_b)
            cfg_b = read_config_file(path_b)
            name_b = recipe_b
        except FileNotFoundError:
            print(f"Recipe not found: {recipe_b}")
            raise typer.Exit(code=1)
    else:
        # Diff against defaults
        base = c.schema()
        cfg_b = base.model_dump(mode="python")
        name_b = "(defaults)"

    # Strip metadata
    if isinstance(cfg_a, dict):
        cfg_a = {k: v for k, v in cfg_a.items() if not k.startswith("__")}
    if isinstance(cfg_b, dict):
        cfg_b = {k: v for k, v in cfg_b.items() if not k.startswith("__")}

    flat_a = flatten_config(cfg_a)
    flat_b = flatten_config(cfg_b)

    all_keys = sorted(set(flat_a.keys()) | set(flat_b.keys()))

    print(f"Diff: {recipe_a} vs {name_b}")
    print("-" * 60)

    has_diff = False

    for key in all_keys:
        val_a = flat_a.get(key, _NOT_FOUND)
        val_b = flat_b.get(key, _NOT_FOUND)

        if val_a == val_b:
            continue

        has_diff = True
        if val_a is _NOT_FOUND:
            print(f"+ {key}: {val_b}")
        elif val_b is _NOT_FOUND:
            print(f"- {key}: {val_a}")
        else:
            print(f"~ {key}: {val_a} -> {val_b}")

    if not has_diff:
        print("No differences found.")


def cmd_schema(
    ctx: typer.Context,
    json_out: Annotated[
        bool,
        typer.Option("--json", help="Output the schema in JSON format."),
    ] = False,
) -> None:
    """Display the configuration schema and field documentation."""
    c: Ctx = ctx.obj
    if json_out:
        print(json.dumps(c.schema.model_json_schema(), indent=2))
        return

    print(f"Schema: {c.schema.__module__}:{c.schema.__name__}")

    fields = extract_fields(c.schema)
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
                label = (
                    child_path[len(group) + 1 :]
                    if child_path.startswith(f"{group}.")
                    else child_path
                )
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


# ============================================================================ 
# Helpers
# ============================================================================ 

_NOT_FOUND = object()


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
            if recipe_val != default_val:
                return "recipe"

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
        if value != 0 and (abs(value) < 0.001 or abs(value) >= 10000):
            return f"{value:.2e}"
        return str(value)
    return str(value)


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
            _print_with_sources(
                value, defaults, recipe, override_info, path, indent + 1
            )
        else:
            source = _get_source(path, value, defaults, recipe, override_info)
            val_str = _format_value(value)
            left_part = f"{pad}{key}: {val_str}"
            padding = max(1, 45 - len(left_part))
            print(f"{left_part}{' ' * padding}({source})")