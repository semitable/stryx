import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from stryx.config_builder import (
    apply_override,
    build_config,
    load_and_override,
    read_config_file,
    validate_or_die,
)
from stryx.context import Ctx
from stryx.lifecycle import RunContext, get_rank, record_run_manifest
from stryx.run_id import derive_run_id
from stryx.schema import FieldInfo, extract_fields
from stryx.utils import (
    get_next_sequential_name,
    resolve_recipe_path,
    flatten_config,
    save_recipe,
    read_yaml
)


def cmd_new(ctx: Ctx, ns: argparse.Namespace) -> Path:
    """Handle: new [name] [overrides...] - create from defaults."""
    from filelock import FileLock

    cfg = build_config(ctx.schema, ns.overrides)
    cfg_data = cfg.model_dump(mode="python")

    # Create directory
    ctx.configs_dir.mkdir(parents=True, exist_ok=True)

    try:
        if ns.recipe:
            name = ns.recipe
            if "." not in name:
                name = f"{name}.yaml"
            out_path = ctx.configs_dir / name

            save_recipe(
                path=out_path,
                cfg_data=cfg_data,
                schema_cls=ctx.schema,
                overrides=ns.overrides,
                description=ns.message,
                force=ns.force,
                kind="canonical",
            )
        else:
            # Auto-generate name with lock
            lock_path = ctx.configs_dir / ".stryx.lock"
            with FileLock(lock_path):
                name = get_next_sequential_name(ctx.configs_dir)
                out_path = ctx.configs_dir / f"{name}.yaml"

                save_recipe(
                    path=out_path,
                    cfg_data=cfg_data,
                    schema_cls=ctx.schema,
                    overrides=ns.overrides,
                    description=ns.message,
                    force=False,
                    kind="canonical",
                )

    except FileExistsError as e:
        raise SystemExit(f"Error: {e} Use --force to overwrite.")

    print(f"Created recipe: {out_path}")
    return out_path


def cmd_fork(ctx: Ctx, ns: argparse.Namespace) -> Path:
    """Handle: fork <source> <name> [overrides...]"""

    # Resolve and Load Source
    try:
        from_path = resolve_recipe_path(ctx.configs_dir, ns.source)
    except FileNotFoundError:
        raise SystemExit(f"Source recipe not found: {ns.source}")

    cfg = load_and_override(ctx.schema, from_path, ns.overrides)
    cfg_data = cfg.model_dump(mode="python")

    # Determine output path
    name = ns.name
    if "." not in name:
        name = f"{name}.yaml"
    out_path = ctx.configs_dir / name

    ctx.configs_dir.mkdir(parents=True, exist_ok=True)

    try:
        save_recipe(
            path=out_path,
            cfg_data=cfg_data,
            schema_cls=ctx.schema,
            overrides=ns.overrides,
            description=ns.message,
            force=ns.force,
            kind="canonical",
            source=str(ns.source),
        )
    except FileExistsError as e:
        raise SystemExit(f"Error: {e} Use --force to overwrite.")

    print(f"Forked recipe: {out_path}")
    return out_path


def cmd_run(ctx: Ctx, ns: argparse.Namespace) -> Any:
    """Handle: run <target> - run a recipe exactly."""
    try:
        path = resolve_recipe_path(ctx.configs_dir, ns.target)
    except FileNotFoundError:
        raise SystemExit(f"Recipe not found: {ns.target}")

    # Load config (run is strict, no overrides)
    cfg = load_and_override(ctx.schema, path, [])

    return _execute(
        ctx,
        cfg,
        source={"kind": "file", "path": str(path), "name": path.stem},
        overrides=[],
        run_id_override=getattr(ns, "run_id", None),
    )


def cmd_try(ctx: Ctx, ns: argparse.Namespace) -> Any:
    """Handle: try [target] [overrides...] - run experimental variant."""
    import petname
    
    target_token = ns.target
    overrides = ns.overrides
    
    # If target token looks like an override (contains '='), shift it
    if target_token and "=" in target_token:
        overrides = [target_token] + overrides
        target_token = None
        
    # Resolve source
    if target_token:
        try:
            from_path = resolve_recipe_path(ctx.configs_dir, target_token)
            cfg = load_and_override(ctx.schema, from_path, overrides)
            lineage = target_token
            name_label = from_path.stem
        except FileNotFoundError:
            raise SystemExit(f"Source recipe not found: {target_token}")
    else:
        cfg = build_config(ctx.schema, overrides)
        lineage = None
        name_label = "defaults"

    # Generate scratch name
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    name = f"{timestamp}_{petname.generate(2)}"

    out_dir = ctx.configs_dir / "scratches"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}.yaml"

    # Save scratch
    save_recipe(
        path=out_path,
        cfg_data=cfg.model_dump(mode="python"),
        schema_cls=ctx.schema,
        overrides=overrides,
        kind="scratch",
        source=lineage,
        description=ns.message,
        force=False,
    )

    print(f"Running scratch: scratches/{name}.yaml")

    return _execute(
        ctx,
        cfg,
        source={"kind": "scratch", "path": str(out_path), "name": name_label},
        overrides=overrides,
        run_id_override=getattr(ns, "run_id", None),
    )


def _execute(
    ctx: Ctx,
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
        record_run_manifest(ctx, cfg, run_id, source, overrides)

    manifest_path = ctx.runs_dir / run_id / "manifest.yaml"

    # 3. Execute User Function
    with RunContext(manifest_path, rank) as run_ctx:
        result = ctx.func(cfg)
        run_ctx.record_result(result)
        return result


def cmd_list_configs(ctx: Ctx, ns: argparse.Namespace) -> None:
    """Handle: list configs - show all recipes in a smart table."""
    if not ctx.configs_dir.exists():
        print(f"No recipes found in {ctx.configs_dir}")
        return

    # Collect all recipes
    canonicals = sorted(ctx.configs_dir.glob("*.yaml")) + sorted(ctx.configs_dir.glob("*.yml"))

    scratches_dir = ctx.configs_dir / "scratches"
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
                created = created[:16].replace("T", " ") # Simplified ISO format

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


def cmd_list_runs(ctx: Ctx, ns: argparse.Namespace) -> None:
    """Handle: list runs - show execution history."""
    if not ctx.runs_dir.exists():
        print(f"No runs found in {ctx.runs_dir}")
        return

    rows = []
    all_keys = set()

    for p in ctx.runs_dir.iterdir():
        if not p.is_dir(): continue
        manifest_path = p / "manifest.yaml"
        if not manifest_path.exists(): continue
        
        try:
            data = read_yaml(manifest_path)
            
            # Extract key info
            run_id = data.get("run_id", p.name)
            status = data.get("status", "UNKNOWN")
            created = data.get("created_at", "")[:16].replace("T", " ")
            
            # Config subset?
            config = data.get("config", {})
            flat_cfg = flatten_config(config)
            
            row = {"Run ID": run_id, "Status": status, "Created": created, **flat_cfg}
            rows.append(row)
            all_keys.update(flat_cfg.keys())
        except Exception:
            continue

    # Sort by created desc
    rows.sort(key=lambda x: x.get("Created", ""), reverse=True)
    
    _print_smart_table(rows, ["Run ID", "Status", "Created"], all_keys)


def _print_smart_table(rows: list[dict], fixed_cols: list[str], potential_cols: set[str]) -> None:
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


def cmd_edit(ctx: Ctx, ns: argparse.Namespace) -> None:
    """Handle: edit <recipe> - launch TUI editor."""
    from stryx.tui import PydanticConfigTUI

    # Use ns.recipe if present, or ns.target if repurposed (but edit has recipe arg)
    name = ns.recipe
    
    try:
        recipe_path = resolve_recipe_path(ctx.configs_dir, name)
    except FileNotFoundError:
        # Offer to create it? For now just exit
        raise SystemExit(f"Recipe not found: {name}\nCreate it first with: new {name}")

    tui = PydanticConfigTUI(ctx.schema, recipe_path)
    tui.run()


def cmd_show(ctx: Ctx, ns: argparse.Namespace) -> None:
    """Handle: show [recipe] [--config path] [overrides...]"""
    # Get schema defaults
    try:
        defaults_instance = ctx.schema()
        schema_defaults = defaults_instance.model_dump(mode="python")
    except Exception as e:
        raise SystemExit(f"Schema has required fields without defaults:\n{e}")

    # Determine source file
    source_name = "defaults"
    recipe_data: dict[str, Any] | None = None
    
    # Check if a config path was provided explicitly
    path_arg = getattr(ns, "config_path", None)
    if path_arg:
        if not path_arg.exists():
            raise SystemExit(f"Config not found: {path_arg}")
        path = path_arg
    else:
        # Fallback to recipe/target name
        target = getattr(ns, "recipe", None) or getattr(ns, "target", None)
        path = None
        if target:
            try:
                path = resolve_recipe_path(ctx.configs_dir, target)
            except FileNotFoundError:
                raise SystemExit(f"Config not found: {target}")

    if path:
        recipe_data = read_config_file(path)
        # Strip metadata
        if isinstance(recipe_data, dict):
            recipe_data = {k: v for k, v in recipe_data.items() if not k.startswith("__")}
        source_name = path.stem

    # Build the config data (before validation, to track sources)
    if recipe_data is not None:
        data = dict(recipe_data)
    else:
        data = dict(schema_defaults)

    # Track override paths and their previous values
    override_info: dict[str, Any] = {}  # path → previous value
    for tok in ns.overrides:
        key, _ = tok.split("=", 1)
        key = key.strip()
        # Get previous value before override
        prev = _get_nested(data, key.split("."))
        override_info[key] = prev
        apply_override(data, tok)

    # Validate
    cfg = validate_or_die(ctx.schema, data, "show")
    final_data = cfg.model_dump(mode="python")

    # Print header
    header_parts = ["Config"]
    if source_name != "defaults":
        header_parts.append(f"recipe: {source_name}")
    if ns.overrides:
        header_parts.append(f"{len(ns.overrides)} override{'s' if len(ns.overrides) > 1 else ''}")
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


def cmd_diff(ctx: Ctx, ns: argparse.Namespace) -> None:
    """Handle: diff <recipe_a> <recipe_b>"""
    # Load both configs
    try:
        path_a = resolve_recipe_path(ctx.configs_dir, ns.recipe_a)
        cfg_a = read_config_file(path_a)
    except FileNotFoundError:
        raise SystemExit(f"Recipe not found: {ns.recipe_a}")

    if ns.recipe_b:
        try:
            path_b = resolve_recipe_path(ctx.configs_dir, ns.recipe_b)
            cfg_b = read_config_file(path_b)
            name_b = ns.recipe_b
        except FileNotFoundError:
            raise SystemExit(f"Recipe not found: {ns.recipe_b}")
    else:
        # Diff against defaults
        base = ctx.schema()
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

    print(f"Diff: {ns.recipe_a} vs {name_b}")
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


def cmd_schema(ctx: Ctx, ns: argparse.Namespace) -> None:
    """Handle: schema - print the configuration schema."""
    print(f"Schema: {ctx.schema.__module__}:{ctx.schema.__name__}")

    fields = extract_fields(ctx.schema)
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
            _print_with_sources(value, defaults, recipe, override_info, path, indent + 1)
        else:
            source = _get_source(path, value, defaults, recipe, override_info)
            val_str = _format_value(value)
            left_part = f"{pad}{key}: {val_str}"
            padding = max(1, 45 - len(left_part))
            print(f"{left_part}{' ' * padding}({source})")
