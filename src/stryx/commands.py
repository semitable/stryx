import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, List, Optional

import typer
from filelock import FileLock
import petname

from stryx.config import (
    apply_override,
    build_config,
    load_and_override,
    read_config_file,
    validate_or_die,
    save_recipe,
)
from stryx.lifecycle import RunContext, get_rank, record_run_manifest
from stryx.run_id import derive_run_id
from stryx.schema import extract_fields
from stryx.utils import (
    Ctx,
    flatten_config,
    get_next_sequential_name,
    read_yaml,
    resolve_recipe_path,
)

# ============================================================================
# Recipe Commands
# ============================================================================

def recipe_init(
    ctx: typer.Context,
    name: Annotated[
        Optional[str],
        typer.Argument(
            metavar="name",
            help="Name for the recipe. If omitted, uses 'exp_XXX'.",
        ),
    ] = None,
    overrides: Annotated[
        Optional[List[str]],
        typer.Argument(
            metavar="overrides",
            help="Configuration overrides (key=value).",
        ),
    ] = None,
    message: Annotated[
        Optional[str],
        typer.Option("--message", "-m", help="Description for metadata."),
    ] = None,
    force: Annotated[
        bool,
        typer.Option("--force", help="Overwrite existing recipe."),
    ] = False,
) -> Path:
    """Create a new recipe from defaults."""
    c: Ctx = ctx.obj
    
    # Smart parse: if name looks like override, shift it
    name, overrides = _shift_arg_if_override(name, overrides)

    cfg = build_config(c.schema, overrides)
    cfg_data = cfg.model_dump(mode="python")

    c.configs_dir.mkdir(parents=True, exist_ok=True)

    try:
        if name:
            _validate_name_not_override(name, "name")
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

    print(f"Initialized recipe: {out_path}")
    return out_path


def recipe_fork(
    ctx: typer.Context,
    source: Annotated[str, typer.Argument(help="Source recipe name/path.")],
    name: Annotated[str, typer.Argument(help="New recipe name.")],
    overrides: Annotated[
        Optional[List[str]],
        typer.Argument(help="Overrides to apply."),
    ] = None,
    message: Annotated[
        Optional[str],
        typer.Option("--message", "-m", help="Description."),
    ] = None,
    force: Annotated[
        bool,
        typer.Option("--force", help="Overwrite destination."),
    ] = False,
) -> Path:
    """Fork an existing recipe with modifications."""
    c: Ctx = ctx.obj
    overrides = overrides or []

    # Validate names to prevent "stryx recipes fork source optim.lr=1" error
    # source might be a path, but usually won't contain '=' unless it's weird.
    # But 'name' definitely shouldn't.
    _validate_name_not_override(source, "source")
    _validate_name_not_override(name, "name")

    try:
        from_path = resolve_recipe_path(c.configs_dir, source)
    except FileNotFoundError:
        print(f"Source recipe not found: {source}")
        raise typer.Exit(code=1)

    cfg = load_and_override(c.schema, from_path, overrides)
    cfg_data = cfg.model_dump(mode="python")

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


def recipe_edit(
    ctx: typer.Context,
    name: Annotated[str, typer.Argument(help="Recipe to edit.")],
) -> None:
    """Edit a recipe interactively (TUI)."""
    from stryx.tui import PydanticConfigTUI
    c: Ctx = ctx.obj
    _validate_name_not_override(name, "name")

    try:
        recipe_path = resolve_recipe_path(c.configs_dir, name)
    except FileNotFoundError:
        print(f"Recipe not found: {name}")
        raise typer.Exit(code=1)

    tui = PydanticConfigTUI(c.schema, recipe_path)
    tui.run()


def recipe_show(
    ctx: typer.Context,
    name: Annotated[
        Optional[str],
        typer.Argument(help="Recipe to show (default: defaults)."),
    ] = None,
    overrides: Annotated[
        Optional[List[str]],
        typer.Argument(help="Overrides to apply temporarily."),
    ] = None,
) -> None:
    """Show recipe configuration and provenance."""
    c: Ctx = ctx.obj
    
    name, overrides = _shift_arg_if_override(name, overrides)

    _show_config(c, name, overrides, title="Recipe")


def recipe_diff(
    ctx: typer.Context,
    name_a: Annotated[str, typer.Argument(help="First recipe.")],
    name_b: Annotated[
        Optional[str],
        typer.Argument(help="Second recipe (default: defaults)."),
    ] = None,
) -> None:
    """Compare two recipes."""
    c: Ctx = ctx.obj
    _validate_name_not_override(name_a, "name_a")
    if name_b:
        _validate_name_not_override(name_b, "name_b")

    try:
        path_a = resolve_recipe_path(c.configs_dir, name_a)
        cfg_a = read_config_file(path_a)
    except FileNotFoundError:
        print(f"Recipe not found: {name_a}")
        raise typer.Exit(code=1)

    if name_b:
        try:
            path_b = resolve_recipe_path(c.configs_dir, name_b)
            cfg_b = read_config_file(path_b)
            label_b = name_b
        except FileNotFoundError:
            print(f"Recipe not found: {name_b}")
            raise typer.Exit(code=1)
    else:
        base = c.schema()
        cfg_b = base.model_dump(mode="python")
        label_b = "(defaults)"

    _diff_dicts(cfg_a, cfg_b, name_a, label_b)


def recipe_list(ctx: typer.Context) -> None:
    """List all recipes."""
    c: Ctx = ctx.obj
    if not c.configs_dir.exists():
        print(f"No recipes found in {c.configs_dir}")
        return

    canonicals = sorted(c.configs_dir.glob("*.yaml")) + sorted(
        c.configs_dir.glob("*.yml")
    )
    scratches_dir = c.configs_dir / "scratches"
    scratches = sorted(scratches_dir.glob("*.yaml"), reverse=True) if scratches_dir.exists() else []

    all_recipes = canonicals + scratches
    rows = []
    all_keys = set()

    for p in all_recipes:
        try:
            data = read_yaml(p)
            meta = data.get("__stryx__", {}) if isinstance(data, dict) else {}
            created = meta.get("created_at", "")[:16].replace("T", " ")
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


def recipe_schema(
    ctx: typer.Context,
    json_out: Annotated[bool, typer.Option("--json", help="JSON output")] = False,
) -> None:
    """Print configuration schema."""
    c: Ctx = ctx.obj
    if json_out:
        schema = c.schema.model_json_schema()
        # Inject __stryx__ metadata to allow validation of recipe files
        if "properties" not in schema:
            schema["properties"] = {}
            
        schema["properties"]["__stryx__"] = {
            "title": "Stryx Metadata",
            "description": "Metadata managed by Stryx",
            "type": "object",
            "properties": {
                "type": {"type": "string", "title": "Type"},
                "source": {"type": "string", "title": "Source"},
                "overrides": {
                    "type": "array", 
                    "items": {"type": "string"},
                    "title": "Overrides"
                },
                "description": {"type": "string", "title": "Description"},
                "created_at": {"type": "string", "title": "Created At"}
            }
        }
        print(json.dumps(schema, indent=2))
        return

    print(f"Schema: {c.schema.__module__}:{c.schema.__name__}")
    fields = extract_fields(c.schema)
    if fields:
        print("Fields:")
        # Grouping logic... same as before
        groups = {}
        order = []
        for field in fields:
            g = field.path.split(".", 1)[0]
            if g not in groups:
                groups[g] = []
                order.append(g)
            groups[g].append(field)
        
        for g in order:
            entries = groups[g]
            parent = next((e for e in entries if e.path == g), None)
            children = [e for e in entries if e.path != g]
            
            if parent:
                for line in _format_field_lines("  ", g, parent.type_str, parent.default_str, parent.description):
                    print(line)
            else:
                print(f"  {g}:")
            
            for child in children:
                label = child.path[len(g)+1:] if child.path.startswith(f"{g}.") else child.path
                for line in _format_field_lines("    ", label, child.type_str, child.default_str, child.description):
                    print(line)
            print()


# ============================================================================
# Run Commands
# ============================================================================

def run_exec(
    ctx: typer.Context,
    target: Annotated[
        Optional[str],
        typer.Argument(metavar="recipe", help="Recipe to execute (or overrides)."),
    ] = None,
    overrides: Annotated[
        Optional[List[str]],
        typer.Argument(metavar="overrides", help="Configuration overrides."),
    ] = None,
    dry: Annotated[bool, typer.Option("--dry", help="Dry run.")] = False,
    run_id: Annotated[Optional[str], typer.Option("--run-id", help="Explicit Run ID.")] = None,
) -> Any:
    """Start an experiment run."""
    c: Ctx = ctx.obj
    
    # Handle implicit overrides
    target, overrides = _shift_arg_if_override(target, overrides)
            
    # Resolve Config
    lineage = None
    name_label = "defaults"
    
    if target:
        try:
            path = resolve_recipe_path(c.configs_dir, target)
            cfg = load_and_override(c.schema, path, overrides)
            lineage = target
            name_label = path.stem
        except FileNotFoundError:
            print(f"Recipe not found: {target}")
            raise typer.Exit(code=1)
    else:
        cfg = build_config(c.schema, overrides)

    if dry:
        print("Dry run: Config resolved successfully.")
        return

    # Scratch logic
    is_variant = bool(overrides) or (target is None)
    source_info = {}
    
    if is_variant:
        timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
        name = f"{timestamp}_{petname.generate(2)}"
        out_dir = c.configs_dir / "scratches"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{name}.yaml"
        
        save_recipe(
            path=out_path,
            cfg_data=cfg.model_dump(mode="python"),
            schema_cls=c.schema,
            overrides=overrides,
            kind="scratch",
            source=lineage,
            force=False
        )
        print(f"Running scratch: scratches/{name}.yaml")
        source_info = {"kind": "scratch", "path": str(out_path), "name": name_label}
    else:
        path = resolve_recipe_path(c.configs_dir, target)
        source_info = {"kind": "file", "path": str(path), "name": path.stem}

    # Execution
    run_id = derive_run_id(label=source_info.get("name") or "run", run_id_override=run_id)
    rank = get_rank()
    
    if rank == 0:
        record_run_manifest(c, cfg, run_id, source_info, overrides)
        
    manifest_path = c.runs_dir / run_id / "manifest.yaml"
    
    with RunContext(manifest_path, rank) as run_ctx:
        result = c.func(cfg)
        run_ctx.record_result(result)
        return result


def run_list(ctx: typer.Context) -> None:
    """List execution history."""
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
            run_id = data.get("run_id", p.name)
            status = data.get("status", "UNKNOWN")
            created = data.get("created_at", "")[:16].replace("T", " ")
            
            config_path = p / "config.yaml"
            config = read_yaml(config_path) if config_path.exists() else {}
            
            flat_cfg = flatten_config(config)
            row = {"Run ID": run_id, "Status": status, "Created": created, **flat_cfg}
            rows.append(row)
            all_keys.update(flat_cfg.keys())
        except Exception:
            continue

    rows.sort(key=lambda x: x.get("Created", ""), reverse=True)
    _print_smart_table(rows, ["Run ID", "Status", "Created"], all_keys)


def run_show(ctx: typer.Context, run_id: Annotated[str, typer.Argument(help="Run ID to show")]) -> None:
    """Show configuration for a run."""
    c: Ctx = ctx.obj
    path = c.runs_dir / run_id / "manifest.yaml"
    if not path.exists():
        print(f"Run not found: {run_id}")
        raise typer.Exit(code=1)
    
    data = read_yaml(path)
    config_path = c.runs_dir / run_id / "config.yaml"
    config = read_yaml(config_path) if config_path.exists() else {}
    
    defaults_instance = c.schema()
    schema_defaults = defaults_instance.model_dump(mode="python")
    
    print(f"Run: {run_id}")
    print(f"Status: {data.get('status')}")
    print("=" * 60)
    
    _print_with_sources(config, schema_defaults, None, {}, "", 0)


def run_diff(ctx: typer.Context, id_a: str, id_b: str) -> None:
    """Diff two runs."""
    c: Ctx = ctx.obj
    
    def load_run(rid):
        p = c.runs_dir / rid / "manifest.yaml"
        if not p.exists():
            print(f"Run not found: {rid}")
            raise typer.Exit(code=1)
        config_path = c.runs_dir / rid / "config.yaml"
        return read_yaml(config_path) if config_path.exists() else {}

    cfg_a = load_run(id_a)
    cfg_b = load_run(id_b)
    
    _diff_dicts(cfg_a, cfg_b, id_a, id_b)


# ============================================================================
# Shared Helpers
# ============================================================================

_NOT_FOUND = object()

def _validate_name_not_override(value: str, arg_name: str) -> None:
    """Raise error if a name argument looks like an override (contains '=')."""
    if "=" in value:
        print(f"Error: Argument '{arg_name}' ('{value}') looks like an override (contains '=').")
        print(f"Did you forget to provide the '{arg_name}'?")
        raise typer.Exit(code=1)

def _shift_arg_if_override(
    arg: str | None,
    overrides: list[str] | None
) -> tuple[str | None, list[str]]:
    """Shift argument to overrides if it looks like an override."""
    overrides = overrides or []
    if arg and "=" in arg:
        return None, [arg] + overrides
    return arg, overrides

def _get_nested(data: dict[str, Any], path: list[str]) -> Any:
    current = data
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return _NOT_FOUND
        current = current[key]
    return current

def _show_config(c: Ctx, target: str | None, overrides: list[str], title: str) -> None:
    # Schema defaults
    try:
        defaults = c.schema().model_dump(mode="python")
    except Exception as e:
        print(f"Schema error: {e}")
        raise typer.Exit(code=1)

    recipe_data = None
    source_name = "defaults"

    if target:
        try:
            path = resolve_recipe_path(c.configs_dir, target)
            recipe_data = read_config_file(path)
            if isinstance(recipe_data, dict):
                recipe_data = {k: v for k, v in recipe_data.items() if not k.startswith("__")}
            source_name = path.stem
        except FileNotFoundError:
            print(f"Config not found: {target}")
            raise typer.Exit(code=1)

    data = dict(recipe_data) if recipe_data else dict(defaults)
    override_info = {}
    
    for tok in overrides:
        try:
            key, _ = tok.split("=", 1)
            key = key.strip()
            override_info[key] = _get_nested(data, key.split("."))
            apply_override(data, tok)
        except ValueError:
            print(f"Invalid override: {tok}")
            raise typer.Exit(code=1)

    cfg = validate_or_die(c.schema, data, "show")
    final = cfg.model_dump(mode="python")

    print(f"{title}: {source_name}")
    if overrides:
        print(f"Overrides: {len(overrides)}")
    print("=" * 60)
    _print_with_sources(final, defaults, recipe_data, override_info, "", 0)

def _diff_dicts(dict_a: dict, dict_b: dict, label_a: str, label_b: str) -> None:
    # Strip metadata if present
    def clean(d):
        return {k: v for k, v in d.items() if not k.startswith("__")} if isinstance(d, dict) else d
        
    flat_a = flatten_config(clean(dict_a))
    flat_b = flatten_config(clean(dict_b))
    
    all_keys = sorted(set(flat_a.keys()) | set(flat_b.keys()))
    print(f"Diff: {label_a} vs {label_b}")
    print("-" * 60)
    
    has_diff = False
    for key in all_keys:
        val_a = flat_a.get(key, _NOT_FOUND)
        val_b = flat_b.get(key, _NOT_FOUND)
        if val_a != val_b:
            has_diff = True
            if val_a is _NOT_FOUND:
                print(f"+ {key}: {val_b}")
            elif val_b is _NOT_FOUND:
                print(f"- {key}: {val_a}")
            else:
                print(f"~ {key}: {val_a} -> {val_b}")
    
    if not has_diff:
        print("No differences found.")

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

def _print_with_sources(final, defaults, recipe, override_info, prefix, indent):
    pad = "  " * indent
    for key, value in final.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            print(f"{pad}{key}:")
            _print_with_sources(value, defaults, recipe, override_info, path, indent + 1)
        else:
            source = _get_source(path, value, defaults, recipe, override_info)
            val_str = _format_value(value)
            left = f"{pad}{key}: {val_str}"
            padding = max(1, 45 - len(left))
            print(f"{left}{' ' * padding}({source})")

def _get_source(path, value, defaults, recipe, override_info):
    if path in override_info:
        prev = override_info[path]
        return "override (new)" if prev is _NOT_FOUND else f"override ← {_format_value(prev)}"
    
    default_val = _get_nested(defaults, path.split("."))
    if recipe:
        recipe_val = _get_nested(recipe, path.split("."))
        if recipe_val is not _NOT_FOUND and recipe_val != default_val:
            return "recipe"
    return "default"

def _format_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return f'"{value}"'
    return str(value)

def _format_field_lines(indent, label, type_name, default_str, description):
    line = f"{indent}{label}: {type_name}"
    if default_str:
        line += f" = {default_str}"
    if description:
        padding = max(1, 42 - len(line)) if len(line) < 40 else 1
        return [f"{line}{' ' * padding}# {description}"]
    return [line]
