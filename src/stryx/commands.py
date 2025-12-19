import argparse
from datetime import datetime, timezone
from pathlib import Path

from stryx.config_builder import build_config, load_and_override
from stryx.context import Ctx
from stryx.utils import get_next_sequential_name, write_yaml, resolve_recipe_path


def cmd_new(ctx: Ctx, ns: argparse.Namespace) -> Path:
    """Handle: new [name] [overrides...] - create from defaults."""
    from filelock import FileLock

    cfg = build_config(ctx.schema, ns.overrides)

    # Prepare payload
    payload = {
        "__stryx__": {
            "schema": f"{ctx.schema.__module__}:{ctx.schema.__name__}",
            "created_at": datetime.now(tz=timezone.utc).isoformat(),
            "overrides": ns.overrides,
            "type": "canonical",
        }
    }
    
    if ns.message:
        payload["__stryx__"]["description"] = ns.message

    payload.update(cfg.model_dump(mode="python"))

    # Create directory
    ctx.configs_dir.mkdir(parents=True, exist_ok=True)

    if ns.recipe:
        name = ns.recipe
        if "." not in name:
             name = f"{name}.yaml"
        
        out_path = ctx.configs_dir / name
        
        if out_path.exists() and not ns.force:
             raise SystemExit(f"Error: Recipe '{out_path}' already exists. Use --force to overwrite.")

        write_yaml(out_path, payload)
        print(f"Created recipe: {out_path}")
        return out_path

    # Auto-generate name
    lock_path = ctx.configs_dir / ".stryx.lock"
    with FileLock(lock_path):
        name = get_next_sequential_name(ctx.configs_dir)
        out_path = ctx.configs_dir / f"{name}.yaml"
        write_yaml(out_path, payload)

    print(f"Created recipe: {out_path}")
    return out_path


def cmd_fork(ctx: Ctx, ns: argparse.Namespace) -> Path:
    """Handle: fork <source> <name> [overrides...]"""
    
    # 1. Resolve and Load Source
    try:
        from_path = resolve_recipe_path(ctx.configs_dir, ns.source)
    except FileNotFoundError:
        raise SystemExit(f"Source recipe not found: {ns.source}")

    cfg = load_and_override(ctx.schema, from_path, ns.overrides)

    # 2. Prepare payload with lineage
    payload = {
        "__stryx__": {
            "schema": f"{ctx.schema.__module__}:{ctx.schema.__name__}",
            "created_at": datetime.now(tz=timezone.utc).isoformat(),
            "overrides": ns.overrides,
            "type": "canonical",
            "from": str(ns.source),
        }
    }
    
    if ns.message:
        payload["__stryx__"]["description"] = ns.message

    payload.update(cfg.model_dump(mode="python"))

    # 3. Determine output path
    name = ns.name
    if "." not in name:
        name = f"{name}.yaml"
    out_path = ctx.configs_dir / name
    
    if out_path.exists() and not ns.force:
        raise SystemExit(f"Error: Recipe '{out_path}' already exists. Use --force to overwrite.")

    # 4. Write
    ctx.configs_dir.mkdir(parents=True, exist_ok=True)
    write_yaml(out_path, payload)
    
    print(f"Forked recipe: {out_path}")
    return out_path