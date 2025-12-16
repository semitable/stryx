Stryx — Typed configuration compiler for ML experiments
=======================================================

Stryx sits between Sacred and Hydra: lightweight and composable like Sacred, without Hydra’s indirection. You define configs as Pydantic models, get a CLI for free, and manage reusable recipes in plain YAML. The goal is clean, type-safe experiment configs that are easy to share, inspect, and reproduce—no hidden magic.

Guiding principles
------------------
- Type-first: configs are Pydantic models with IDE/type checker support.
- Recipes as source: YAML recipes live in `configs/`, carry schema + provenance in `__stryx__`, and stay human-readable.
- Minimal surface: a single decorator powers run/new/show/list/edit; no global state or daemon.
- Reproducibility: defaults come from code, overrides are explicit, and recipes record lineage (`--from`, overrides).
- Non-goals: logging/metrics/backends (use W&B, MLflow, etc. alongside Stryx).

Quick start
-----------
1) Install deps from the lockfile: `uv sync` (add extras with `uv add <pkg>` → `uv lock` → `uv sync`).
2) Run the example: `uv run examples/train.py` (defaults) or override values `uv run examples/train.py train.steps=500`.
3) Create and reuse a recipe:
   - `uv run examples/train.py new my_exp optim.lr=1e-4`
   - `uv run examples/train.py list`
   - `uv run examples/train.py run my_exp`
   - `uv run examples/train.py show my_exp` (annotates values with their source)
4) Edit interactively: `uv run examples/train.py edit my_exp` opens the TUI editor.

CLI at a glance
---------------
- `@stryx.cli(schema=Config, recipes_dir="configs")` wraps your entrypoint and adds commands: run (default), new, run <recipe|path>, show, list, edit.
- Overrides use dot paths: `train.steps=1000` or `optim.lr=3e-4`.
- Recipes include a `__stryx__` block with schema, timestamp, lineage, and overrides for traceability.

Why not Hydra?
--------------
- Config sprawl: hierarchical `conf/` trees and defaults lists bloat quickly and are hard to audit or refactor. Steep learning curve causes repeated values throughout config files.
- Opaque resolution: OmegaConf/interpolation and custom resolvers add non-obvious runtime behavior that often breaks silently.
- Weak typing: YAML is the source of truth, so you lose IDE hints and static validation; errors land at runtime.
- Fragile composition: merging configs from multiple groups can produce surprising overrides; provenance is unclear. Learning curve is unnecessarily steep.
- Heavy indirection: plugins/launchers/sweep/global state layers for simple “run with overrides” workflows add cognitive load.

Stryx advantages
----------------
- Single source of truth: configs are Pydantic models with type hints, validation, and IDE completion.
- Plain recipes: YAML recipes stay small, readable, and include `__stryx__` metadata for schema + lineage.
- Predictable overrides: dot-path CLI overrides map directly to model fields; provenance is shown by `show`.
- Minimal surface: one decorator, a handful of commands; no global state, plugins, or hidden mutation.
- Repro-friendly: defaults come from code, recipes are immutable snapshots, and sequential naming is lock-safe.
- Interop-first: pair with your logging/observability stack instead of being locked into one.
