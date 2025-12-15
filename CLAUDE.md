# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Stryx is a typed configuration compiler for ML experiments** — an alternative to Hydra that doesn't suck.

### The Problem with Hydra
- Runtime config resolution with magic interpolation
- YAML authoring with obscure syntax
- Difficult to debug, hard to version, non-reproducible

### The Stryx Approach
1. **Define once** — Write your experiment config as a Pydantic schema in Python
2. **Compile** — Use Hydra-style CLI overrides to materialize fully-resolved, validated YAML files ("recipes")
3. **Consume** — Training code loads exactly ONE frozen config file — no runtime resolution, no magic

This makes runs **reproducible**, **debuggable**, and **easy to inspect or version**.

## Core API: The `@stryx.cli` Decorator

```python
import stryx
from pydantic import BaseModel

class Config(BaseModel):
    lr: float = 1e-4
    batch_size: int = 32

@stryx.cli(schema=Config)
def main(cfg: Config):
    train(cfg)

if __name__ == "__main__":
    main()
```

That's it. The decorator provides the entire CLI.

## CLI Commands

```bash
# Run with defaults
python train.py

# Run with overrides (ephemeral, not saved)
python train.py lr=1e-3 batch_size=64

# Save a recipe
python train.py new my_exp lr=1e-3

# Copy and modify a recipe
python train.py new my_exp_v2 --from my_exp batch_size=128

# Run from a saved recipe
python train.py run my_exp

# Run from recipe with additional overrides
python train.py run my_exp lr=1e-4

# Edit recipe interactively (TUI)
python train.py edit my_exp

# Load from explicit path
python train.py --config path/to/config.yaml
```

## Config Hierarchy

Precedence (lowest → highest):
1. **Schema defaults** — `lr: float = 1e-4` in Pydantic model
2. **Recipe file** — Values in the YAML recipe
3. **CLI overrides** — `lr=1e-3` on command line

## Development Commands

**IMPORTANT: Always prefix commands with `uv run`** — this ensures the correct virtual environment.

```bash
# Install
uv sync

# Run example with defaults
uv run python examples/train.py

# Run with overrides
uv run python examples/train.py lr=1e-4 train.steps=100

# Create a recipe
uv run python examples/train.py new my_exp exp_name=my_exp

# Run from recipe
uv run python examples/train.py run my_exp

# Edit interactively
uv run python examples/train.py edit my_exp
```

## Architecture

### Core Module

**src/stryx/decorator.py** — The `@stryx.cli` decorator (heart of Stryx)
- `cli()`: Decorator that transforms a function into a full CLI
- `_dispatch()`: Routes CLI commands to handlers
- `_cmd_new()`: Creates and saves recipes
- `_cmd_edit()`: Launches TUI editor
- `_build_config()`: Builds config from schema defaults + overrides
- `_load_and_override()`: Loads recipe + applies overrides
- `_parse_value()`: Smart type inference for CLI values

### Supporting Modules

**src/stryx/config.py** — Config data management
- `ConfigManager`: Loads/saves YAML, validates against schema
- Used by TUI for interactive editing

**src/stryx/schema.py** — Pydantic schema introspection
- `SchemaIntrospector`: Analyzes schema structure
- `extract_fields()`: Flattens nested schemas
- Helper functions: `is_union_type()`, `unwrap_optional()`, `is_discriminated_union()`

**src/stryx/tui.py** — Interactive terminal UI
- `PydanticConfigTUI`: prompt_toolkit TUI with fuzzy search
- `launch_tui()`: Entry point for TUI
- Used by `edit` command

**src/stryx/utils.py** — Shared utilities
- `read_yaml()`, `write_yaml()`: Atomic YAML I/O
- `get_nested()`, `set_nested()`, `set_dotpath()`: Nested dict access
- `path_to_str()`: Path formatting

**src/stryx/cli.py** — Standalone `stryx` command (just shows usage info)

## Schema Patterns

```python
from typing import Literal, Union
from pydantic import BaseModel, ConfigDict, Field

# Discriminated unions for variant types
class AdamWCfg(BaseModel):
    kind: Literal["adamw"] = "adamw"
    lr: float = 3e-4

class SGDCfg(BaseModel):
    kind: Literal["sgd"] = "sgd"
    lr: float = 1e-2

OptimizerCfg = Union[AdamWCfg, SGDCfg]

# Main config
class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")  # Catch typos!

    exp_name: str = "demo"
    train: TrainCfg = Field(default_factory=TrainCfg)
    optim: OptimizerCfg = Field(default_factory=AdamWCfg, discriminator="kind")
```

Key points:
- `Field(discriminator="kind")` for union types
- `ConfigDict(extra="forbid")` catches typos in CLI overrides
- `default_factory` for mutable defaults
- Provide sensible defaults for all fields

## Override Syntax

```bash
# Simple values
lr=1e-4
batch_size=32
exp_name=my_experiment

# Nested paths
train.steps=1000
optim.lr=1e-3
optim.weight_decay=0.01

# Special values
enabled=true
disabled=false
optional_field=null

# Quoted strings
name="hello world"
path='/path/with spaces'

# JSON for complex values
tags='["tag1", "tag2"]'
```

## Recipe Format

Compiled recipes are frozen YAML with metadata:

```yaml
__stryx__:
  schema: train:Config
  created_at: '2024-01-15T10:30:00+00:00'
  from: base_exp          # if copied from another recipe
  overrides:              # CLI overrides used
    - lr=1e-4
    - batch_size=64
exp_name: my_exp
train:
  batch_size: 64
  steps: 1000
optim:
  kind: adamw
  lr: 0.0001
```

## Key Dependencies

- **pydantic>=2** — Schema definition and validation
- **pyyaml>=6.0** — YAML serialization
- **prompt-toolkit>=3.0** — Terminal UI
- **rapidfuzz>=3.0** — Fuzzy search in TUI
