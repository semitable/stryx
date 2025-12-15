# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Stryx is an interactive config builder for Pydantic schemas. It provides two primary modes:

1. **Recipe-based CLI** - A Hydra-style CLI for creating, managing, and running configurations with override support
2. **Interactive TUI** - A terminal UI for schema-aware config editing with fuzzy search, live validation, and type information

The project enables ML researchers and engineers to define configurations as Pydantic schemas and easily create, modify, and manage configuration files through both programmatic and interactive interfaces.

## Development Commands

### Installation
```bash
# Install dependencies with uv (package manager)
uv sync

# Install in development mode
uv pip install -e .
```

### Running the CLI
**IMPORTANT: Always prefix commands with `uv run`** - this ensures the correct virtual environment and dependencies are used.

```bash
# Using the installed CLI
uv run stryx new examples/train.py:Config
uv run stryx edit config/my_exp.yaml examples/train.py:Config
uv run stryx show config/my_exp.yaml

# Running examples directly
cd examples
uv run python train.py new my_exp exp_name=my_exp model_name=vit_b
uv run python train.py show my_exp
uv run python train.py run my_exp
```

### Package Management
The project uses `uv` as its build backend and package manager (specified in pyproject.toml). Always use `uv` commands for dependency management rather than pip. All Python commands must be prefixed with `uv run` to use the correct environment.

## Architecture

### Core Components

**src/stryx/config.py** - Core configuration management
- `ConfigManager`: Manages Pydantic-based config data with validation and persistence
- Handles: loading/saving YAML files, validation, data access (get_at/set_at)
- Schema introspection: `get_field_info_for_path()`, `get_union_variants()`, `is_discriminator_field()`
- Discriminated union support: detects discriminator fields dynamically (not hardcoded to "kind")
- `_build_defaults_dict()`: Instantiates discriminated unions with first variant's defaults
- No UI dependencies - pure data management

**src/stryx/schema.py** - Pydantic schema introspection utilities
- `extract_fields()`: Recursively walks Pydantic schemas to flatten nested structures
- `FieldInfo`: Data class representing flattened config field for UI display
- Handles nested BaseModels, Optional types, and discriminated unions
- Discriminated unions: extracts fields from default variant (first member or from default_value)

**src/stryx/recipes.py** - Recipe-based configuration system
- `recipe_cli()`: Entry point that parses CLI arguments into a RecipeCmd object
- `RecipeCmd`: Generic command object with methods for each operation (write/text/load)
- `_apply_override_token()`: Hydra-style override parsing (key=value or a.b.c=value)
- `_parse_value()`: Smart type inference for overrides (null/bool/numbers/JSON/strings)
- Supports YAML and JSON config files
- Adds metadata header `__stryx__` with schema info, build timestamp, and overrides

**src/stryx/cli.py** - CLI entry point
- `load_schema_from_path()`: Dynamic module loading from 'path/file.py:ClassName' format
- Commands: new (create config), edit (modify existing), show (display config)
- Imports and uses `launch_tui()` from tui.py for interactive editing

**src/stryx/tui.py** - Interactive terminal UI (pure UI code)
- `PydanticConfigTUI`: prompt_toolkit-based TUI with fuzzy search and live validation
- `launch_tui()`: Entry point for interactive config editing
- Uses `ConfigManager` for all data operations
- Features: fuzzy search (rapidfuzz), live preview, boolean toggling, variant selection
- Discriminator field detection: editing discriminator fields triggers variant selector
- Validates on save and displays validation errors in status bar

### Schema Structure Requirements

Pydantic schemas used with Stryx should follow these patterns:

1. **Nested configurations** - Use nested BaseModel classes for grouping related fields
2. **Discriminated unions** - For variant types (e.g., optimizer choices), use Field(discriminator="kind")
3. **Defaults** - Provide sensible defaults where possible; fields without defaults are marked required
4. **Type hints** - Use proper Python type hints; supports Optional, Union, Literal, nested models

Example pattern (see examples/train.py):
```python
class AdamWCfg(BaseModel):
    kind: Literal["adamw"] = "adamw"
    lr: float = 3e-4

class SGDCfg(BaseModel):
    kind: Literal["sgd"] = "sgd"
    lr: float = 1e-2

class Config(BaseModel):
    optim: Union[AdamWCfg, SGDCfg] = Field(discriminator="kind")
```

### Override System

The recipe CLI supports Hydra-style overrides:
- Simple: `key=value`
- Nested: `a.b.c=value`
- Smart parsing: `null`, `true`/`false`, numbers, JSON literals, quoted strings
- Overrides are stored in the `__stryx__.overrides` field of compiled recipes

### TUI Integration

The TUI is fully integrated with the CLI and works as follows:
1. User runs `stryx new` or `stryx edit` command
2. CLI loads the Pydantic schema using `load_schema_from_path()`
3. CLI calls `launch_tui()` with the schema and config path
4. TUI initializes config with schema defaults (for new) or loads existing config (for edit)
5. User edits fields interactively with live validation feedback
6. On Ctrl+S, TUI validates against schema and saves if valid
7. Validation errors are displayed in the status bar

## Key Dependencies

- **pydantic>=2** - Schema definition and validation
- **prompt-toolkit>=3.0** - Terminal UI framework
- **pyyaml>=6.0** - YAML parsing and serialization
- **rapidfuzz>=3.0** - Fuzzy search for TUI field filtering
