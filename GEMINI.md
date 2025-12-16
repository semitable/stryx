# Stryx Project Context

## Project Overview
**Stryx** is a typed configuration compiler for Machine Learning experiments. It sits between tools like Sacred and Hydra, offering a lightweight, composable approach without the complexity of Hydra's indirection.

**Key Features:**
*   **Type-First:** Configurations are defined as Pydantic models (or dataclasses), providing IDE support and validation.
*   **Recipes as Source:** Experiment configurations are compiled into static YAML "recipes" that include metadata and lineage.
*   **Minimal Surface:** A single decorator (`@stryx.cli`) adds powerful CLI capabilities (run, new, show, list, edit, schema) to any script.
*   **Reproducibility:** Defaults are code-driven, overrides are explicit, and recipes track their origin.

## Building and Running

This project uses `uv` for dependency management and task execution.

### Setup
*   **Install dependencies:**
    ```bash
    uv sync
    ```
    *   To include example dependencies: `uv sync --extra examples`
    *   To include dev dependencies: `uv sync --extra dev`

### Running Examples
*   **Basic usage (defaults):**
    ```bash
    uv run examples/train.py
    ```
*   **With overrides:**
    ```bash
    uv run examples/train.py train.steps=500 optim.lr=1e-4
    ```
*   **Create a new recipe:**
    ```bash
    uv run examples/train.py new my_exp optim.lr=1e-4
    ```
*   **Run a specific recipe:**
    ```bash
    uv run examples/train.py run my_exp
    ```
*   **Interactive Edit (TUI):**
    ```bash
    uv run examples/train.py edit my_exp
    ```
*   **Inspect Schema:**
    ```bash
    uv run examples/train.py schema
    ```

### Testing
*   **Run all tests:**
    ```bash
    uv run pytest
    ```
*   **Run specific test:**
    ```bash
    uv run pytest tests/test_cli.py
    ```

### Building
*   **Build package:**
    ```bash
    uv build
    ```

## Development Conventions

*   **Language:** Python 3.11+
*   **Typing:** Fully typed codebase. Return concrete types instead of `Any`.
*   **Style:**
    *   4-space indentation.
    *   Snake_case for modules and files.
    *   Imperative mood for commit messages and docstrings.
*   **Error Handling:** Use `SystemExit` with clear messages for user-facing CLI errors; avoid bare exceptions.
*   **File I/O:** Use `stryx.utils.read_yaml` and `stryx.utils.write_yaml` for atomic operations. Prefer `pathlib.Path` over strings.
*   **Testing:**
    *   Unit tests in `tests/`.
    *   Integration tests in `tests/integration/`.
    *   Use `tmp_path` fixture for filesystem operations to ensure hermetic runs.

## Key Files & Directories

*   **`src/stryx/decorator.py`**: The core implementation of the `@stryx.cli` decorator, handling command parsing and dispatch.
*   **`src/stryx/config.py`**: Manages configuration loading, saving, and validation against the schema.
*   **`src/stryx/schema.py`**: Utilities for introspecting Pydantic models (extracting fields, handling unions/optionals).
*   **`src/stryx/tui.py`**: Implementation of the interactive Terminal UI using `prompt_toolkit`.
*   **`src/stryx/utils.py`**: Shared utilities for YAML I/O and nested dictionary manipulation.
*   **`examples/`**: meaningful examples like `train.py` (standard usage) and `sft.py` (HuggingFace integration).
*   **`pyproject.toml`**: Project configuration and dependencies.
