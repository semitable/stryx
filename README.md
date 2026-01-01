# Stryx

**Stryx** is a typed configuration compiler and experiment manager designed for Machine Learning workflows. It provides a lightweight, type-safe approach to experiment tracking and configuration management.

## Key Features

*   **Type-First:** Configurations are defined as Pydantic models, ensuring validation and IDE support.
*   **Configs as Source:** Experiment configurations are compiled into static YAML "recipes".
*   **Minimal Surface:** A single decorator (`@stryx.cli`) adds powerful CLI capabilities to any entry point.

## Installation

Stryx is available on PyPI. Install it using `uv` (or pip):

```bash
uv add stryx
```

## Quick Start

Define your configuration schema and decorate your main function:

```python
# example.py
from pydantic import BaseModel
import stryx

class Config(BaseModel):
    lr: float = 1e-4
    epochs: int = 10

@stryx.cli(schema=Config)
def main(cfg: Config):
    print(f"Training with lr={cfg.lr}, epochs={cfg.epochs}")

if __name__ == "__main__":
    main()
```

## CLI Usage

Your script now has access to Stryx's full CLI functionality.

### Managing Recipes (`configs`)

Create, modify, and inspect your experiment "recipes" (static YAML files).

*   **Create new:**
    ```bash
    uv run example.py configs new my_exp lr=0.01
    # Creates: configs/my_exp.yaml
    ```

*   **List all:**
    ```bash
    uv run example.py configs list
    ```

*   **Show details:**
    Shows the resolved configuration and the source of each value (default, recipe, or override).
    ```bash
    uv run example.py configs show my_exp
    ```

*   **Diff:**
    Compare two configurations to see what changed.
    ```bash
    uv run example.py configs diff my_exp other_exp
    ```

*   **Fork:**
    Create a new recipe based on an existing one.
    ```bash
    uv run example.py configs fork my_exp new_exp epochs=20
    ```

*   **Edit (Interactive):**
    Open a terminal UI to edit a configuration.
    ```bash
    uv run example.py configs edit my_exp
    ```

### Running Experiments (`run` / `runs`)

Execute experiments and track their history.

*   **Run a recipe:**
    ```bash
    uv run example.py run my_exp
    ```

*   **Run with overrides:**
    This creates a temporary "scratch" config to ensure reproducibility without cluttering your main recipes.
    ```bash
    uv run example.py run my_exp lr=0.02
    ```

*   **List history:**
    See all past execution runs.
    ```bash
    uv run example.py runs list
    ```

*   **Inspect a run:**
    View the exact configuration used for a past run.
    ```bash
    uv run example.py runs show <run_id>
    ```

*   **Compare runs:**
    See how configuration differed between two runs.
    ```bash
    uv run example.py runs diff <run_id_1> <run_id_2>
    ```

### Validation

Type checking is enforced automatically. Invalid inputs are caught early:

```bash
uv run example.py configs new lr=astring
# Output:
# Config validation failed (building config):
# optim.adamw.lr: Input should be a valid number, unable to parse string as a number
```
