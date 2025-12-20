# Stryx

**Stryx** is a typed configuration compiler and experiment manager for Machine Learning. It sits between tools like Sacred and Hydra, offering a lightweight, composable approach without the complexity of Hydra's indirection.

## Key Features

*   **Type-First:** Configurations are defined as Pydantic models (or dataclasses), providing IDE support and validation.
*   **Recipes as Source:** Experiment configurations are compiled into static YAML "recipes" that include metadata and lineage.
*   **Minimal Surface:** A single decorator (`@stryx.cli`) adds powerful CLI capabilities to any script.
*   **Reproducibility:** Defaults are code-driven, overrides are explicit, and recipes track their origin.
*   **Lifecycle Management:** Automatic manifest recording (git state, logs, config) for every run.

## Installation

```bash
pip install stryx
```

## Quick Start

Define your configuration schema and decorate your entry point:

```python
# train.py
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

Your script now has a powerful CLI.

### 1. New Experiments
Create a reusable recipe (configuration file).

```bash
# Create from defaults with overrides
python train.py new my_exp lr=1e-3

# Create without name (auto-generated: exp_001.yaml)
python train.py new --message "Baseline run"
```

### 2. Experimentation (`try`)
Run a quick experimental variant without creating a permanent recipe. These are saved to `scratches/`.

```bash
# Try modifying a specific recipe
python train.py try my_exp epochs=5

# Try modifying defaults
python train.py try lr=5e-4
```

### 3. Execution (`run`)
Run a recipe exactly as defined (strict mode).

```bash
python train.py run my_exp
```

### 4. Lineage (`fork`)
Create a new experiment based on an existing one.

```bash
python train.py fork my_exp better_exp lr=2e-4 --message "Lower LR"
```

### 5. Inspection & Management

*   **List:** See all recipes and runs.
    ```bash
    python train.py list configs  # List recipes
    python train.py list runs     # List execution history
    ```

*   **Show:** Inspect a configuration with annotated sources (default vs recipe vs override).
    ```bash
    python train.py show my_exp
    ```

*   **Diff:** Compare two recipes.
    ```bash
    python train.py diff my_exp better_exp
    ```

*   **Schema:** Print the configuration schema (or JSON).
    ```bash
    python train.py schema
    python train.py schema --json
    ```

*   **Edit:** Open a recipe in an interactive TUI (Terminal UI).
    ```bash
    python train.py edit my_exp
    ```

## CLI Reference

| Command | Description |
| :--- | :--- |
| `new [name] [overrides...]` | Create a fresh experiment recipe from defaults. |
| `fork <src> <name> [ov...]` | Fork an existing recipe with modifications. |
| `try [target] [ov...]` | Run an experimental variant (saved to scratches). |
| `run <target>` | Run an existing recipe exactly (strict). |
| `list {configs,runs}` | List saved recipes or execution history. |
| `show [target] [ov...]` | Display configuration with source annotations. |
| `diff <A> [B]` | Compare two recipes (or A vs defaults). |
| `schema` | Show the configuration schema. |
| `edit <recipe>` | Edit a recipe interactively (TUI). |

### Global Options
*   `--runs-dir`: Override directory for run logs and manifests.
*   `--configs-dir`: Override directory for recipe storage.