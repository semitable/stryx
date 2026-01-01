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
    uv run examples/train.py configs new my_exp train.steps=100 optim.lr=0.001
    # Created recipe: configs/my_exp.yaml
    ```

*   **List all:**
    ```bash
    uv run examples/train.py configs list
    # stryx.name      stryx.created_at  optim.lr  train.steps
    #     my_exp      2026-01-01 15:31    0.0010          100
    ```

*   **Show details:**
    Shows the resolved configuration and the source of each value (default, recipe, or override).
    ```bash
    uv run examples/train.py configs show my_exp
    # Recipe: my_exp
    # ============================================================
    # exp_name: "demo"                             (default)
    # train:
    #   batch_size: 128                            (default)
    #   steps: 100                                 (recipe)
    # optim:
    #   lr: 0.001                                  (recipe)
    # ...
    ```

*   **Diff:**
    Compare two configurations to see what changed.
    ```bash
    uv run examples/train.py configs diff my_exp other_exp
    ```

*   **Fork:**
    Create a new recipe based on an existing one.
    ```bash
    uv run examples/train.py configs fork my_exp new_exp epochs=20
    ```

*   **Edit (Interactive):**
    Open a terminal UI to edit a configuration.
    ```bash
    uv run examples/train.py configs edit my_exp
    ```

### Running Experiments (`run` / `runs`)

Execute experiments and track their history.

*   **Run a recipe:**
    ```bash
    uv run examples/train.py run my_exp
    # [demo] Starting training...
    #   Model: resnet50
    #   Dataset: cifar10
    #   Optimizer: adamw (lr=0.001)
    #   ...
    #   step=0080 loss=0.012346
    ```

*   **Run with overrides:**
    This creates a temporary "scratch" config to ensure reproducibility without cluttering your main recipes.
    ```bash
    uv run examples/train.py run my_exp lr=0.02
    ```

*   **List history:**
    See all past execution runs.
    ```bash
    uv run examples/train.py runs list
    #                            stryx.run_id stryx.status stryx.created_at  accuracy
    #    run_20260101_153136_my_exp-big-aphid    COMPLETED 2026-01-01 15:31       0.8
    # ...
    ```

*   **Inspect a run:**
    View the exact configuration used for a past run.
    ```bash
    uv run examples/train.py runs show <run_id>
    ```

*   **Compare runs:**
    See how configuration differed between two runs.
    ```bash
    uv run examples/train.py runs diff <run_id_1> <run_id_2>
    ```

### Validation

Type checking is enforced automatically. Invalid inputs are caught early:

```bash
uv run examples/train.py configs new lr=astring
# Output:
# Config validation failed (building config):
# optim.adamw.lr: Input should be a valid number, unable to parse string as a number
```

### Capturing Results

If your decorated function returns a dictionary, Stryx automatically captures these values and stores them in `results.yaml` within the run directory. These results are then displayed in the run history.

```python
@stryx.cli(schema=Config)
def main(cfg: Config):
    # ... training logic ...
    return {"accuracy": 0.95, "loss": 0.05}
```

The returned keys (e.g., `accuracy`, `loss`) will appear as columns when you run `runs list`:

```bash
uv run examples/train.py runs list
# stryx.run_id ... accuracy  loss
# run_...      ...     0.95  0.05
```
