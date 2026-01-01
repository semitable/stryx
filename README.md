# Stryx

**Stryx** is a typed configuration compiler and experiment manager for Machine Learning. It provides a lightweight, composable approach to experiment tracking and configuration, using Pydantic for validation and YAML for persistence.

## Key Features

*   **Type-First:** Configurations are defined as Pydantic models, providing IDE support and validation.
*   **Configs as Source:** Experiment configurations are compiled into static YAML "configs" that include metadata and lineage.
*   **Minimal Surface:** A single decorator (`@stryx.cli`) adds powerful CLI capabilities to any script.
*   **Reproducibility:** Defaults are code-driven, overrides are explicit, and every run is recorded with a manifest (git state, logs, absolute paths).
*   **Lifecycle Management:** Automatic recording of status, results, and system metadata for every execution.

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

Your script now has a structured CLI.

### 1. Running Experiments (`run`)
Execute an experiment. If you provide overrides, Stryx automatically creates a "scratch" config to track the variant.

```bash
# Run with defaults
python train.py run

# Run with overrides (creates a scratch config)
python train.py run lr=1e-3 epochs=50

# Run a specific named config
python train.py run my_exp
```

### 2. Config Management (`configs`)
Manage your experiment templates (recipes).

```bash
# Initialize a new config from defaults
python train.py configs init my_exp lr=1e-3

# Fork an existing config
python train.py configs fork my_exp better_exp lr=2e-4

# List all saved configs
python train.py configs list

# Inspect a config with source annotations
python train.py configs show my_exp

# Print the JSON schema for VS Code validation
python train.py configs schema --json
```

### 3. Run Management (`runs`)
Inspect and compare execution history.

```bash
# List all runs
python train.py runs list

# Show detailed info for a specific run
python train.py runs show run_20260101_120000

# Diff two runs to see what changed
python train.py runs diff run_A run_B
```

## CLI Reference

### Top-level Commands
| Command | Description |
| :--- | :--- |
| `run [config] [ov...]` | Execute an experiment (alias for `runs exec`). |
| `configs` | Subcommands for managing configuration files. |
| `runs` | Subcommands for managing execution history. |

### Configs Subcommands (`configs ...`)
| Command | Description |
| :--- | :--- |
| `init [name] [ov...]` | Create a fresh config file from defaults. |
| `fork <src> <name> [ov...]`| Fork an existing config with modifications. |
| `list` | List all saved configs and scratches. |
| `show [name] [ov...]` | Display config with source annotations. |
| `diff <A> [B]` | Compare two configs (or A vs defaults). |
| `schema` | Show the configuration schema (use `--json` for files). |

### Runs Subcommands (`runs ...`)
| Command | Description |
| :--- | :--- |
| `exec [config] [ov...]` | Execute an experiment. |
| `list` | List execution history and statuses. |
| `show <run_id>` | Show configuration and metadata for a run. |
| `diff <id_A> <id_B>` | Compare configuration between two runs. |

### Global Options
*   `--runs-dir`: Override directory for run logs and manifests (default: `runs/`).
*   `--configs-dir`: Override directory for config storage (default: `configs/`).
