# Getting Started Tutorial

Welcome to Stryx! This guide will walk you through managing a machine learning experiment from scratch.

## 1. Define your Config

Stryx uses **Pydantic** models to define configuration. This gives you type safety and validation for free.

Create a file named `train.py`:

```python
from pydantic import BaseModel
import stryx

class Config(BaseModel):
    lr: float = 1e-4
    batch_size: int = 32
    model_name: str = "resnet18"

@stryx.cli(schema=Config)
def main(cfg: Config):
    # Your training logic here
    print(f"Training {cfg.model_name} with lr={cfg.lr}")

if __name__ == "__main__":
    main()
```

## 2. Create your first Recipe

Initialize a canonical recipe from your schema defaults.

```bash
uv run train.py new baseline
```

This creates `configs/baseline.yaml`.

## 3. Experiment (`try`)

Now you want to try a higher learning rate. Don't edit the file manually! Use `try` to run a **scratchpad** experiment.

```bash
uv run train.py try lr=1e-3
```

Stryx will:
1.  Auto-generate a name (e.g., `configs/scratches/2023...fast-badger.yaml`).
2.  Save the config with your override.
3.  Run the code.

## 4. Iterate & Edit

You can edit recipes interactively using the TUI.

```bash
uv run train.py edit baseline
```

Or edit a scratchpad you just ran:
```bash
uv run train.py edit scratches/fast-badger
```

## 5. Fork & Organise

You found that `lr=1e-3` works better. Let's make it a permanent experiment.

```bash
# Branch from 'baseline' to 'experiment_v2' with new settings
uv run train.py fork baseline experiment_v2 lr=1e-3
```

Now compare them to see exactly what changed:

```bash
uv run train.py diff baseline experiment_v2
```

See all your experiments in a smart table:

```bash
uv run train.py list
```

## 6. Reproduce (`run`)

When you are ready to launch the final training (or reproduce it later), use `run`.

```bash
uv run train.py run experiment_v2
```

**Note:** `run` is **strict**. It does not accept overrides. This guarantees that `experiment_v2` always means *exactly* what is in the file.

## 7. Distributed Training

Running on multiple GPUs? Stryx keeps you safe.

If you use `torchrun`, everything just works:
```bash
torchrun --nproc_per_node=4 train.py run experiment_v2
```

If you launch manually (e.g., separate SSH sessions), you **must** provide a shared ID so logs don't diverge:

```bash
# Generate an ID
export STRYX_RUN_ID=$(stryx create-run-id)

# Run on all nodes
python train.py run experiment_v2
```
