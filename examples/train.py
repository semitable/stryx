from __future__ import annotations

from pathlib import Path
from typing import Literal, Union

from pydantic import BaseModel, ConfigDict, Field

import stryx


# =========================
# Schema Definition
# =========================

class TrainCfg(BaseModel):
    batch_size: int = 128
    steps: int = 200
    seed: int = 0
    out_dir: str = "outputs"


class AdamWCfg(BaseModel):
    kind: Literal["adamw"] = "adamw"
    lr: float = 3e-4
    weight_decay: float = 0.05


class SGDCfg(BaseModel):
    kind: Literal["sgd"] = "sgd"
    lr: float = 1e-2
    momentum: float = 0.9


OptimizerCfg = Union[AdamWCfg, SGDCfg]


class CosineSchedCfg(BaseModel):
    kind: Literal["cosine"] = "cosine"
    warmup_steps: int = 50
    min_lr: float = 1e-5


class LinearSchedCfg(BaseModel):
    kind: Literal["linear"] = "linear"
    warmup_steps: int = 50


SchedulerCfg = Union[CosineSchedCfg, LinearSchedCfg]


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    exp_name: str = "demo"
    model_name: str = "resnet50"
    dataset: str = "cifar10"

    train: TrainCfg = TrainCfg()
    optim: OptimizerCfg = Field(discriminator="kind")
    sched: SchedulerCfg | None = Field(default=None, discriminator="kind")


# =========================
# Training Logic
# =========================

def train(cfg: Config) -> None:
    out_dir = Path(cfg.train.out_dir) / cfg.exp_name
    out_dir.mkdir(parents=True, exist_ok=True)

    for step in range(cfg.train.steps):
        loss = 1.0 / (step + 1)

        if step % max(1, cfg.train.steps // 10) == 0:
            print(
                f"[{cfg.exp_name}] step={step:04d} loss={loss:.6f} "
                f"model={cfg.model_name} data={cfg.dataset} optim={cfg.optim.kind} "
                f"lr={getattr(cfg.optim, 'lr', None)}"
            )

    ckpt_path = out_dir / "checkpoint.pt"
    ckpt_path.write_bytes(b"pretend model bytes")
    print(f"Wrote artifact: {ckpt_path}")


# =========================
# CLI Interface
# =========================

def main() -> None:
    """
    Usage:
      # Create a compiled recipe
      python train.py new vit_big exp_name=vit_big model_name=vit_b optim.kind=adamw optim.lr=1e-4

      # Show compiled recipe
      python train.py show vit_big

      # Run from compiled recipe
      python train.py run vit_big
      # OR
      python train.py --config conf/recipes/vit_big.yaml
    """
    cmd = stryx.recipe_cli(
        schema=Config,
        recipes_dir="conf/recipes",
        argv=None,
    )

    if cmd.kind == "new":
        path = cmd.write()
        print(f"Wrote compiled recipe: {path}")
    elif cmd.kind == "show":
        print(cmd.text())
    elif cmd.kind in ("run", "config"):
        cfg = cmd.load()
        train(cfg)
    else:
        raise RuntimeError(f"Unhandled command kind: {cmd.kind}")


if __name__ == "__main__":
    main()
