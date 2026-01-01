from __future__ import annotations

from typing import Literal, Union, Any

from pydantic import BaseModel, ConfigDict, Field

import stryx


class TrainCfg(BaseModel):
    """Training hyperparameters."""

    batch_size: int = 128
    steps: int = 200
    seed: int = 0


class AdamWCfg(BaseModel):
    """AdamW optimizer configuration."""

    kind: Literal["adamw"] = "adamw"
    lr: float = 3e-4
    weight_decay: float = 0.05


class SGDCfg(BaseModel):
    """SGD optimizer configuration."""

    kind: Literal["sgd"] = "sgd"
    lr: float = 1e-2
    momentum: float = 0.9


OptimizerCfg = Union[AdamWCfg, SGDCfg]


class CosineSchedCfg(BaseModel):
    """Cosine annealing scheduler."""

    kind: Literal["cosine"] = "cosine"
    warmup_steps: int = 50
    min_lr: float = 1e-5


class LinearSchedCfg(BaseModel):
    """Linear warmup scheduler."""

    kind: Literal["linear"] = "linear"
    warmup_steps: int = 50


SchedulerCfg = Union[CosineSchedCfg, LinearSchedCfg]


class Config(BaseModel):
    """Main experiment configuration."""

    model_config = ConfigDict(extra="forbid")

    exp_name: str = "demo"
    model_name: str = "resnet50"
    dataset: str = "cifar10"

    train: TrainCfg = Field(default_factory=TrainCfg)
    optim: OptimizerCfg = Field(default_factory=AdamWCfg, discriminator="kind")
    sched: SchedulerCfg | None = Field(default=None, discriminator="kind")


def train(cfg: Config) -> dict[str, Any]:
    """Simulated training loop."""

    print(f"[{cfg.exp_name}] Starting training...")
    print(f"  Model: {cfg.model_name}")
    print(f"  Dataset: {cfg.dataset}")
    print(f"  Optimizer: {cfg.optim.kind} (lr={cfg.optim.lr})")
    print(f"  Scheduler: {cfg.sched.kind if cfg.sched else 'none'}")
    print()

    for step in range(cfg.train.steps):
        loss = 1.0 / (step + 1)

        if step % max(1, cfg.train.steps // 5) == 0:
            print(f"  step={step:04d} loss={loss:.6f}")

    return {"accuracy": 0.8}


@stryx.cli(schema=Config)
def main(cfg: Config) -> dict[str, Any]:
    """Train a model with the given configuration."""
    return train(cfg)


if __name__ == "__main__":
    main()
