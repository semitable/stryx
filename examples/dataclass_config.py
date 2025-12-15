"""Example using plain Python dataclasses with Stryx.

This demonstrates stryx.from_dataclass() which converts dataclasses to Pydantic,
giving you the simplicity of dataclasses with full Stryx features.

Usage:
    python examples/dataclass_config.py
    python examples/dataclass_config.py lr=1e-5 train.epochs=20
    python examples/dataclass_config.py new my_exp lr=1e-5
    python examples/dataclass_config.py edit my_exp
"""
from __future__ import annotations

from dataclasses import dataclass, field

import stryx


# =============================================================================
# Config Definition (plain dataclasses - no Pydantic imports needed!)
# =============================================================================


@dataclass
class TrainConfig:
    """Training hyperparameters."""

    epochs: int = 10
    batch_size: int = 32
    seed: int = 42


@dataclass
class OptimizerConfig:
    """Optimizer settings."""

    lr: float = 1e-4
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.999)


@dataclass
class Config:
    """Main experiment configuration."""

    exp_name: str = "dataclass_demo"
    model: str = "bert-base"

    # Nested dataclasses work!
    train: TrainConfig = field(default_factory=TrainConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)

    # Optional nested dataclass
    eval_config: TrainConfig | None = None


# =============================================================================
# Training Logic
# =============================================================================


def train(cfg: Config) -> None:
    """Simulated training showing config values."""
    print("=" * 50)
    print(f"Experiment: {cfg.exp_name}")
    print("=" * 50)
    print(f"Model: {cfg.model}")
    print()
    print("Training:")
    print(f"  epochs: {cfg.train.epochs}")
    print(f"  batch_size: {cfg.train.batch_size}")
    print(f"  seed: {cfg.train.seed}")
    print()
    print("Optimizer:")
    print(f"  lr: {cfg.optimizer.lr}")
    print(f"  weight_decay: {cfg.optimizer.weight_decay}")
    print(f"  betas: {cfg.optimizer.betas}")
    print()
    print(f"Eval config: {cfg.eval_config}")
    print()
    print(f"Config type: {type(cfg).__name__}")
    print(f"(Pydantic model converted from dataclass)")


# =============================================================================
# Entry Point
# =============================================================================


# Convert dataclass to Pydantic model for Stryx
ConfigModel = stryx.from_dataclass(Config)


@stryx.cli(schema=ConfigModel)
def main(cfg) -> None:
    """Run training with the given config."""
    train(cfg)


if __name__ == "__main__":
    main()
