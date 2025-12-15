"""SFT config example using the real TRL SFTConfig.

Demonstrates wrapping a HuggingFace dataclass with Pydantic for use with Stryx.

Usage:
    uv run --extra examples python examples/sft.py
    uv run --extra examples python examples/sft.py learning_rate=1e-5
    uv run --extra examples python examples/sft.py new my_exp learning_rate=1e-5
"""
from __future__ import annotations

import dataclasses
from typing import Any

from pydantic import BaseModel, ConfigDict
from trl import SFTConfig as TRLSFTConfig

import stryx


def _get_dataclass_defaults(cls: type) -> dict[str, Any]:
    """Extract default values from a dataclass."""
    defaults = {}
    for field in dataclasses.fields(cls):
        if field.default is not dataclasses.MISSING:
            defaults[field.name] = field.default
        elif field.default_factory is not dataclasses.MISSING:
            defaults[field.name] = field.default_factory()
    return defaults


# Get defaults from the real TRL SFTConfig
_TRL_DEFAULTS = _get_dataclass_defaults(TRLSFTConfig)


class SFTConfig(BaseModel):
    """Pydantic wrapper around TRL's SFTConfig (151 fields).

    This wraps the real trl.SFTConfig dataclass, allowing it to work with Stryx.
    All 151 fields from TrainingArguments + SFT-specific options are available.
    """

    model_config = ConfigDict(extra="allow")  # Allow all TRL fields

    def __init__(self, **data):
        # Merge with TRL defaults
        merged = {**_TRL_DEFAULTS, **data}
        super().__init__(**merged)

    def to_trl_config(self) -> TRLSFTConfig:
        """Convert back to TRL's SFTConfig for use with SFTTrainer."""
        return TRLSFTConfig(**self.model_dump())


def train(cfg: SFTConfig) -> None:
    """Simulated SFT training showing config values."""
    print("=" * 60)
    print("SFT Training Configuration")
    print("=" * 60)

    # Show key training params
    d = cfg.model_dump()
    print(f"output_dir: {d.get('output_dir')}")
    print(f"num_train_epochs: {d.get('num_train_epochs')}")
    print(f"per_device_train_batch_size: {d.get('per_device_train_batch_size')}")
    print(f"gradient_accumulation_steps: {d.get('gradient_accumulation_steps')}")
    print(f"learning_rate: {d.get('learning_rate')}")
    print(f"lr_scheduler_type: {d.get('lr_scheduler_type')}")
    print(f"warmup_ratio: {d.get('warmup_ratio')}")
    print(f"bf16: {d.get('bf16')}")
    print(f"gradient_checkpointing: {d.get('gradient_checkpointing')}")
    print()
    print("SFT-specific:")
    print(f"max_seq_length: {d.get('max_seq_length')}")
    print(f"packing: {d.get('packing')}")
    print(f"dataset_text_field: {d.get('dataset_text_field')}")
    print()
    print(f"Total config fields: {len(d)}")

    # Show how to convert back to TRL
    print()
    print("To use with TRL:")
    print("  trl_config = cfg.to_trl_config()")
    print("  trainer = SFTTrainer(args=trl_config, ...)")


@stryx.cli(schema=SFTConfig)
def main(cfg: SFTConfig) -> None:
    """Run SFT training with the given config."""
    train(cfg)


if __name__ == "__main__":
    main()
