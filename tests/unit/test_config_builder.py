from __future__ import annotations
from pydantic import BaseModel, Field
from stryx.config_builder import build_config

class TrainConfig(BaseModel):
    steps: int = 100
    batch_size: int = 32

class Config(BaseModel):
    name: str = "test"
    lr: float = 1e-4
    train: TrainConfig = Field(default_factory=TrainConfig)

class TestConfigBuilding:
    """Test config building with overrides."""

    def test_build_with_overrides(self):
        """Config builds correctly with CLI-style overrides."""
        cfg = build_config(Config, ["lr=0.001", "name=experiment"])

        assert cfg.lr == 0.001
        assert cfg.name == "experiment"

    def test_nested_overrides(self):
        """Nested fields can be overridden with dot notation."""
        cfg = build_config(Config, ["train.steps=500", "train.batch_size=64"])

        assert cfg.train.steps == 500
        assert cfg.train.batch_size == 64
