"""Integration tests for stryx CLI functionality.

These tests verify the public interface works end-to-end.
They focus on user-facing behavior, not implementation details.
"""
from __future__ import annotations

import tempfile
import sys
import os
from pathlib import Path

import pytest
from pydantic import BaseModel, Field

import stryx


# =============================================================================
# Test Schema
# =============================================================================


class TrainConfig(BaseModel):
    steps: int = 100
    batch_size: int = 32


class Config(BaseModel):
    name: str = "test"
    lr: float = 1e-4
    train: TrainConfig = Field(default_factory=TrainConfig)


# =============================================================================
# Tests
# =============================================================================


class TestCLIDecorator:
    """Test the @stryx.cli decorator."""

    def test_run_with_defaults(self):
        """Decorated function receives config with defaults."""
        received = []

        @stryx.cli(schema=Config, recipes_dir="configs")
        def main(cfg: Config):
            received.append(cfg)

        # Call directly with a config (bypasses CLI parsing)
        main(Config())

        assert len(received) == 1
        assert received[0].name == "test"
        assert received[0].lr == 1e-4

    def test_run_with_config_object(self):
        """Can call decorated function with config directly."""
        result = []

        @stryx.cli(schema=Config)
        def main(cfg: Config):
            result.append(cfg.lr)

        # Direct call with config object
        main(Config(lr=0.001))
        assert result[0] == 0.001

    def test_create_run_id_command(self, monkeypatch, capsys):
        """create-run-id prints an id and does not run the user function."""
        calls = []

        @stryx.cli(schema=Config)
        def main(cfg: Config):
            calls.append(cfg)

        monkeypatch.setenv("STRYX_RUN_ID", "", prepend=False)
        monkeypatch.setenv("TORCHELASTIC_RUN_ID", "", prepend=False)
        monkeypatch.setenv("SLURM_JOB_ID", "", prepend=False)

        monkeypatch.setenv("PYTHONHASHSEED", "0", prepend=False)  # keep env stable
        monkeypatch.setenv("WORLD_SIZE", "", prepend=False)

        monkeypatch.setattr(sys, "argv", ["train.py", "create-run-id"])
        main()

        out = capsys.readouterr().out.strip()
        assert out.startswith("run_")
        assert not calls  # ensure user function not called

    def test_resolved_config_written(self, monkeypatch, tmp_path):
        """run writes resolved config."""
        calls = []

        @stryx.cli(schema=Config, recipes_dir=tmp_path / "configs", runs_dir=tmp_path / "runs")
        def main(cfg: Config):
            calls.append(cfg.lr)

        monkeypatch.setattr(sys, "argv", ["train.py"])
        main()

        run_dirs = sorted((tmp_path / "runs").iterdir())
        assert run_dirs, "run directory should exist"
        run_root = run_dirs[0]
        resolved = run_root / "config.yaml"
        assert resolved.exists()

        data = stryx.utils.read_yaml(resolved)
        assert data["lr"] == 1e-4

    def test_create_run_id_help(self, monkeypatch, capsys):
        """create-run-id --help shows help and does not run the user function."""
        calls = []

        @stryx.cli(schema=Config)
        def main(cfg: Config):
            calls.append(cfg)

        monkeypatch.setattr(sys, "argv", ["train.py", "create-run-id", "--help"])
        main()

        out = capsys.readouterr().out
        assert "run id" in out.lower()
        assert not calls

    def test_configs_dir_override(self, monkeypatch, tmp_path):
        """--configs-dir writes recipes to the overridden directory."""
        recipes_dir = tmp_path / "alt_configs"

        @stryx.cli(schema=Config)
        def main(cfg: Config):
            raise AssertionError("user function should not run for 'new'")

        monkeypatch.setattr(
            sys,
            "argv",
            ["train.py", "--configs-dir", str(recipes_dir), "new", "custom", "lr=0.2"],
        )
        main()

        recipe_path = recipes_dir / "custom.yaml"
        assert recipe_path.exists()
        data = stryx.utils.read_yaml(recipe_path)
        assert data["lr"] == 0.2


class TestFromDataclass:
    """Test stryx.from_dataclass() conversion."""

    def test_simple_dataclass(self):
        """Simple dataclass converts to working Pydantic model."""
        from dataclasses import dataclass

        @dataclass
        class SimpleConfig:
            lr: float = 1e-4
            name: str = "simple"

        Model = stryx.from_dataclass(SimpleConfig)

        # Can instantiate with defaults
        instance = Model()
        assert instance.lr == 1e-4
        assert instance.name == "simple"

        # Can override
        instance = Model(lr=0.001)
        assert instance.lr == 0.001

    def test_nested_dataclass(self):
        """Nested dataclasses convert recursively."""
        from dataclasses import dataclass, field

        @dataclass
        class Inner:
            x: int = 1

        @dataclass
        class Outer:
            inner: Inner = field(default_factory=Inner)

        Model = stryx.from_dataclass(Outer)
        instance = Model()

        assert instance.inner.x == 1

    def test_consistent_types(self):
        """Same dataclass converts to same Pydantic model."""
        from dataclasses import dataclass

        @dataclass
        class MyConfig:
            value: int = 42

        Model1 = stryx.from_dataclass(MyConfig)
        Model2 = stryx.from_dataclass(MyConfig)

        # Should be the exact same class (cached)
        assert Model1 is Model2

    def test_shared_nested_types(self):
        """Nested dataclasses shared across parents get same type."""
        from dataclasses import dataclass, field

        @dataclass
        class Shared:
            value: int = 1

        @dataclass
        class Parent1:
            shared: Shared = field(default_factory=Shared)

        @dataclass
        class Parent2:
            shared: Shared = field(default_factory=Shared)

        Model1 = stryx.from_dataclass(Parent1)
        Model2 = stryx.from_dataclass(Parent2)

        # The Shared type should be the same in both
        instance1 = Model1()
        instance2 = Model2()

        # Both should accept each other's inner type
        assert type(instance1.shared) is type(instance2.shared)


class TestConfigBuilding:
    """Test config building with overrides."""

    def test_build_with_overrides(self):
        """Config builds correctly with CLI-style overrides."""
        from stryx.decorator import _build_config

        cfg = _build_config(Config, ["lr=0.001", "name=experiment"])

        assert cfg.lr == 0.001
        assert cfg.name == "experiment"

    def test_nested_overrides(self):
        """Nested fields can be overridden with dot notation."""
        from stryx.decorator import _build_config

        cfg = _build_config(Config, ["train.steps=500", "train.batch_size=64"])

        assert cfg.train.steps == 500
        assert cfg.train.batch_size == 64


class TestRecipeIO:
    """Test recipe save/load functionality."""

    def test_save_and_load_recipe(self):
        """Can save a recipe and load it back."""
        from stryx.decorator import _build_config, _load_and_override
        from stryx.utils import write_yaml

        with tempfile.TemporaryDirectory() as tmpdir:
            recipe_path = Path(tmpdir) / "test.yaml"

            # Build and save
            cfg = _build_config(Config, ["lr=0.001"])
            write_yaml(recipe_path, cfg.model_dump())

            # Load back
            loaded = _load_and_override(Config, recipe_path, [])

            assert loaded.lr == 0.001

    def test_load_with_overrides(self):
        """Can load recipe and apply additional overrides."""
        from stryx.decorator import _build_config, _load_and_override
        from stryx.utils import write_yaml

        with tempfile.TemporaryDirectory() as tmpdir:
            recipe_path = Path(tmpdir) / "test.yaml"

            # Save with lr=0.001
            cfg = _build_config(Config, ["lr=0.001"])
            write_yaml(recipe_path, cfg.model_dump())

            # Load and override lr
            loaded = _load_and_override(Config, recipe_path, ["lr=0.0001"])

            assert loaded.lr == 0.0001
