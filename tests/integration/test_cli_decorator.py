"""Integration tests for stryx CLI functionality."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from pydantic import BaseModel, Field

import stryx
from stryx.config_builder import build_config, load_and_override
from stryx.utils import write_yaml

class TrainConfig(BaseModel):
    steps: int = 100
    batch_size: int = 32

class Config(BaseModel):
    name: str = "test"
    lr: float = 1e-4
    train: TrainConfig = Field(default_factory=TrainConfig)

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
        # Assuming run_id derivation logic respects env vars which we unset
        monkeypatch.setenv("PYTHONHASHSEED", "0", prepend=False) 

        monkeypatch.setattr(sys, "argv", ["train.py", "create-run-id"])
        main()

        out = capsys.readouterr().out.strip()
        assert out.startswith("run_")
        assert not calls

    def test_resolved_config_written(self, monkeypatch, tmp_path):
        """run writes resolved config."""
        calls = []

        @stryx.cli(schema=Config, recipes_dir=tmp_path / "configs", runs_dir=tmp_path / "runs")
        def main(cfg: Config):
            calls.append(cfg.lr)

        # 'run' strict command requires a recipe now. 
        # But wait, did we test 'stryx run' with no args?
        # In parser default is 'try'.
        # 'stryx try' writes config to scratches/ and runs/id/config.yaml.
        # Let's verify 'try' (default behavior for no args).
        
        monkeypatch.setattr(sys, "argv", ["train.py"]) # Implicit 'try'
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

    def test_schema_command(self, monkeypatch, capsys):
        """schema command prints the schema and fields."""
        calls = []

        @stryx.cli(schema=Config)
        def main(cfg: Config):
            calls.append(cfg)

        monkeypatch.setattr(sys, "argv", ["train.py", "schema"])
        main()

        out = capsys.readouterr().out
        assert "Schema: tests.integration.test_cli_decorator:Config" in out
        assert "Fields:" in out
        assert "lr: float = 1.00e-04" in out
        assert not calls

    def test_configs_dir_override(self, monkeypatch, tmp_path):
        """--configs-dir writes recipes to the overridden directory."""
        recipes_dir = tmp_path / "alt_configs"

        @stryx.cli(schema=Config)
        def main(cfg: Config):
            raise AssertionError("user function should not run for 'fork'/'new'")

        # Use 'new' or 'fork defaults'
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

class TestRecipeIO:
    """Test recipe save/load functionality."""

    def test_save_and_load_recipe(self):
        """Can save a recipe and load it back."""
        with tempfile.TemporaryDirectory() as tmpdir:
            recipe_path = Path(tmpdir) / "test.yaml"

            # Build and save
            cfg = build_config(Config, ["lr=0.001"])
            write_yaml(recipe_path, cfg.model_dump())

            # Load back
            loaded = load_and_override(Config, recipe_path, [])

            assert loaded.lr == 0.001

    def test_load_with_overrides(self):
        """Can load recipe and apply additional overrides."""
        with tempfile.TemporaryDirectory() as tmpdir:
            recipe_path = Path(tmpdir) / "test.yaml"

            # Save with lr=0.001
            cfg = build_config(Config, ["lr=0.001"])
            write_yaml(recipe_path, cfg.model_dump())

            # Load and override lr
            loaded = load_and_override(Config, recipe_path, ["lr=0.0001"])

            assert loaded.lr == 0.0001
