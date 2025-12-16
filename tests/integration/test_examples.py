"""Integration tests verifying examples work correctly.

These tests check that examples run and produce valid output.
We verify structure and key values, but avoid brittle string matching.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

EXAMPLES_DIR = Path(__file__).parent.parent.parent / "examples"


def run(script: str, args: list[str] = None, cwd: str = None) -> subprocess.CompletedProcess:
    """Run an example script."""
    cmd = [sys.executable, str(EXAMPLES_DIR / script)]
    if args:
        cmd.extend(args)
    return subprocess.run(cmd, capture_output=True, text=True, cwd=cwd, timeout=30)


def load_recipe(path: Path) -> dict:
    """Load and parse a recipe YAML file."""
    return yaml.safe_load(path.read_text())


class TestTrainExample:
    """examples/train.py works."""

    def test_runs(self, tmp_path):
        result = run("train.py", cwd=tmp_path)
        assert result.returncode == 0
        assert result.stdout  # produces output

    def test_with_overrides(self, tmp_path):
        result = run("train.py", ["train.steps=5"], cwd=tmp_path)
        assert result.returncode == 0

    def test_help(self):
        result = run("train.py", ["--help"])
        assert result.returncode == 0
        assert result.stdout  # help text is printed

    def test_show(self, tmp_path):
        result = run("train.py", ["show"], cwd=tmp_path)
        assert result.returncode == 0
        assert result.stdout

    def test_recipe_workflow(self, tmp_path):
        """new -> list -> run -> show works and produces valid config."""
        # Create recipe with custom values
        result = run("train.py", ["new", "test", "train.steps=42"], cwd=tmp_path)
        assert result.returncode == 0

        # Recipe file exists and is valid YAML with expected structure
        recipe_path = tmp_path / "configs" / "test.yaml"
        assert recipe_path.exists()

        recipe = load_recipe(recipe_path)
        assert "__stryx__" in recipe  # has metadata
        assert recipe["train"]["steps"] == 42  # override was applied
        assert "exp_name" in recipe  # has expected fields

        # Can list, run, and show the recipe
        assert run("train.py", ["list"], cwd=tmp_path).returncode == 0
        assert run("train.py", ["run", "test"], cwd=tmp_path).returncode == 0
        assert run("train.py", ["show", "test"], cwd=tmp_path).returncode == 0

    def test_new_from(self, tmp_path):
        """new --from copies values from base recipe."""
        # Create base with custom value
        run("train.py", ["new", "base", "train.steps=100"], cwd=tmp_path)

        # Derive from it with another override
        result = run("train.py", ["new", "derived", "--from", "base", "train.batch_size=64"], cwd=tmp_path)
        assert result.returncode == 0

        # Derived recipe has both values
        recipe = load_recipe(tmp_path / "configs" / "derived.yaml")
        assert recipe["train"]["steps"] == 100  # from base
        assert recipe["train"]["batch_size"] == 64  # new override
        assert recipe["__stryx__"].get("from") == "base"  # tracks lineage

    def test_auto_naming(self, tmp_path):
        """new without name auto-generates sequential names."""
        # Create first auto-named recipe
        result = run("train.py", ["new", "train.steps=10"], cwd=tmp_path)
        assert result.returncode == 0
        assert (tmp_path / "configs" / "exp_001.yaml").exists()

        # Create second - should be exp_002
        result = run("train.py", ["new", "train.steps=20"], cwd=tmp_path)
        assert result.returncode == 0
        assert (tmp_path / "configs" / "exp_002.yaml").exists()

        # Verify values are correct
        recipe1 = load_recipe(tmp_path / "configs" / "exp_001.yaml")
        recipe2 = load_recipe(tmp_path / "configs" / "exp_002.yaml")
        assert recipe1["train"]["steps"] == 10
        assert recipe2["train"]["steps"] == 20

        # Create second - should be exp_002
        result = run("train.py", ["new", "train.steps=20"], cwd=tmp_path)
        assert result.returncode == 0
        assert (tmp_path / "configs" / "exp_002.yaml").exists()

        # Verify values are correct
        recipe1 = load_recipe(tmp_path / "configs" / "exp_001.yaml")
        recipe2 = load_recipe(tmp_path / "configs" / "exp_002.yaml")
        assert recipe1["train"]["steps"] == 10
        assert recipe2["train"]["steps"] == 20


class TestDataclassExample:
    """examples/dataclass_config.py works."""

    def test_runs(self, tmp_path):
        result = run("dataclass_config.py", cwd=tmp_path)
        assert result.returncode == 0
        assert result.stdout

    def test_with_overrides(self, tmp_path):
        result = run("dataclass_config.py", ["optimizer.lr=1e-5"], cwd=tmp_path)
        assert result.returncode == 0

    def test_recipe_has_nested_structure(self, tmp_path):
        """Dataclass config produces valid nested YAML."""
        run("dataclass_config.py", ["new", "dc_test", "train.epochs=99"], cwd=tmp_path)

        recipe = load_recipe(tmp_path / "configs" / "dc_test.yaml")
        assert recipe["train"]["epochs"] == 99
        assert "optimizer" in recipe  # nested dataclass converted
        assert "lr" in recipe["optimizer"]
