"""Integration tests verifying examples run without errors.

These tests just check that the examples work - they don't verify
specific output content, which would be brittle and annoying to maintain.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).parent.parent.parent / "examples"


def run(script: str, args: list[str] = None, cwd: str = None) -> subprocess.CompletedProcess:
    """Run an example script."""
    cmd = [sys.executable, str(EXAMPLES_DIR / script)]
    if args:
        cmd.extend(args)
    return subprocess.run(cmd, capture_output=True, text=True, cwd=cwd, timeout=30)


class TestTrainExample:
    """examples/train.py works."""

    def test_runs(self, tmp_path):
        assert run("train.py", cwd=tmp_path).returncode == 0

    def test_with_overrides(self, tmp_path):
        assert run("train.py", ["train.steps=5"], cwd=tmp_path).returncode == 0

    def test_help(self):
        assert run("train.py", ["--help"]).returncode == 0

    def test_show(self, tmp_path):
        assert run("train.py", ["show"], cwd=tmp_path).returncode == 0

    def test_recipe_workflow(self, tmp_path):
        """new -> list -> run -> show works."""
        assert run("train.py", ["new", "test"], cwd=tmp_path).returncode == 0
        assert (tmp_path / "configs" / "test.yaml").exists()

        assert run("train.py", ["list"], cwd=tmp_path).returncode == 0
        assert run("train.py", ["run", "test"], cwd=tmp_path).returncode == 0
        assert run("train.py", ["show", "test"], cwd=tmp_path).returncode == 0

    def test_new_from(self, tmp_path):
        """new --from works."""
        run("train.py", ["new", "base"], cwd=tmp_path)
        assert run("train.py", ["new", "derived", "--from", "base"], cwd=tmp_path).returncode == 0


class TestDataclassExample:
    """examples/dataclass_config.py works."""

    def test_runs(self, tmp_path):
        assert run("dataclass_config.py", cwd=tmp_path).returncode == 0

    def test_with_overrides(self, tmp_path):
        assert run("dataclass_config.py", ["optimizer.lr=1e-5"], cwd=tmp_path).returncode == 0

    def test_show(self, tmp_path):
        assert run("dataclass_config.py", ["show"], cwd=tmp_path).returncode == 0
