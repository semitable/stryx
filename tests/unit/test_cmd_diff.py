from __future__ import annotations
import argparse
import pytest
from pathlib import Path
from pydantic import BaseModel, ConfigDict
from stryx.context import Ctx
from stryx.commands import cmd_new, cmd_diff

class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = "default"
    value: int = 1

@pytest.fixture
def ctx(tmp_path):
    """Create a context with temporary directories."""
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()
    return Ctx(
        schema=Config,
        configs_dir=configs_dir,
        runs_dir=tmp_path / "runs",
        func=lambda x: None
    )

def test_diff_two_recipes(ctx, capsys):
    """Test diffing two recipes."""
    cmd_new(ctx, argparse.Namespace(recipe="r1", overrides=["value=10"], message=None, force=False))
    cmd_new(ctx, argparse.Namespace(recipe="r2", overrides=["value=20"], message=None, force=False))
    
    ns = argparse.Namespace(recipe_a="r1", recipe_b="r2")
    cmd_diff(ctx, ns)
    
    captured = capsys.readouterr()
    assert "Diff: r1 vs r2" in captured.out
    assert "~ value: 10 -> 20" in captured.out

def test_diff_vs_defaults(ctx, capsys):
    """Test diffing a recipe against defaults."""
    cmd_new(ctx, argparse.Namespace(recipe="r1", overrides=["value=10"], message=None, force=False))
    
    ns = argparse.Namespace(recipe_a="r1", recipe_b=None)
    cmd_diff(ctx, ns)
    
    captured = capsys.readouterr()
    assert "Diff: r1 vs (defaults)" in captured.out
    assert "~ value: 10 -> 1" in captured.out

def test_diff_no_differences(ctx, capsys):
    """Test diffing identical recipes."""
    cmd_new(ctx, argparse.Namespace(recipe="r1", overrides=["value=10"], message=None, force=False))
    cmd_new(ctx, argparse.Namespace(recipe="r2", overrides=["value=10"], message=None, force=False))
    
    ns = argparse.Namespace(recipe_a="r1", recipe_b="r2")
    cmd_diff(ctx, ns)
    
    captured = capsys.readouterr()
    assert "No differences found" in captured.out
