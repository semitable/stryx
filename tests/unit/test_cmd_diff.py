from __future__ import annotations
import pytest
from unittest.mock import MagicMock
import typer
from pydantic import BaseModel, ConfigDict
from stryx.utils import Ctx
from stryx.commands import cmd_new, cmd_diff

class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = "default"
    value: int = 1

@pytest.fixture
def mock_ctx(tmp_path):
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    c = Ctx(
        schema=Config,
        configs_dir=configs_dir,
        runs_dir=runs_dir,
        func=lambda x: None
    )
    ctx = MagicMock(spec=typer.Context)
    ctx.obj = c
    return ctx

def test_diff_two_recipes(mock_ctx, capsys):
    """Test diffing two recipes."""
    cmd_new(mock_ctx, recipe="r1", overrides=["value=10"])
    cmd_new(mock_ctx, recipe="r2", overrides=["value=20"])
    
    cmd_diff(mock_ctx, recipe_a="r1", recipe_b="r2")
    
    captured = capsys.readouterr()
    assert "Diff: r1 vs r2" in captured.out
    assert "~ value: 10 -> 20" in captured.out

def test_diff_vs_defaults(mock_ctx, capsys):
    """Test diffing a recipe against defaults."""
    cmd_new(mock_ctx, recipe="r1", overrides=["value=10"])
    
    cmd_diff(mock_ctx, recipe_a="r1")
    
    captured = capsys.readouterr()
    assert "Diff: r1 vs (defaults)" in captured.out
    assert "~ value: 10 -> 1" in captured.out

def test_diff_no_differences(mock_ctx, capsys):
    """Test diffing identical recipes."""
    cmd_new(mock_ctx, recipe="r1", overrides=["value=10"])
    cmd_new(mock_ctx, recipe="r2", overrides=["value=10"])
    
    cmd_diff(mock_ctx, recipe_a="r1", recipe_b="r2")
    
    captured = capsys.readouterr()
    assert "No differences found" in captured.out