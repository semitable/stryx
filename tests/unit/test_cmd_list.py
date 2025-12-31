from __future__ import annotations
import pytest
from unittest.mock import MagicMock
import typer
from pydantic import BaseModel, ConfigDict
from stryx.utils import Ctx, write_yaml
from stryx.commands import cmd_new, cmd_list_configs, cmd_list_runs

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

def test_list_configs(mock_ctx, capsys):
    """Test listing recipes."""
    cmd_new(mock_ctx, recipe="a", overrides=["value=10"])
    cmd_new(mock_ctx, recipe="b", overrides=["value=20"])
    
    cmd_list_configs(mock_ctx)
    
    captured = capsys.readouterr()
    assert "a" in captured.out
    assert "b" in captured.out
    assert "10" in captured.out
    assert "20" in captured.out

def test_list_runs(mock_ctx, capsys):
    """Test listing runs."""
    run1 = mock_ctx.obj.runs_dir / "run_1"
    run1.mkdir()
    write_yaml(run1 / "manifest.yaml", {
        "run_id": "run_1", 
        "status": "COMPLETED", 
        "created_at": "2023-01-01T12:00:00",
        "config": {"value": 100}
    })
    
    run2 = mock_ctx.obj.runs_dir / "run_2"
    run2.mkdir()
    write_yaml(run2 / "manifest.yaml", {
        "run_id": "run_2", 
        "status": "FAILED", 
        "created_at": "2023-01-02T12:00:00",
        "config": {"value": 200}
    })
    
    cmd_list_runs(mock_ctx)
    
    captured = capsys.readouterr()
    assert "run_1" in captured.out
    assert "COMPLETED" in captured.out
    assert "run_2" in captured.out
    assert "FAILED" in captured.out
    assert "100" in captured.out

def test_list_configs_resilience(mock_ctx, capsys):
    """Test that listing configs skips bad files gracefully."""
    cmd_new(mock_ctx, recipe="good")
    
    bad_path = mock_ctx.obj.configs_dir / "bad.yaml"
    bad_path.write_text(":: invalid yaml ::")
    
    cmd_list_configs(mock_ctx)
    
    captured = capsys.readouterr()
    assert "good" in captured.out
    assert "bad" not in captured.out
