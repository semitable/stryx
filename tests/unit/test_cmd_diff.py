from __future__ import annotations
import pytest
from unittest.mock import MagicMock
import typer
from pydantic import BaseModel, ConfigDict
from stryx.utils import Ctx, write_yaml
from stryx.commands import recipe_init, recipe_diff, run_diff

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

def test_diff_recipes(mock_ctx, capsys):
    recipe_init(mock_ctx, name="r1", overrides=["value=10"])
    recipe_init(mock_ctx, name="r2", overrides=["value=20"])
    
    recipe_diff(mock_ctx, name_a="r1", name_b="r2")
    
    captured = capsys.readouterr()
    assert "~ value: 10 -> 20" in captured.out

def test_diff_runs(mock_ctx, capsys):
    """Test `run diff <id1> <id2>`."""
    def create_run(rid, val):
        d = mock_ctx.obj.runs_dir / rid
        d.mkdir()
        write_yaml(d / "manifest.yaml", {"config": {"value": val}})
        
    create_run("run_1", 100)
    create_run("run_2", 200)
    
    run_diff(mock_ctx, id_a="run_1", id_b="run_2")
    
    captured = capsys.readouterr()
    assert "Diff: run_1 vs run_2" in captured.out
    assert "~ value: 100 -> 200" in captured.out