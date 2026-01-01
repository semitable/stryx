from __future__ import annotations
import pytest
from unittest.mock import MagicMock
import typer
from pydantic import BaseModel, ConfigDict
from stryx.utils import Ctx, write_yaml
from stryx.commands import recipe_init, recipe_show, run_show

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

def test_show_recipe(mock_ctx, capsys):
    """Test showing a recipe configuration."""
    recipe_init(mock_ctx, name="base", overrides=["value=10"])
    
    recipe_show(mock_ctx, name="base")
    
    captured = capsys.readouterr()
    assert 'value: 10' in captured.out
    assert '(recipe)' in captured.out

def test_show_implicit_overrides(mock_ctx, capsys):
    """Test showing defaults with implicit overrides."""
    recipe_show(mock_ctx, name="value=99")
    
    captured = capsys.readouterr()
    assert 'value: 99' in captured.out
    assert '(override ← 1)' in captured.out

def test_show_run(mock_ctx, capsys):
    """Test `run show <id>`."""
    run_dir = mock_ctx.obj.runs_dir / "run_abc"
    run_dir.mkdir()
    write_yaml(run_dir / "stryx.manifest.yaml", {
        "status": "COMPLETED",
    })
    write_yaml(run_dir / "stryx.config.yaml", {"value": 123})
    
    run_show(mock_ctx, run_id="run_abc")
    
    captured = capsys.readouterr()
    assert "Run: run_abc" in captured.out
    assert "value: 123" in captured.out

def test_show_run_missing(mock_ctx):
    with pytest.raises(typer.Exit):
        run_show(mock_ctx, run_id="missing")