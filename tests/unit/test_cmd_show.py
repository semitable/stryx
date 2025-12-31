from __future__ import annotations
import pytest
from unittest.mock import MagicMock
import typer
from pydantic import BaseModel, ConfigDict
from stryx.utils import Ctx, write_yaml
from stryx.commands import cmd_new, cmd_show

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

def test_show_basic(mock_ctx, capsys):
    """Test showing a recipe configuration."""
    cmd_new(mock_ctx, recipe="base", overrides=["value=10"])
    
    cmd_show(mock_ctx, target="base")
    
    captured = capsys.readouterr()
    assert 'value: 10' in captured.out
    assert '(recipe)' in captured.out

def test_show_with_overrides(mock_ctx, capsys):
    """Test showing with CLI overrides."""
    cmd_new(mock_ctx, recipe="base", overrides=["value=10"])
    
    cmd_show(mock_ctx, target="base", overrides=["value=99"])
    
    captured = capsys.readouterr()
    assert 'value: 99' in captured.out
    assert '(override ← 10)' in captured.out

def test_show_defaults(mock_ctx, capsys):
    """Test showing defaults (no recipe)."""
    cmd_show(mock_ctx)
    
    captured = capsys.readouterr()
    assert 'value: 1' in captured.out
    assert '(default)' in captured.out

def test_show_implicit_overrides(mock_ctx, capsys):
    """Test showing defaults with implicit overrides (no recipe name)."""
    # stryx show value=99
    cmd_show(mock_ctx, target="value=99")
    
    captured = capsys.readouterr()
    assert 'value: 99' in captured.out
    assert '(override ← 1)' in captured.out # override vs default

def test_show_explicit_path(mock_ctx, capsys):
    """Test showing a config from an explicit path."""
    path = mock_ctx.obj.configs_dir / "external.yaml"
    write_yaml(path, {"value": 55})
    
    cmd_show(mock_ctx, target=str(path))
    
    captured = capsys.readouterr()
    assert 'value: 55' in captured.out
    assert 'recipe: external' in captured.out