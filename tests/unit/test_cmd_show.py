from __future__ import annotations
import argparse
import pytest
from pathlib import Path
from pydantic import BaseModel, ConfigDict
from stryx.context import Ctx
from stryx.commands import cmd_new, cmd_show

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

def test_show_basic(ctx, capsys):
    """Test showing a recipe configuration."""
    cmd_new(ctx, argparse.Namespace(recipe="base", overrides=["value=10"], message=None, force=False))
    
    # Show it
    ns = argparse.Namespace(recipe="base", overrides=[], config_path=None)
    cmd_show(ctx, ns)
    
    captured = capsys.readouterr()
    assert 'value: 10' in captured.out
    assert '(recipe)' in captured.out
    assert 'name: "default"' in captured.out
    assert '(default)' in captured.out

def test_show_with_overrides(ctx, capsys):
    """Test showing with CLI overrides."""
    cmd_new(ctx, argparse.Namespace(recipe="base", overrides=["value=10"], message=None, force=False))
    
    # Show with override
    ns = argparse.Namespace(recipe="base", overrides=["value=99"], config_path=None)
    cmd_show(ctx, ns)
    
    captured = capsys.readouterr()
    assert 'value: 99' in captured.out
    assert '(override ← 10)' in captured.out
    assert 'Config (recipe: base, 1 override)' in captured.out

def test_show_defaults(ctx, capsys):
    """Test showing defaults (no recipe)."""
    ns = argparse.Namespace(recipe=None, overrides=[], config_path=None)
    cmd_show(ctx, ns)
    
    captured = capsys.readouterr()
    assert 'value: 1' in captured.out
    assert '(default)' in captured.out

def test_show_explicit_path(ctx, capsys):
    """Test showing a config from an explicit path."""
    # Create a file manually (simulating external config)
    path = ctx.configs_dir / "external.yaml"
    from stryx.utils import write_yaml
    write_yaml(path, {"value": 55})
    
    ns = argparse.Namespace(recipe=None, overrides=[], config_path=path)
    cmd_show(ctx, ns)
    
    captured = capsys.readouterr()
    assert 'value: 55' in captured.out
    assert 'recipe: external' in captured.out
