from __future__ import annotations
import argparse
import pytest
from pydantic import BaseModel, ConfigDict
from stryx.commands import cmd_new, cmd_show

class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = "default"
    value: int = 1

@pytest.fixture
def base_ns(tmp_path):
    """Create a base namespace with context fields."""
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()
    return argparse.Namespace(
        stryx_schema=Config,
        configs_dir=configs_dir,
        runs_dir=tmp_path / "runs",
        stryx_func=lambda x: None
    )

def test_show_basic(base_ns, capsys):
    """Test showing a recipe configuration."""
    ns_new = argparse.Namespace(**vars(base_ns))
    ns_new.recipe = "base"
    ns_new.overrides = ["value=10"]
    ns_new.message = None
    ns_new.force = False
    cmd_new(ns_new)
    
    # Show it
    ns_show = argparse.Namespace(**vars(base_ns))
    ns_show.target = "base"
    ns_show.overrides = []
    cmd_show(ns_show)
    
    captured = capsys.readouterr()
    assert 'value: 10' in captured.out
    assert '(recipe)' in captured.out
    assert 'name: "default"' in captured.out
    assert '(default)' in captured.out

def test_show_with_overrides(base_ns, capsys):
    """Test showing with CLI overrides."""
    ns_new = argparse.Namespace(**vars(base_ns))
    ns_new.recipe = "base"
    ns_new.overrides = ["value=10"]
    ns_new.message = None
    ns_new.force = False
    cmd_new(ns_new)
    
    # Show with override
    ns_show = argparse.Namespace(**vars(base_ns))
    ns_show.target = "base"
    ns_show.overrides = ["value=99"]
    cmd_show(ns_show)
    
    captured = capsys.readouterr()
    assert 'value: 99' in captured.out
    assert '(override ← 10)' in captured.out
    assert 'Config (recipe: base, 1 override)' in captured.out

def test_show_defaults(base_ns, capsys):
    """Test showing defaults (no recipe)."""
    ns_show = argparse.Namespace(**vars(base_ns))
    ns_show.target = None
    ns_show.overrides = []
    cmd_show(ns_show)
    
    captured = capsys.readouterr()
    assert 'value: 1' in captured.out
    assert '(default)' in captured.out

def test_show_explicit_path(base_ns, capsys):
    """Test showing a config from an explicit path."""
    # Create a file manually (simulating external config)
    path = base_ns.configs_dir / "external.yaml"
    from stryx.utils import write_yaml
    write_yaml(path, {"value": 55})
    
    # Pass path as target
    ns_show = argparse.Namespace(**vars(base_ns))
    ns_show.target = str(path)
    ns_show.overrides = []
    cmd_show(ns_show)
    
    captured = capsys.readouterr()
    assert 'value: 55' in captured.out
    assert 'recipe: external' in captured.out