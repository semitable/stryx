from __future__ import annotations
import argparse
import pytest
from pydantic import BaseModel, ConfigDict
from stryx.commands import cmd_new, cmd_diff

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

def test_diff_two_recipes(base_ns, capsys):
    """Test diffing two recipes."""
    ns1 = argparse.Namespace(**vars(base_ns))
    ns1.recipe = "r1"
    ns1.overrides = ["value=10"]
    ns1.message = None
    ns1.force = False
    cmd_new(ns1)
    
    ns2 = argparse.Namespace(**vars(base_ns))
    ns2.recipe = "r2"
    ns2.overrides = ["value=20"]
    ns2.message = None
    ns2.force = False
    cmd_new(ns2)
    
    ns_diff = argparse.Namespace(**vars(base_ns))
    ns_diff.recipe_a = "r1"
    ns_diff.recipe_b = "r2"
    cmd_diff(ns_diff)
    
    captured = capsys.readouterr()
    assert "Diff: r1 vs r2" in captured.out
    assert "~ value: 10 -> 20" in captured.out

def test_diff_vs_defaults(base_ns, capsys):
    """Test diffing a recipe against defaults."""
    ns1 = argparse.Namespace(**vars(base_ns))
    ns1.recipe = "r1"
    ns1.overrides = ["value=10"]
    ns1.message = None
    ns1.force = False
    cmd_new(ns1)
    
    ns_diff = argparse.Namespace(**vars(base_ns))
    ns_diff.recipe_a = "r1"
    ns_diff.recipe_b = None
    cmd_diff(ns_diff)
    
    captured = capsys.readouterr()
    assert "Diff: r1 vs (defaults)" in captured.out
    assert "~ value: 10 -> 1" in captured.out

def test_diff_no_differences(base_ns, capsys):
    """Test diffing identical recipes."""
    ns1 = argparse.Namespace(**vars(base_ns))
    ns1.recipe = "r1"
    ns1.overrides = ["value=10"]
    ns1.message = None
    ns1.force = False
    cmd_new(ns1)
    
    ns2 = argparse.Namespace(**vars(base_ns))
    ns2.recipe = "r2"
    ns2.overrides = ["value=10"]
    ns2.message = None
    ns2.force = False
    cmd_new(ns2)
    
    ns_diff = argparse.Namespace(**vars(base_ns))
    ns_diff.recipe_a = "r1"
    ns_diff.recipe_b = "r2"
    cmd_diff(ns_diff)
    
    captured = capsys.readouterr()
    assert "No differences found" in captured.out