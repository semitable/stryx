from __future__ import annotations
import argparse
import pytest
from pydantic import BaseModel, ConfigDict
from stryx.commands import cmd_new, cmd_list_configs, cmd_list_runs
from stryx.utils import write_yaml

class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = "default"
    value: int = 1

@pytest.fixture
def base_ns(tmp_path):
    """Create a base namespace with context fields."""
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    return argparse.Namespace(
        stryx_schema=Config,
        configs_dir=configs_dir,
        runs_dir=runs_dir,
        stryx_func=lambda x: None
    )

def test_list_configs(base_ns, capsys):
    """Test listing recipes."""
    # Create some recipes
    ns1 = argparse.Namespace(**vars(base_ns))
    ns1.recipe = "a"
    ns1.overrides = ["value=10"]
    ns1.message = None
    ns1.force = False
    cmd_new(ns1)
    
    ns2 = argparse.Namespace(**vars(base_ns))
    ns2.recipe = "b"
    ns2.overrides = ["value=20"]
    ns2.message = None
    ns2.force = False
    cmd_new(ns2)
    
    # List them
    cmd_list_configs(base_ns)
    
    captured = capsys.readouterr()
    assert "a" in captured.out
    assert "b" in captured.out
    assert "10" in captured.out  # Should show interesting column 'value'
    assert "20" in captured.out

def test_list_runs(base_ns, capsys):
    """Test listing runs."""
    # Create fake runs
    run1 = base_ns.runs_dir / "run_1"
    run1.mkdir()
    write_yaml(run1 / "manifest.yaml", {
        "run_id": "run_1", 
        "status": "COMPLETED", 
        "created_at": "2023-01-01T12:00:00",
        "config": {"value": 100}
    })
    
    run2 = base_ns.runs_dir / "run_2"
    run2.mkdir()
    write_yaml(run2 / "manifest.yaml", {
        "run_id": "run_2", 
        "status": "FAILED", 
        "created_at": "2023-01-02T12:00:00",
        "config": {"value": 200}
    })
    
    cmd_list_runs(base_ns)
    
    captured = capsys.readouterr()
    assert "run_1" in captured.out
    assert "COMPLETED" in captured.out
    assert "run_2" in captured.out
    assert "FAILED" in captured.out
    assert "100" in captured.out # interesting column

def test_list_configs_resilience(base_ns, capsys):
    """Test that listing configs skips bad files gracefully."""
    # Good file
    ns = argparse.Namespace(**vars(base_ns))
    ns.recipe = "good"
    ns.overrides = []
    ns.message = None
    ns.force = False
    cmd_new(ns)
    
    # Bad file
    bad_path = base_ns.configs_dir / "bad.yaml"
    bad_path.write_text(":: invalid yaml ::")
    
    cmd_list_configs(base_ns)
    
    captured = capsys.readouterr()
    assert "good" in captured.out
    # Should run without error and not show 'bad' (or show it with limited info, implementation skips)
    # The implementation catches Exception and 'continue's loop.
    assert "bad" not in captured.out