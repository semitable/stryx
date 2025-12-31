from __future__ import annotations
import argparse
import pytest
from unittest.mock import MagicMock
from pydantic import BaseModel, ConfigDict
from stryx.commands import cmd_new, cmd_run, cmd_try
from stryx.utils import read_yaml, write_yaml

class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = "default"
    value: int = 1

@pytest.fixture
def mock_func():
    return MagicMock(return_value="success_result")

@pytest.fixture
def base_ns(tmp_path, mock_func):
    """Create a base namespace with context fields."""
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    return argparse.Namespace(
        stryx_schema=Config,
        configs_dir=configs_dir,
        runs_dir=runs_dir,
        stryx_func=mock_func
    )

def test_run_basic(base_ns):
    """Test running an existing recipe exactly (no overrides)."""
    # 1. Setup: create a recipe
    # We clone the base_ns or just set attributes. Since we run sequentially in test, setting is fine.
    # But for clarity let's just set what we need.
    ns_new = argparse.Namespace(**vars(base_ns))
    ns_new.recipe = "base"
    ns_new.overrides = ["value=10"]
    ns_new.message = None
    ns_new.force = False
    cmd_new(ns_new)
    
    # 2. Execute run
    ns_run = argparse.Namespace(**vars(base_ns))
    ns_run.target = "base"
    ns_run.run_id = None
    
    result = cmd_run(ns_run)
    
    assert result == "success_result"
    base_ns.stryx_func.assert_called_once()
    
    # Verify the config passed to func
    passed_cfg = base_ns.stryx_func.call_args[0][0]
    assert passed_cfg.value == 10
    
    # Verify artifacts
    run_dirs = list(base_ns.runs_dir.glob("run_*"))
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    assert (run_dir / "manifest.yaml").exists()
    assert (run_dir / "stdout.log").exists()

def test_run_with_id(base_ns):
    """Test 'run' with explicit --run-id."""
    ns_new = argparse.Namespace(**vars(base_ns))
    ns_new.recipe = "base"
    ns_new.overrides = []
    ns_new.message = None
    ns_new.force = False
    cmd_new(ns_new)
    
    ns_run = argparse.Namespace(**vars(base_ns))
    ns_run.target = "base"
    ns_run.run_id = "my-custom-id"
    cmd_run(ns_run)
    
    assert (base_ns.runs_dir / "my-custom-id" / "manifest.yaml").exists()

def test_try_defaults(base_ns):
    """Test 'try' starting from schema defaults."""
    ns = argparse.Namespace(**vars(base_ns))
    ns.target = None
    ns.overrides = ["value=42"]
    ns.message = None
    ns.run_id = None
    
    result = cmd_try(ns)
    
    assert result == "success_result"
    passed_cfg = base_ns.stryx_func.call_args[0][0]
    assert passed_cfg.value == 42
    
    # Verify scratch file created
    scratches = list((base_ns.configs_dir / "scratches").glob("*.yaml"))
    assert len(scratches) == 1
    data = read_yaml(scratches[0])
    assert data["value"] == 42
    assert data["__stryx__"]["type"] == "scratch"

def test_try_from_recipe(base_ns):
    """Test 'try' based on an existing recipe."""
    ns_new = argparse.Namespace(**vars(base_ns))
    ns_new.recipe = "base"
    ns_new.overrides = ["value=10"]
    ns_new.message = None
    ns_new.force = False
    cmd_new(ns_new)
    
    # try base value=20
    ns_try = argparse.Namespace(**vars(base_ns))
    ns_try.target = "base"
    ns_try.overrides = ["value=20"]
    ns_try.message = "Experiment 1"
    ns_try.run_id = None
    
    result = cmd_try(ns_try)
    
    assert result == "success_result"
    passed_cfg = base_ns.stryx_func.call_args[0][0]
    assert passed_cfg.value == 20
    
    # Verify scratch lineage
    scratches = list((base_ns.configs_dir / "scratches").glob("*.yaml"))
    data = read_yaml(scratches[0])
    assert data["__stryx__"]["from"] == "base"
    assert data["__stryx__"]["description"] == "Experiment 1"

def test_try_implicit_overrides(base_ns):
    """Test 'try' when the first argument is an override (no recipe name)."""
    # try value=50
    ns = argparse.Namespace(**vars(base_ns))
    ns.target = "value=50"
    ns.overrides = []
    ns.message = None
    ns.run_id = None
    
    result = cmd_try(ns)
    
    assert result == "success_result"
    passed_cfg = base_ns.stryx_func.call_args[0][0]
    assert passed_cfg.value == 50

def test_run_error_missing_recipe(base_ns):
    """Test that 'run' fails gracefully if recipe not found."""
    ns = argparse.Namespace(**vars(base_ns))
    ns.target = "nonexistent"
    ns.run_id = None
    
    with pytest.raises(SystemExit) as exc:
        cmd_run(ns)
    assert "Recipe not found" in str(exc.value)

def test_run_error_invalid_recipe(base_ns):
    """Test that 'run' fails if the recipe content is invalid for the schema."""
    # Manually create an invalid recipe
    recipe_path = base_ns.configs_dir / "invalid.yaml"
    write_yaml(recipe_path, {"value": "not-an-int", "__stryx__": {}})
    
    ns = argparse.Namespace(**vars(base_ns))
    ns.target = "invalid"
    ns.run_id = None
    
    with pytest.raises(SystemExit) as exc:
        cmd_run(ns)
    assert "Config validation failed" in str(exc.value)

def test_run_exception_propagation(base_ns):
    """Test that exceptions from the user function propagate out of cmd_run."""
    ns_new = argparse.Namespace(**vars(base_ns))
    ns_new.recipe = "base"
    ns_new.overrides = []
    ns_new.message = None
    ns_new.force = False
    cmd_new(ns_new)
    
    # Make func fail
    base_ns.stryx_func.side_effect = ValueError("User code failed")
    
    ns_run = argparse.Namespace(**vars(base_ns))
    ns_run.target = "base"
    ns_run.run_id = None
    
    with pytest.raises(ValueError, match="User code failed"):
        cmd_run(ns_run)
    
    # Verify manifest status is FAILED (RunContext handles this on exit)
    run_dir = list(base_ns.runs_dir.glob("run_*"))[0]
    data = read_yaml(run_dir / "manifest.yaml")
    assert data["status"] == "FAILED"
    assert "User code failed" in data["error"]