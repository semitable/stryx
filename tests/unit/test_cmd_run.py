from __future__ import annotations
import argparse
import pytest
from unittest.mock import MagicMock
from pydantic import BaseModel, ConfigDict
from stryx.context import Ctx
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
def ctx(tmp_path, mock_func):
    """Create a context with temporary directories."""
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    return Ctx(
        schema=Config,
        configs_dir=configs_dir,
        runs_dir=runs_dir,
        func=mock_func
    )

def test_run_basic(ctx):
    """Test running an existing recipe exactly (no overrides)."""
    # 1. Setup: create a recipe
    cmd_new(ctx, argparse.Namespace(recipe="base", overrides=["value=10"], message=None, force=False))
    
    # 2. Execute run
    ns = argparse.Namespace(target="base", run_id=None)
    result = cmd_run(ctx, ns)
    
    assert result == "success_result"
    ctx.func.assert_called_once()
    
    # Verify the config passed to func
    passed_cfg = ctx.func.call_args[0][0]
    assert passed_cfg.value == 10
    
    # Verify artifacts
    run_dirs = list(ctx.runs_dir.glob("run_*"))
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    assert (run_dir / "manifest.yaml").exists()
    assert (run_dir / "stdout.log").exists()

def test_run_with_id(ctx):
    """Test 'run' with explicit --run-id."""
    cmd_new(ctx, argparse.Namespace(recipe="base", overrides=[], message=None, force=False))
    
    ns = argparse.Namespace(target="base", run_id="my-custom-id")
    cmd_run(ctx, ns)
    
    assert (ctx.runs_dir / "my-custom-id" / "manifest.yaml").exists()

def test_try_defaults(ctx):
    """Test 'try' starting from schema defaults."""
    ns = argparse.Namespace(target=None, overrides=["value=42"], message=None, run_id=None)
    result = cmd_try(ctx, ns)
    
    assert result == "success_result"
    passed_cfg = ctx.func.call_args[0][0]
    assert passed_cfg.value == 42
    
    # Verify scratch file created
    scratches = list((ctx.configs_dir / "scratches").glob("*.yaml"))
    assert len(scratches) == 1
    data = read_yaml(scratches[0])
    assert data["value"] == 42
    assert data["__stryx__"]["type"] == "scratch"

def test_try_from_recipe(ctx):
    """Test 'try' based on an existing recipe."""
    cmd_new(ctx, argparse.Namespace(recipe="base", overrides=["value=10"], message=None, force=False))
    
    # try base value=20
    ns = argparse.Namespace(target="base", overrides=["value=20"], message="Experiment 1", run_id=None)
    result = cmd_try(ctx, ns)
    
    assert result == "success_result"
    passed_cfg = ctx.func.call_args[0][0]
    assert passed_cfg.value == 20
    
    # Verify scratch lineage
    scratches = list((ctx.configs_dir / "scratches").glob("*.yaml"))
    data = read_yaml(scratches[0])
    assert data["__stryx__"]["from"] == "base"
    assert data["__stryx__"]["description"] == "Experiment 1"

def test_try_implicit_overrides(ctx):
    """Test 'try' when the first argument is an override (no recipe name)."""
    # try value=50
    ns = argparse.Namespace(target="value=50", overrides=[], message=None, run_id=None)
    result = cmd_try(ctx, ns)
    
    assert result == "success_result"
    passed_cfg = ctx.func.call_args[0][0]
    assert passed_cfg.value == 50

def test_run_error_missing_recipe(ctx):
    """Test that 'run' fails gracefully if recipe not found."""
    ns = argparse.Namespace(target="nonexistent", run_id=None)
    with pytest.raises(SystemExit) as exc:
        cmd_run(ctx, ns)
    assert "Recipe not found" in str(exc.value)

def test_run_error_invalid_recipe(ctx):
    """Test that 'run' fails if the recipe content is invalid for the schema."""
    # Manually create an invalid recipe
    recipe_path = ctx.configs_dir / "invalid.yaml"
    write_yaml(recipe_path, {"value": "not-an-int", "__stryx__": {}})
    
    ns = argparse.Namespace(target="invalid", run_id=None)
    with pytest.raises(SystemExit) as exc:
        cmd_run(ctx, ns)
    assert "Config validation failed" in str(exc.value)

def test_run_exception_propagation(ctx):
    """Test that exceptions from the user function propagate out of cmd_run."""
    cmd_new(ctx, argparse.Namespace(recipe="base", overrides=[], message=None, force=False))
    
    # Make func fail
    ctx.func.side_effect = ValueError("User code failed")
    
    ns = argparse.Namespace(target="base", run_id=None)
    
    with pytest.raises(ValueError, match="User code failed"):
        cmd_run(ctx, ns)
    
    # Verify manifest status is FAILED (RunContext handles this on exit)
    # We need to find the run directory
    run_dir = list(ctx.runs_dir.glob("run_*"))[0]
    data = read_yaml(run_dir / "manifest.yaml")
    assert data["status"] == "FAILED"
    assert "User code failed" in data["error"]
