from __future__ import annotations
import pytest
from unittest.mock import MagicMock
import typer
from pydantic import BaseModel, ConfigDict
from stryx.utils import Ctx, read_yaml, write_yaml
from stryx.commands import cmd_new, cmd_run, cmd_try

class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = "default"
    value: int = 1

@pytest.fixture
def mock_func():
    return MagicMock(return_value="success_result")

@pytest.fixture
def mock_ctx(tmp_path, mock_func):
    """Create a mock Typer context with Stryx Ctx."""
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    c = Ctx(
        schema=Config,
        configs_dir=configs_dir,
        runs_dir=runs_dir,
        func=mock_func
    )
    ctx = MagicMock(spec=typer.Context)
    ctx.obj = c
    return ctx

def test_run_basic(mock_ctx):
    """Test running an existing recipe exactly."""
    cmd_new(mock_ctx, recipe="base", overrides=["value=10"])
    
    result = cmd_run(mock_ctx, target="base")
    
    assert result == "success_result"
    mock_ctx.obj.func.assert_called_once()
    
    # Verify the config passed to func
    passed_cfg = mock_ctx.obj.func.call_args[0][0]
    assert passed_cfg.value == 10
    
    # Verify artifacts
    run_dirs = list(mock_ctx.obj.runs_dir.glob("run_*"))
    assert len(run_dirs) == 1
    assert (run_dirs[0] / "manifest.yaml").exists()

def test_run_with_id(mock_ctx):
    """Test 'run' with explicit --run-id."""
    cmd_new(mock_ctx, recipe="base")
    
    cmd_run(mock_ctx, target="base", run_id="my-custom-id")
    
    assert (mock_ctx.obj.runs_dir / "my-custom-id" / "manifest.yaml").exists()

def test_try_defaults(mock_ctx):
    """Test 'try' starting from schema defaults."""
    result = cmd_try(mock_ctx, overrides=["value=42"])
    
    assert result == "success_result"
    passed_cfg = mock_ctx.obj.func.call_args[0][0]
    assert passed_cfg.value == 42
    
    # Verify scratch file
    scratches = list((mock_ctx.obj.configs_dir / "scratches").glob("*.yaml"))
    assert len(scratches) == 1
    assert read_yaml(scratches[0])["value"] == 42

def test_try_from_recipe(mock_ctx):
    """Test 'try' based on an existing recipe."""
    cmd_new(mock_ctx, recipe="base", overrides=["value=10"])
    
    result = cmd_try(mock_ctx, target="base", overrides=["value=20"], message="Experiment 1")
    
    assert result == "success_result"
    passed_cfg = mock_ctx.obj.func.call_args[0][0]
    assert passed_cfg.value == 20
    
    data = read_yaml(list((mock_ctx.obj.configs_dir / "scratches").glob("*.yaml"))[0])
    assert data["__stryx__"]["from"] == "base"

def test_try_implicit_overrides(mock_ctx):
    """Test 'try' when the first argument is an override."""
    # try value=50
    result = cmd_try(mock_ctx, target="value=50")
    
    assert result == "success_result"
    passed_cfg = mock_ctx.obj.func.call_args[0][0]
    assert passed_cfg.value == 50

def test_run_error_missing_recipe(mock_ctx):
    """Test that 'run' fails gracefully if recipe not found."""
    with pytest.raises(typer.Exit):
        cmd_run(mock_ctx, target="nonexistent")

def test_run_error_invalid_recipe(mock_ctx):
    """Test that 'run' fails if recipe invalid."""
    recipe_path = mock_ctx.obj.configs_dir / "invalid.yaml"
    write_yaml(recipe_path, {"value": "not-an-int", "__stryx__": {}})
    
    with pytest.raises(SystemExit):
        cmd_run(mock_ctx, target="invalid")

def test_run_exception_propagation(mock_ctx):
    """Test that exceptions from the user function propagate."""
    cmd_new(mock_ctx, recipe="base")
    mock_ctx.obj.func.side_effect = ValueError("User code failed")
    
    with pytest.raises(ValueError, match="User code failed"):
        cmd_run(mock_ctx, target="base")
    
    # Verify failed status
    run_dir = list(mock_ctx.obj.runs_dir.glob("run_*"))[0]
    data = read_yaml(run_dir / "manifest.yaml")
    assert data["status"] == "FAILED"
