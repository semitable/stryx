from __future__ import annotations
import pytest
from unittest.mock import MagicMock
import typer
from pydantic import BaseModel, ConfigDict
from stryx.utils import Ctx, read_yaml, write_yaml
from stryx.commands import recipe_new, run_exec

class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = "default"
    value: int = 1

@pytest.fixture
def mock_func():
    return MagicMock(return_value="success_result")

@pytest.fixture
def mock_ctx(tmp_path, mock_func):
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
    recipe_new(mock_ctx, name="base", overrides=["value=10"])
    
    result = run_exec(mock_ctx, target="base")
    
    assert result == "success_result"
    mock_ctx.obj.func.assert_called_once()
    assert mock_ctx.obj.func.call_args[0][0].value == 10
    
    # Verify artifacts
    assert len(list(mock_ctx.obj.runs_dir.glob("run_*"))) == 1

def test_run_with_id(mock_ctx):
    """Test 'run' with explicit --run-id."""
    recipe_new(mock_ctx, name="base")
    
    run_exec(mock_ctx, target="base", run_id="my-custom-id")
    
    assert (mock_ctx.obj.runs_dir / "my-custom-id" / "stryx.manifest.yaml").exists()

def test_try_defaults(mock_ctx):
    """Test 'try' (implicit run) starting from schema defaults."""
    # run value=42
    result = run_exec(mock_ctx, target="value=42")
    
    assert result == "success_result"
    assert mock_ctx.obj.func.call_args[0][0].value == 42
    
    # Should create scratch
    scratches = list((mock_ctx.obj.configs_dir / "scratches").glob("*.yaml"))
    assert len(scratches) == 1
    assert read_yaml(scratches[0])["value"] == 42

def test_try_from_recipe(mock_ctx):
    """Test 'try' based on an existing recipe."""
    recipe_new(mock_ctx, name="base", overrides=["value=10"])
    
    # run base value=20
    result = run_exec(mock_ctx, target="base", overrides=["value=20"])
    
    assert result == "success_result"
    assert mock_ctx.obj.func.call_args[0][0].value == 20
    
    # Verify scratch lineage
    data = read_yaml(list((mock_ctx.obj.configs_dir / "scratches").glob("*.yaml"))[0])
    assert data["__stryx__"]["from"] == "base"

def test_run_error_missing_recipe(mock_ctx):
    """Test that run fails gracefully if recipe not found."""
    with pytest.raises(typer.Exit):
        run_exec(mock_ctx, target="nonexistent")

def test_run_error_invalid_recipe(mock_ctx):
    """Test that run fails if recipe invalid."""
    recipe_path = mock_ctx.obj.configs_dir / "invalid.yaml"
    write_yaml(recipe_path, {"value": "not-an-int", "__stryx__": {}})
    
    with pytest.raises(SystemExit):
        run_exec(mock_ctx, target="invalid")

def test_dry_run(mock_ctx, capsys):
    """Test dry run."""
    run_exec(mock_ctx, target="value=10", dry=True)
    captured = capsys.readouterr()
    assert "Dry run: Config resolved successfully." in captured.out
    mock_ctx.obj.func.assert_not_called()