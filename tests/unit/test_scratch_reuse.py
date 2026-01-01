from __future__ import annotations
import pytest
from unittest.mock import MagicMock
import typer
from pydantic import BaseModel, ConfigDict
from stryx.utils import Ctx, read_yaml
from stryx.commands import run_exec, recipe_new

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

def test_scratch_reuse_defaults(mock_ctx, capsys):
    """Test that running with defaults reuses existing scratch."""
    # First run: create scratch
    run_exec(mock_ctx, target=None)
    captured = capsys.readouterr()
    assert "Running scratch" in captured.out
    
    scratches = list((mock_ctx.obj.configs_dir / "scratches").glob("*.yaml"))
    assert len(scratches) == 1
    scratch_name = scratches[0].name
    
    # Second run: reuse scratch
    run_exec(mock_ctx, target=None)
    captured = capsys.readouterr()
    assert f"Reusing scratch: scratches/{scratch_name}" in captured.out
    
    # Should still have only 1 scratch
    scratches_after = list((mock_ctx.obj.configs_dir / "scratches").glob("*.yaml"))
    assert len(scratches_after) == 1
    assert scratches_after[0].name == scratch_name

def test_scratch_reuse_overrides(mock_ctx, capsys):
    """Test that running with same overrides reuses scratch."""
    # First run: value=42
    run_exec(mock_ctx, target="value=42")
    captured = capsys.readouterr()
    assert "Running scratch" in captured.out
    
    scratches = list((mock_ctx.obj.configs_dir / "scratches").glob("*.yaml"))
    assert len(scratches) == 1
    first_scratch = scratches[0]
    
    # Second run: value=42 (reuse)
    run_exec(mock_ctx, target="value=42")
    captured = capsys.readouterr()
    assert f"Reusing scratch: scratches/{first_scratch.name}" in captured.out
    assert len(list((mock_ctx.obj.configs_dir / "scratches").glob("*.yaml"))) == 1

    # Third run: value=99 (new scratch)
    run_exec(mock_ctx, target="value=99")
    captured = capsys.readouterr()
    assert "Running scratch" in captured.out
    assert len(list((mock_ctx.obj.configs_dir / "scratches").glob("*.yaml"))) == 2

def test_canonical_reuse_priority(mock_ctx, capsys):
    """Test that canonical recipes are prioritized over scratches."""
    # 1. Create a canonical recipe
    recipe_new(mock_ctx, name="canonical", overrides=["value=50"])
    
    # 2. Run with matching overrides
    run_exec(mock_ctx, target="value=50")
    captured = capsys.readouterr()
    
    # 3. Should reuse the canonical recipe
    assert "Reusing recipe: canonical.yaml" in captured.out
    
    # 4. Should NOT create a scratch
    scratches = list((mock_ctx.obj.configs_dir / "scratches").glob("*.yaml"))
    assert len(scratches) == 0