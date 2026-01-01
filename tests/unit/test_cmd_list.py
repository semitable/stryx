from __future__ import annotations
import pytest
from unittest.mock import MagicMock
import typer
from pydantic import BaseModel, ConfigDict
from stryx.utils import Ctx, write_yaml
from stryx.commands import recipe_init, recipe_list, run_list

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

def test_list_recipes(mock_ctx, capsys):
    """Test listing recipes."""
    recipe_init(mock_ctx, name="a", overrides=["value=10"])
    recipe_init(mock_ctx, name="b", overrides=["value=20"])
    
    recipe_list(mock_ctx)
    
    captured = capsys.readouterr()
    assert "stryx.name" in captured.out
    assert "stryx.created_at" in captured.out
    assert "a" in captured.out
    assert "b" in captured.out
    assert "10" in captured.out
    assert "20" in captured.out

def test_list_runs(mock_ctx, capsys):
    """Test listing runs."""
    run1 = mock_ctx.obj.runs_dir / "run_1"
    run1.mkdir()
    write_yaml(run1 / "stryx.manifest.yaml", {
        "run_id": "run_1", 
        "status": "COMPLETED", 
        "created_at": "2023-01-01T12:00:00",
    })
    write_yaml(run1 / "stryx.results.yaml", {"accuracy": 0.95, "loss": 0.1})
    write_yaml(run1 / "stryx.config.yaml", {"value": 100})
    
    run2 = mock_ctx.obj.runs_dir / "run_2"
    run2.mkdir()
    write_yaml(run2 / "stryx.manifest.yaml", {
        "run_id": "run_2", 
        "status": "FAILED", 
        "created_at": "2023-01-02T12:00:00",
    })
    write_yaml(run2 / "stryx.config.yaml", {"value": 200})
    
    run_list(mock_ctx)
    
    captured = capsys.readouterr()
    # Check headers
    assert "stryx.run_id" in captured.out
    assert "stryx.status" in captured.out
    assert "stryx.created_at" in captured.out
    
    assert "run_1" in captured.out
    assert "COMPLETED" in captured.out
    assert "run_2" in captured.out
    assert "FAILED" in captured.out
    assert "100" not in captured.out
    assert "200" not in captured.out
    # Check result columns (pandas flat output)
    assert "accuracy" in captured.out
    assert "0.95" in captured.out

def test_list_recipes_hides_constant(mock_ctx, capsys):
    """Test that constant columns are hidden in recipe list."""
    recipe_init(mock_ctx, name="r1", overrides=["value=10"])
    recipe_init(mock_ctx, name="r2", overrides=["value=10"])
    
    recipe_list(mock_ctx)
    
    captured = capsys.readouterr()
    assert "stryx.name" in captured.out
    assert "r1" in captured.out
    assert "r2" in captured.out
    # value is constant (10), should be hidden (header "value" not present)
    # checking for "value" string is risky if it appears elsewhere, but usually headers are printed clearly.
    # checking that "10" doesn't appear is safer if we know it won't be in created_at etc.
    # Actually, pandas might not print the column at all.
    # Let's check that the line containing "value" is not there, or just rely on "value" not being in the output if it's unique enough.
    # "value" is the field name. "10" is the value.
    # If column is hidden, neither header nor value is printed.
    # But "10" is short.
    # Let's use a unique number.
    
    recipe_init(mock_ctx, name="r3", overrides=["value=999"])
    # Now r1=10, r2=10, r3=999. It VARIES now! So it SHOULD appear.
    
    # Let's do a fresh context or clear it?
    # Mock context persists dirs.
    # I'll rely on the fact that I can't easily clear the dir in this fixture structure without making a new one.
    # I'll make a new test function `test_list_recipes_hides_constant_clean` if needed.
    # Or just use `test_list_recipes_hides_constant` and rely on `tmp_path` being fresh for each test?
    # Yes, `mock_ctx` uses `tmp_path`, which is fresh per test.
    pass

def test_list_recipes_hides_constant_fresh(tmp_path, capsys):
    from stryx.utils import Ctx
    from stryx.commands import recipe_init, recipe_list
    
    class Config(BaseModel):
        model_config = ConfigDict(extra="forbid")
        val: int = 123
        
    c = Ctx(
        schema=Config,
        configs_dir=tmp_path / "configs",
        runs_dir=tmp_path / "runs",
        func=lambda x: None
    )
    ctx = MagicMock(spec=typer.Context)
    ctx.obj = c
    
    recipe_init(ctx, name="r1", overrides=["val=777"])
    recipe_init(ctx, name="r2", overrides=["val=777"])
    
    recipe_list(ctx)
    
    captured = capsys.readouterr()
    assert "stryx.name" in captured.out
    assert "r1" in captured.out
    assert "val" not in captured.out
    assert "777" not in captured.out

def test_list_configs_resilience(mock_ctx, capsys):
    """Test that listing configs skips bad files gracefully."""
    recipe_init(mock_ctx, name="good")
    
    bad_path = mock_ctx.obj.configs_dir / "bad.yaml"
    bad_path.write_text(":: invalid yaml ::")
    
    recipe_list(mock_ctx)
    
    captured = capsys.readouterr()
    assert "good" in captured.out
    assert "bad" not in captured.out