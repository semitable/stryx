from __future__ import annotations
import pytest
from unittest.mock import MagicMock
import typer
from pydantic import BaseModel, ConfigDict
from stryx.commands import recipe_new, recipe_fork
from stryx.utils import Ctx, read_yaml

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
        func=lambda x: None,
    )
    ctx = MagicMock(spec=typer.Context)
    ctx.obj = c
    return ctx

def test_fork_basic(mock_ctx):
    """Test basic forking from one recipe to another."""
    recipe_new(mock_ctx, name="source", overrides=["value=10"])
    
    out_path = recipe_fork(mock_ctx, source="source", name="dest")
    
    assert out_path.name == "dest.yaml"
    data = read_yaml(out_path)
    assert data["value"] == 10
    assert data["__stryx__"]["from"] == "source"

def test_fork_with_overrides(mock_ctx):
    """Test forking and overriding at the same time."""
    recipe_new(mock_ctx, name="source", overrides=["value=10"])
    
    out_path = recipe_fork(mock_ctx, source="source", name="dest", overrides=["value=20"])
    
    data = read_yaml(out_path)
    assert data["value"] == 20

def test_fork_metadata_updates(mock_ctx):
    """Ensure metadata is updated."""
    recipe_new(mock_ctx, name="old", message="Old msg")
    old_ts = read_yaml(mock_ctx.obj.configs_dir / "old.yaml")["__stryx__"]["created_at"]
    
    recipe_fork(mock_ctx, source="old", name="new", message="New msg")
    new_data = read_yaml(mock_ctx.obj.configs_dir / "new.yaml")
    
    assert new_data["__stryx__"]["created_at"] != old_ts
    assert new_data["__stryx__"]["description"] == "New msg"

def test_fork_invalid_override(mock_ctx):
    """Test that invalid overrides during fork cause failure."""
    recipe_new(mock_ctx, name="base")
    
    with pytest.raises(SystemExit):
        recipe_fork(mock_ctx, source="base", name="dest", overrides=["typo=999"])

def test_fork_name_conflict(mock_ctx):
    """Test that providing an override-like name causes error."""
    recipe_new(mock_ctx, name="base")
    
    # "optim.lr=4" looks like override, should fail validation for 'name' arg
    with pytest.raises(typer.Exit):
        recipe_fork(mock_ctx, source="base", name="optim.lr=4")

def test_fork_source_conflict(mock_ctx):
    """Test that providing an override-like source causes error."""
    # "optim.lr=4" looks like override, should fail validation for 'source' arg
    with pytest.raises(typer.Exit):
        recipe_fork(mock_ctx, source="optim.lr=4", name="dest")

def test_fork_missing_source(mock_ctx):
    """Test error when source doesn't exist."""
    with pytest.raises(typer.Exit):
        recipe_fork(mock_ctx, source="ghost", name="dest")