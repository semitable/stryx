from __future__ import annotations
import pytest
from unittest.mock import MagicMock
import typer
from pydantic import BaseModel, ConfigDict
from stryx.commands import recipe_init, recipe_clone
from stryx.utils import Ctx, read_yaml, write_yaml

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

def test_clone_basic(mock_ctx):
    """Test basic cloning."""
    recipe_init(mock_ctx, name="source", overrides=["value=10"])
    
    out_path = recipe_clone(mock_ctx, source="source", name="dest")
    
    assert out_path.name == "dest.yaml"
    data = read_yaml(out_path)
    assert data["value"] == 10
    assert data["__stryx__"]["from"] == "source"

def test_clone_with_overrides(mock_ctx):
    """Test cloning and overriding."""
    recipe_init(mock_ctx, name="source", overrides=["value=10"])
    
    out_path = recipe_clone(mock_ctx, source="source", name="dest", overrides=["value=20"])
    
    data = read_yaml(out_path)
    assert data["value"] == 20

def test_clone_metadata_updates(mock_ctx):
    """Ensure metadata is updated."""
    recipe_init(mock_ctx, name="old", message="Old msg")
    old_ts = read_yaml(mock_ctx.obj.configs_dir / "old.yaml")["__stryx__"]["created_at"]
    
    recipe_clone(mock_ctx, source="old", name="new", message="New msg")
    new_data = read_yaml(mock_ctx.obj.configs_dir / "new.yaml")
    
    assert new_data["__stryx__"]["created_at"] != old_ts
    assert new_data["__stryx__"]["description"] == "New msg"

def test_clone_invalid_override(mock_ctx):
    """Test that invalid overrides cause failure."""
    recipe_init(mock_ctx, name="base")
    
    with pytest.raises(SystemExit):
        recipe_clone(mock_ctx, source="base", name="dest", overrides=["typo=999"])

def test_clone_missing_source(mock_ctx):
    """Test error when source doesn't exist."""
    with pytest.raises(typer.Exit):
        recipe_clone(mock_ctx, source="ghost", name="dest")