from __future__ import annotations
import pytest
from unittest.mock import MagicMock
import typer
from pydantic import BaseModel, ConfigDict
from stryx.commands import cmd_new, cmd_fork
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

def test_fork_basic(mock_ctx):
    """Test basic forking from one recipe to another."""
    cmd_new(mock_ctx, recipe="source", overrides=["value=10"])
    
    out_path = cmd_fork(mock_ctx, source="source", name="dest")
    
    assert out_path.name == "dest.yaml"
    data = read_yaml(out_path)
    assert data["value"] == 10
    assert data["__stryx__"]["from"] == "source"

def test_fork_with_overrides(mock_ctx):
    """Test forking and overriding at the same time."""
    cmd_new(mock_ctx, recipe="source", overrides=["value=10"])
    
    out_path = cmd_fork(mock_ctx, source="source", name="dest", overrides=["value=20"])
    
    data = read_yaml(out_path)
    assert data["value"] == 20

def test_fork_metadata_updates(mock_ctx):
    """Ensure metadata (created_at) is updated and not just copied."""
    cmd_new(mock_ctx, recipe="old", message="Old msg")
    old_data = read_yaml(mock_ctx.obj.configs_dir / "old.yaml")
    old_ts = old_data["__stryx__"]["created_at"]
    
    cmd_fork(mock_ctx, source="old", name="new", message="New msg")
    new_data = read_yaml(mock_ctx.obj.configs_dir / "new.yaml")
    
    assert new_data["__stryx__"]["created_at"] != old_ts
    assert new_data["__stryx__"]["description"] == "New msg"

def test_fork_to_subdirectory(mock_ctx):
    """Test forking into a subdirectory."""
    cmd_new(mock_ctx, recipe="root")
    
    out_path = cmd_fork(mock_ctx, source="root", name="experiments/subdir/deep_fork")
    
    assert out_path.exists()
    assert (mock_ctx.obj.configs_dir / "experiments/subdir").exists()

def test_fork_self_overwrite(mock_ctx):
    """Test forking a recipe onto itself (inplace update)."""
    cmd_new(mock_ctx, recipe="selfie", overrides=["value=1"])
    
    # Fail without force
    with pytest.raises(typer.Exit):
        cmd_fork(mock_ctx, source="selfie", name="selfie", overrides=["value=2"])
        
    # Succeed with force
    cmd_fork(mock_ctx, source="selfie", name="selfie", overrides=["value=2"], message="Updated", force=True)
    
    data = read_yaml(mock_ctx.obj.configs_dir / "selfie.yaml")
    assert data["value"] == 2
    assert data["__stryx__"]["description"] == "Updated"

def test_fork_invalid_override(mock_ctx):
    """Test that invalid overrides during fork cause failure."""
    cmd_new(mock_ctx, recipe="base")
    
    with pytest.raises(SystemExit):
        cmd_fork(mock_ctx, source="base", name="dest", overrides=["typo=999"])

def test_fork_source_resolution(mock_ctx):
    """Test finding source in different ways."""
    scratches = mock_ctx.obj.configs_dir / "scratches"
    scratches.mkdir()
    scratch_path = scratches / "temp.yaml"
    write_yaml(scratch_path, {"value": 50, "name": "scratch", "__stryx__": {}})
    
    # Fork from scratch by name
    out_path = cmd_fork(mock_ctx, source="temp", name="from_scratch")
    assert read_yaml(out_path)["value"] == 50
    
    # Fork from full path
    out_path_2 = cmd_fork(mock_ctx, source=str(scratch_path), name="from_path")
    assert read_yaml(out_path_2)["value"] == 50

def test_fork_overwrite_protection(mock_ctx):
    """Test that fork respects --force."""
    cmd_new(mock_ctx, recipe="s", overrides=["value=1"])
    cmd_new(mock_ctx, recipe="d", overrides=["value=2"])
    
    with pytest.raises(typer.Exit):
        cmd_fork(mock_ctx, source="s", name="d")
        
    cmd_fork(mock_ctx, source="s", name="d", force=True)
    assert read_yaml(mock_ctx.obj.configs_dir / "d.yaml")["value"] == 1

def test_fork_missing_source(mock_ctx):
    """Test error when source doesn't exist."""
    with pytest.raises(typer.Exit):
        cmd_fork(mock_ctx, source="ghost", name="dest")