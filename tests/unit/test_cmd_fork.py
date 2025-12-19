from __future__ import annotations
import argparse
import pytest
from datetime import datetime
from pathlib import Path
from pydantic import BaseModel, ConfigDict, ValidationError
from stryx.context import Ctx
from stryx.commands import cmd_new, cmd_fork
from stryx.utils import read_yaml, write_yaml

class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = "default"
    value: int = 1
    nested: dict[str, int] = {}

@pytest.fixture
def ctx(tmp_path):
    """Create a context with temporary directories."""
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()
    return Ctx(
        schema=Config,
        configs_dir=configs_dir,
        runs_dir=tmp_path / "runs",
    )

def test_fork_basic(ctx):
    """Test basic forking from one recipe to another."""
    # 1. Create source
    ns_new = argparse.Namespace(recipe="source", overrides=["value=10"], message=None, force=False)
    cmd_new(ctx, ns_new)
    
    # 2. Fork it
    ns_fork = argparse.Namespace(source="source", name="dest", overrides=[], message=None, force=False)
    out_path = cmd_fork(ctx, ns_fork)
    
    assert out_path.name == "dest.yaml"
    data = read_yaml(out_path)
    assert data["value"] == 10
    assert data["__stryx__"]["from"] == "source"

def test_fork_with_overrides(ctx):
    """Test forking and overriding at the same time."""
    # 1. Create source
    ns_new = argparse.Namespace(recipe="source", overrides=["value=10"], message=None, force=False)
    cmd_new(ctx, ns_new)
    
    # 2. Fork with override
    ns_fork = argparse.Namespace(source="source", name="dest", overrides=["value=20"], message=None, force=False)
    out_path = cmd_fork(ctx, ns_fork)
    
    data = read_yaml(out_path)
    assert data["value"] == 20
    assert data["__stryx__"]["overrides"] == ["value=20"]

def test_fork_metadata_updates(ctx):
    """Ensure metadata (created_at) is updated and not just copied."""
    # 1. Create source
    cmd_new(ctx, argparse.Namespace(recipe="old", overrides=[], message="Old msg", force=False))
    old_data = read_yaml(ctx.configs_dir / "old.yaml")
    old_ts = old_data["__stryx__"]["created_at"]
    
    # 2. Fork it
    cmd_fork(ctx, argparse.Namespace(source="old", name="new", overrides=[], message="New msg", force=False))
    new_data = read_yaml(ctx.configs_dir / "new.yaml")
    
    # Check updates
    assert new_data["__stryx__"]["created_at"] != old_ts
    assert new_data["__stryx__"]["description"] == "New msg"
    assert new_data["__stryx__"]["from"] == "old"

def test_fork_to_subdirectory(ctx):
    """Test forking into a subdirectory (should create dir)."""
    cmd_new(ctx, argparse.Namespace(recipe="root", overrides=[], message=None, force=False))
    
    ns = argparse.Namespace(source="root", name="experiments/subdir/deep_fork", overrides=[], message=None, force=False)
    out_path = cmd_fork(ctx, ns)
    
    assert out_path.exists()
    assert (ctx.configs_dir / "experiments/subdir").exists()
    assert out_path.parent.name == "subdir"

def test_fork_self_overwrite(ctx):
    """Test forking a recipe onto itself (inplace update)."""
    cmd_new(ctx, argparse.Namespace(recipe="selfie", overrides=["value=1"], message=None, force=False))
    
    # Fail without force
    with pytest.raises(SystemExit):
        cmd_fork(ctx, argparse.Namespace(source="selfie", name="selfie", overrides=["value=2"], message=None, force=False))
        
    # Succeed with force
    cmd_fork(ctx, argparse.Namespace(source="selfie", name="selfie", overrides=["value=2"], message="Updated", force=True))
    
    data = read_yaml(ctx.configs_dir / "selfie.yaml")
    assert data["value"] == 2
    assert data["__stryx__"]["description"] == "Updated"

def test_fork_invalid_override(ctx):
    """Test that invalid overrides during fork cause failure (strict config)."""
    cmd_new(ctx, argparse.Namespace(recipe="base", overrides=[], message=None, force=False))
    
    ns = argparse.Namespace(source="base", name="dest", overrides=["typo=999"], message=None, force=False)
    
    # validate_or_die raises SystemExit on validation error
    with pytest.raises(SystemExit) as exc:
        cmd_fork(ctx, ns)
    assert "Config validation failed" in str(exc.value)

def test_fork_source_resolution(ctx):
    """Test finding source in different ways."""
    # Create a scratch recipe manually
    scratches = ctx.configs_dir / "scratches"
    scratches.mkdir()
    scratch_path = scratches / "temp.yaml"
    write_yaml(scratch_path, {"value": 50, "name": "scratch", "__stryx__": {}})
    
    # Fork from scratch by name
    ns = argparse.Namespace(source="temp", name="from_scratch", overrides=[], message=None, force=False)
    out_path = cmd_fork(ctx, ns)
    
    assert read_yaml(out_path)["value"] == 50
    
    # Fork from full path
    ns_path = argparse.Namespace(source=str(scratch_path), name="from_path", overrides=[], message=None, force=False)
    out_path_2 = cmd_fork(ctx, ns_path)
    assert read_yaml(out_path_2)["value"] == 50

def test_fork_overwrite_protection(ctx):
    """Test that fork respects --force."""
    # Create source and dest
    cmd_new(ctx, argparse.Namespace(recipe="s", overrides=["value=1"], message=None, force=False))
    cmd_new(ctx, argparse.Namespace(recipe="d", overrides=["value=2"], message=None, force=False))
    
    # Fork s to d without force
    ns = argparse.Namespace(source="s", name="d", overrides=[], message=None, force=False)
    with pytest.raises(SystemExit):
        cmd_fork(ctx, ns)
        
    # Fork s to d with force
    ns_force = argparse.Namespace(source="s", name="d", overrides=[], message=None, force=True)
    cmd_fork(ctx, ns_force)
    assert read_yaml(ctx.configs_dir / "d.yaml")["value"] == 1

def test_fork_missing_source(ctx):
    """Test error when source doesn't exist."""
    ns = argparse.Namespace(source="ghost", name="dest", overrides=[], message=None, force=False)
    with pytest.raises(SystemExit) as exc:
        cmd_fork(ctx, ns)
    assert "Source recipe not found" in str(exc.value)