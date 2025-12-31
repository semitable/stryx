from __future__ import annotations
import argparse
import pytest
from pydantic import BaseModel, ConfigDict
from stryx.commands import cmd_new, cmd_fork
from stryx.utils import read_yaml, write_yaml

class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = "default"
    value: int = 1
    nested: dict[str, int] = {}

@pytest.fixture
def base_ns(tmp_path):
    """Create a base namespace with context fields."""
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    return argparse.Namespace(
        stryx_schema=Config,
        configs_dir=configs_dir,
        runs_dir=runs_dir,
        stryx_func=lambda x: None,
    )

def test_fork_basic(base_ns):
    """Test basic forking from one recipe to another."""
    # 1. Create source
    ns_new = argparse.Namespace(**vars(base_ns))
    ns_new.recipe = "source"
    ns_new.overrides = ["value=10"]
    ns_new.message = None
    ns_new.force = False
    cmd_new(ns_new)
    
    # 2. Fork it
    ns_fork = argparse.Namespace(**vars(base_ns))
    ns_fork.source = "source"
    ns_fork.name = "dest"
    ns_fork.overrides = []
    ns_fork.message = None
    ns_fork.force = False
    out_path = cmd_fork(ns_fork)
    
    assert out_path.name == "dest.yaml"
    data = read_yaml(out_path)
    assert data["value"] == 10
    assert data["__stryx__"]["from"] == "source"

def test_fork_with_overrides(base_ns):
    """Test forking and overriding at the same time."""
    # 1. Create source
    ns_new = argparse.Namespace(**vars(base_ns))
    ns_new.recipe = "source"
    ns_new.overrides = ["value=10"]
    ns_new.message = None
    ns_new.force = False
    cmd_new(ns_new)
    
    # 2. Fork with override
    ns_fork = argparse.Namespace(**vars(base_ns))
    ns_fork.source = "source"
    ns_fork.name = "dest"
    ns_fork.overrides = ["value=20"]
    ns_fork.message = None
    ns_fork.force = False
    out_path = cmd_fork(ns_fork)
    
    data = read_yaml(out_path)
    assert data["value"] == 20
    assert data["__stryx__"]["overrides"] == ["value=20"]

def test_fork_metadata_updates(base_ns):
    """Ensure metadata (created_at) is updated and not just copied."""
    # 1. Create source
    ns_new = argparse.Namespace(**vars(base_ns))
    ns_new.recipe = "old"
    ns_new.overrides = []
    ns_new.message = "Old msg"
    ns_new.force = False
    cmd_new(ns_new)
    old_data = read_yaml(base_ns.configs_dir / "old.yaml")
    old_ts = old_data["__stryx__"]["created_at"]
    
    # 2. Fork it
    ns_fork = argparse.Namespace(**vars(base_ns))
    ns_fork.source = "old"
    ns_fork.name = "new"
    ns_fork.overrides = []
    ns_fork.message = "New msg"
    ns_fork.force = False
    cmd_fork(ns_fork)
    new_data = read_yaml(base_ns.configs_dir / "new.yaml")
    
    # Check updates
    assert new_data["__stryx__"]["created_at"] != old_ts
    assert new_data["__stryx__"]["description"] == "New msg"
    assert new_data["__stryx__"]["from"] == "old"

def test_fork_to_subdirectory(base_ns):
    """Test forking into a subdirectory (should create dir)."""
    ns_new = argparse.Namespace(**vars(base_ns))
    ns_new.recipe = "root"
    ns_new.overrides = []
    ns_new.message = None
    ns_new.force = False
    cmd_new(ns_new)
    
    ns_fork = argparse.Namespace(**vars(base_ns))
    ns_fork.source = "root"
    ns_fork.name = "experiments/subdir/deep_fork"
    ns_fork.overrides = []
    ns_fork.message = None
    ns_fork.force = False
    out_path = cmd_fork(ns_fork)
    
    assert out_path.exists()
    assert (base_ns.configs_dir / "experiments/subdir").exists()
    assert out_path.parent.name == "subdir"

def test_fork_self_overwrite(base_ns):
    """Test forking a recipe onto itself (inplace update)."""
    ns_new = argparse.Namespace(**vars(base_ns))
    ns_new.recipe = "selfie"
    ns_new.overrides = ["value=1"]
    ns_new.message = None
    ns_new.force = False
    cmd_new(ns_new)
    
    # Fail without force
    ns_fail = argparse.Namespace(**vars(base_ns))
    ns_fail.source = "selfie"
    ns_fail.name = "selfie"
    ns_fail.overrides = ["value=2"]
    ns_fail.message = None
    ns_fail.force = False
    with pytest.raises(SystemExit):
        cmd_fork(ns_fail)
        
    # Succeed with force
    ns_force = argparse.Namespace(**vars(base_ns))
    ns_force.source = "selfie"
    ns_force.name = "selfie"
    ns_force.overrides = ["value=2"]
    ns_force.message = "Updated"
    ns_force.force = True
    cmd_fork(ns_force)
    
    data = read_yaml(base_ns.configs_dir / "selfie.yaml")
    assert data["value"] == 2
    assert data["__stryx__"]["description"] == "Updated"

def test_fork_invalid_override(base_ns):
    """Test that invalid overrides during fork cause failure (strict config)."""
    ns_new = argparse.Namespace(**vars(base_ns))
    ns_new.recipe = "base"
    ns_new.overrides = []
    ns_new.message = None
    ns_new.force = False
    cmd_new(ns_new)
    
    ns = argparse.Namespace(**vars(base_ns))
    ns.source = "base"
    ns.name = "dest"
    ns.overrides = ["typo=999"]
    ns.message = None
    ns.force = False
    
    # validate_or_die raises SystemExit on validation error
    with pytest.raises(SystemExit) as exc:
        cmd_fork(ns)
    assert "Config validation failed" in str(exc.value)

def test_fork_source_resolution(base_ns):
    """Test finding source in different ways."""
    # Create a scratch recipe manually
    scratches = base_ns.configs_dir / "scratches"
    scratches.mkdir()
    scratch_path = scratches / "temp.yaml"
    write_yaml(scratch_path, {"value": 50, "name": "scratch", "__stryx__": {}})
    
    # Fork from scratch by name
    ns = argparse.Namespace(**vars(base_ns))
    ns.source = "temp"
    ns.name = "from_scratch"
    ns.overrides = []
    ns.message = None
    ns.force = False
    out_path = cmd_fork(ns)
    
    assert read_yaml(out_path)["value"] == 50
    
    # Fork from full path
    ns_path = argparse.Namespace(**vars(base_ns))
    ns_path.source = str(scratch_path)
    ns_path.name = "from_path"
    ns_path.overrides = []
    ns_path.message = None
    ns_path.force = False
    out_path_2 = cmd_fork(ns_path)
    assert read_yaml(out_path_2)["value"] == 50

def test_fork_overwrite_protection(base_ns):
    """Test that fork respects --force."""
    # Create source and dest
    ns_s = argparse.Namespace(**vars(base_ns))
    ns_s.recipe = "s"
    ns_s.overrides = ["value=1"]
    ns_s.message = None
    ns_s.force = False
    cmd_new(ns_s)
    
    ns_d = argparse.Namespace(**vars(base_ns))
    ns_d.recipe = "d"
    ns_d.overrides = ["value=2"]
    ns_d.message = None
    ns_d.force = False
    cmd_new(ns_d)
    
    # Fork s to d without force
    ns = argparse.Namespace(**vars(base_ns))
    ns.source = "s"
    ns.name = "d"
    ns.overrides = []
    ns.message = None
    ns.force = False
    with pytest.raises(SystemExit):
        cmd_fork(ns)
        
    # Fork s to d with force
    ns_force = argparse.Namespace(**vars(base_ns))
    ns_force.source = "s"
    ns_force.name = "d"
    ns_force.overrides = []
    ns_force.message = None
    ns_force.force = True
    cmd_fork(ns_force)
    assert read_yaml(base_ns.configs_dir / "d.yaml")["value"] == 1

def test_fork_missing_source(base_ns):
    """Test error when source doesn't exist."""
    ns = argparse.Namespace(**vars(base_ns))
    ns.source = "ghost"
    ns.name = "dest"
    ns.overrides = []
    ns.message = None
    ns.force = False
    with pytest.raises(SystemExit) as exc:
        cmd_fork(ns)
    assert "Source recipe not found" in str(exc.value)