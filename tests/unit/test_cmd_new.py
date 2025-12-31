from __future__ import annotations
import argparse
import pytest
from pydantic import BaseModel, ConfigDict
from stryx.commands import cmd_new
from stryx.utils import read_yaml


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = "default_name"
    value: int = 1


@pytest.fixture
def base_ns(tmp_path):
    """Create a base namespace with context fields."""
    return argparse.Namespace(
        stryx_schema=Config,
        configs_dir=tmp_path / "configs",
        runs_dir=tmp_path / "runs",
        stryx_func=lambda x: None,
    )


def test_new_defaults(base_ns):
    """Test creating a new recipe with defaults (sequential name)."""
    # Clone and set command args
    ns = argparse.Namespace(**vars(base_ns))
    ns.recipe = None
    ns.overrides = []
    ns.message = None
    ns.force = False

    # Ensure directory doesn't exist yet to test creation
    assert not base_ns.configs_dir.exists()

    out_path = cmd_new(ns)

    assert out_path.name == "exp_001.yaml"
    assert out_path.exists()
    assert base_ns.configs_dir.exists()

    data = read_yaml(out_path)
    assert data["name"] == "default_name"
    assert data["value"] == 1

    # Verify Metadata
    meta = data["__stryx__"]
    assert meta["type"] == "canonical"
    assert "schema" in meta
    assert meta["schema"].endswith(":Config")
    assert "created_at" in meta
    assert meta["overrides"] == []


def test_new_named(base_ns):
    """Test creating a named recipe."""
    ns = argparse.Namespace(**vars(base_ns))
    ns.recipe = "my_exp"
    ns.overrides = []
    ns.message = None
    ns.force = False

    out_path = cmd_new(ns)

    assert out_path.name == "my_exp.yaml"
    assert out_path.exists()


def test_new_smart_extension(base_ns):
    """Test extension handling."""
    # Case 1: No extension -> adds .yaml
    ns1 = argparse.Namespace(**vars(base_ns))
    ns1.recipe = "exp1"
    ns1.overrides = []
    ns1.message = None
    ns1.force = False
    p1 = cmd_new(ns1)
    assert p1.name == "exp1.yaml"

    # Case 2: Has extension -> keeps it
    ns2 = argparse.Namespace(**vars(base_ns))
    ns2.recipe = "exp2.yml"
    ns2.overrides = []
    ns2.message = None
    ns2.force = False
    p2 = cmd_new(ns2)
    assert p2.name == "exp2.yml"

    # Case 3: Weird dot -> keeps it
    ns3 = argparse.Namespace(**vars(base_ns))
    ns3.recipe = "v1.final"
    ns3.overrides = []
    ns3.message = None
    ns3.force = False
    p3 = cmd_new(ns3)
    assert p3.name == "v1.final"


def test_new_with_overrides(base_ns):
    """Test applying overrides."""
    ns = argparse.Namespace(**vars(base_ns))
    ns.recipe = "overridden"
    ns.overrides = ["value=99", "name=custom"]
    ns.message = None
    ns.force = False

    out_path = cmd_new(ns)
    data = read_yaml(out_path)

    assert data["value"] == 99
    assert data["name"] == "custom"
    assert data["__stryx__"]["overrides"] == ["value=99", "name=custom"]


def test_new_nested_overrides(base_ns):
    """Test creating a recipe with nested overrides."""

    class NestedConfig(BaseModel):
        model_config = ConfigDict(extra="forbid")
        steps: int = 10

    class DeepConfig(BaseModel):
        model_config = ConfigDict(extra="forbid")
        train: NestedConfig = NestedConfig()

    base_ns.stryx_schema = DeepConfig
    
    ns = argparse.Namespace(**vars(base_ns))
    ns.recipe = "deep"
    ns.overrides = ["train.steps=50"]
    ns.message = None
    ns.force = False

    out_path = cmd_new(ns)
    data = read_yaml(out_path)

    assert data["train"]["steps"] == 50
    assert data["__stryx__"]["overrides"] == ["train.steps=50"]


def test_new_invalid_overrides(base_ns):
    """Test that invalid overrides raise an error."""
    ns = argparse.Namespace(**vars(base_ns))
    ns.recipe = "invalid"
    ns.overrides = ["non_existent=1"]
    ns.message = None
    ns.force = False

    with pytest.raises(SystemExit):
        cmd_new(ns)


def test_new_with_message(base_ns):
    """Test adding metadata message."""
    ns = argparse.Namespace(**vars(base_ns))
    ns.recipe = "desc_test"
    ns.overrides = []
    ns.message = "This is a test run"
    ns.force = False

    out_path = cmd_new(ns)
    data = read_yaml(out_path)

    assert data["__stryx__"]["description"] == "This is a test run"


def test_overwrite_protection(base_ns):
    """Test protection against accidental overwrites."""
    # Create first time
    ns = argparse.Namespace(**vars(base_ns))
    ns.recipe = "locked"
    ns.overrides = []
    ns.message = None
    ns.force = False
    
    cmd_new(ns)

    # Try creating again without force
    with pytest.raises(SystemExit) as exc:
        cmd_new(ns)
    assert "already exists" in str(exc.value)


def test_force_overwrite(base_ns):
    """Test forced overwrite."""
    # Create first time with value=1
    ns1 = argparse.Namespace(**vars(base_ns))
    ns1.recipe = "forced"
    ns1.overrides = ["value=1"]
    ns1.message = None
    ns1.force = False
    p1 = cmd_new(ns1)
    assert read_yaml(p1)["value"] == 1

    # Overwrite with value=2
    ns2 = argparse.Namespace(**vars(base_ns))
    ns2.recipe = "forced"
    ns2.overrides = ["value=2"]
    ns2.message = None
    ns2.force = True
    p2 = cmd_new(ns2)

    assert read_yaml(p2)["value"] == 2


def test_sequential_naming(base_ns):
    """Test that default names increment."""
    ns = argparse.Namespace(**vars(base_ns))
    ns.recipe = None
    ns.overrides = []
    ns.message = None
    ns.force = False

    p1 = cmd_new(ns)
    assert p1.name == "exp_001.yaml"

    p2 = cmd_new(ns)
    assert p2.name == "exp_002.yaml"

    # Create a gap (manual file)
    (base_ns.configs_dir / "exp_004.yaml").touch()

    # Should skip to 005
    p3 = cmd_new(ns)
    assert p3.name == "exp_005.yaml"