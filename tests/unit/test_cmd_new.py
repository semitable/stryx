from __future__ import annotations
import argparse
import pytest
from pathlib import Path
from pydantic import BaseModel, ConfigDict
from stryx.context import Ctx
from stryx.commands import cmd_new
from stryx.utils import read_yaml


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = "default_name"
    value: int = 1


@pytest.fixture
def ctx(tmp_path):
    """Create a context with temporary directories."""
    return Ctx(
        schema=Config,
        configs_dir=tmp_path / "configs",
        runs_dir=tmp_path / "runs",
    )


def test_new_defaults(ctx):
    """Test creating a new recipe with defaults (sequential name)."""
    ns = argparse.Namespace(recipe=None, overrides=[], message=None, force=False)

    # Ensure directory doesn't exist yet to test creation
    assert not ctx.configs_dir.exists()

    out_path = cmd_new(ctx, ns)

    assert out_path.name == "exp_001.yaml"
    assert out_path.exists()
    assert ctx.configs_dir.exists()

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


def test_new_named(ctx):
    """Test creating a named recipe."""
    ns = argparse.Namespace(recipe="my_exp", overrides=[], message=None, force=False)

    out_path = cmd_new(ctx, ns)

    assert out_path.name == "my_exp.yaml"
    assert out_path.exists()


def test_new_smart_extension(ctx):
    """Test extension handling."""
    # Case 1: No extension -> adds .yaml
    ns = argparse.Namespace(recipe="exp1", overrides=[], message=None, force=False)
    p1 = cmd_new(ctx, ns)
    assert p1.name == "exp1.yaml"

    # Case 2: Has extension -> keeps it
    ns = argparse.Namespace(recipe="exp2.yml", overrides=[], message=None, force=False)
    p2 = cmd_new(ctx, ns)
    assert p2.name == "exp2.yml"

    # Case 3: Weird dot -> keeps it (as per current logic "if '.' in name")
    ns = argparse.Namespace(recipe="v1.final", overrides=[], message=None, force=False)
    p3 = cmd_new(ctx, ns)
    assert p3.name == "v1.final"


def test_new_with_overrides(ctx):
    """Test applying overrides."""
    ns = argparse.Namespace(
        recipe="overridden",
        overrides=["value=99", "name=custom"],
        message=None,
        force=False,
    )

    out_path = cmd_new(ctx, ns)
    data = read_yaml(out_path)

    assert data["value"] == 99
    assert data["name"] == "custom"
    assert data["__stryx__"]["overrides"] == ["value=99", "name=custom"]


def test_new_nested_overrides(ctx):
    """Test creating a recipe with nested overrides."""

    class NestedConfig(BaseModel):
        model_config = ConfigDict(extra="forbid")
        steps: int = 10

    class DeepConfig(BaseModel):
        model_config = ConfigDict(extra="forbid")
        train: NestedConfig = NestedConfig()

    ctx.schema = DeepConfig
    ns = argparse.Namespace(
        recipe="deep", overrides=["train.steps=50"], message=None, force=False
    )

    out_path = cmd_new(ctx, ns)
    data = read_yaml(out_path)

    assert data["train"]["steps"] == 50
    assert data["__stryx__"]["overrides"] == ["train.steps=50"]


def test_new_invalid_overrides(ctx):
    """Test that invalid overrides raise an error."""
    ns = argparse.Namespace(
        recipe="invalid", overrides=["non_existent=1"], message=None, force=False
    )

    # build_config (called by cmd_new) should raise SystemExit due to validate_or_die
    with pytest.raises(SystemExit):
        cmd_new(ctx, ns)


def test_new_with_message(ctx):
    """Test adding metadata message."""
    ns = argparse.Namespace(
        recipe="desc_test", overrides=[], message="This is a test run", force=False
    )

    out_path = cmd_new(ctx, ns)
    data = read_yaml(out_path)

    assert data["__stryx__"]["description"] == "This is a test run"


def test_overwrite_protection(ctx):
    """Test protection against accidental overwrites."""
    # Create first time
    ns = argparse.Namespace(recipe="locked", overrides=[], message=None, force=False)
    cmd_new(ctx, ns)

    # Try creating again without force
    with pytest.raises(SystemExit) as exc:
        cmd_new(ctx, ns)
    assert "already exists" in str(exc.value)


def test_force_overwrite(ctx):
    """Test forced overwrite."""
    # Create first time with value=1
    ns1 = argparse.Namespace(
        recipe="forced", overrides=["value=1"], message=None, force=False
    )
    p1 = cmd_new(ctx, ns1)
    assert read_yaml(p1)["value"] == 1

    # Overwrite with value=2
    ns2 = argparse.Namespace(
        recipe="forced", overrides=["value=2"], message=None, force=True
    )
    p2 = cmd_new(ctx, ns2)

    assert read_yaml(p2)["value"] == 2


def test_sequential_naming(ctx):
    """Test that default names increment."""
    ns = argparse.Namespace(recipe=None, overrides=[], message=None, force=False)

    p1 = cmd_new(ctx, ns)
    assert p1.name == "exp_001.yaml"

    p2 = cmd_new(ctx, ns)
    assert p2.name == "exp_002.yaml"

    # Create a gap (manual file)
    (ctx.configs_dir / "exp_004.yaml").touch()

    # Should skip to 005
    p3 = cmd_new(ctx, ns)
    assert p3.name == "exp_005.yaml"

