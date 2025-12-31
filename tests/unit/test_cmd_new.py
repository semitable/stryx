from __future__ import annotations
import pytest
from unittest.mock import MagicMock
import typer
from pydantic import BaseModel, ConfigDict
from stryx.commands import cmd_new
from stryx.utils import Ctx, read_yaml


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = "default_name"
    value: int = 1


@pytest.fixture
def mock_ctx(tmp_path):
    """Create a mock Typer context with Stryx Ctx."""
    c = Ctx(
        schema=Config,
        configs_dir=tmp_path / "configs",
        runs_dir=tmp_path / "runs",
        func=lambda x: None,
    )
    ctx = MagicMock(spec=typer.Context)
    ctx.obj = c
    return ctx


def test_new_defaults(mock_ctx):
    """Test creating a new recipe with defaults (sequential name)."""
    c = mock_ctx.obj
    # Ensure directory doesn't exist yet to test creation
    assert not c.configs_dir.exists()

    out_path = cmd_new(mock_ctx)

    assert out_path.name == "exp_001.yaml"
    assert out_path.exists()
    assert c.configs_dir.exists()

    data = read_yaml(out_path)
    assert data["name"] == "default_name"
    assert data["value"] == 1

    # Verify Metadata
    meta = data["__stryx__"]
    assert meta["type"] == "canonical"
    assert "schema" in meta


def test_new_named(mock_ctx):
    """Test creating a named recipe."""
    out_path = cmd_new(mock_ctx, recipe="my_exp")
    assert out_path.name == "my_exp.yaml"
    assert out_path.exists()


def test_new_smart_extension(mock_ctx):
    """Test extension handling."""
    p1 = cmd_new(mock_ctx, recipe="exp1")
    assert p1.name == "exp1.yaml"

    p2 = cmd_new(mock_ctx, recipe="exp2.yml")
    assert p2.name == "exp2.yml"

    p3 = cmd_new(mock_ctx, recipe="v1.final")
    assert p3.name == "v1.final"


def test_new_with_overrides(mock_ctx):
    """Test applying overrides."""
    out_path = cmd_new(
        mock_ctx, recipe="overridden", overrides=["value=99", "name=custom"]
    )
    data = read_yaml(out_path)

    assert data["value"] == 99
    assert data["name"] == "custom"
    assert data["__stryx__"]["overrides"] == ["value=99", "name=custom"]


def test_new_implicit_overrides(mock_ctx):
    """Test `new value=99` (implicit name)."""
    # This simulates `stryx new value=99`
    # recipe="value=99", overrides=[]
    out_path = cmd_new(mock_ctx, recipe="value=99")
    
    assert out_path.name == "exp_001.yaml" # Auto-generated name
    data = read_yaml(out_path)
    assert data["value"] == 99
    assert data["__stryx__"]["overrides"] == ["value=99"]


def test_new_nested_overrides(mock_ctx):
    """Test creating a recipe with nested overrides."""

    class NestedConfig(BaseModel):
        model_config = ConfigDict(extra="forbid")
        steps: int = 10

    class DeepConfig(BaseModel):
        model_config = ConfigDict(extra="forbid")
        train: NestedConfig = NestedConfig()

    mock_ctx.obj.schema = DeepConfig

    out_path = cmd_new(mock_ctx, recipe="deep", overrides=["train.steps=50"])
    data = read_yaml(out_path)

    assert data["train"]["steps"] == 50


def test_new_invalid_overrides(mock_ctx):
    """Test that invalid overrides raise Exit."""
    with pytest.raises(SystemExit):
        cmd_new(mock_ctx, recipe="invalid", overrides=["non_existent=1"])


def test_new_with_message(mock_ctx):
    """Test adding metadata message."""
    out_path = cmd_new(mock_ctx, recipe="desc_test", message="This is a test run")
    data = read_yaml(out_path)
    assert data["__stryx__"]["description"] == "This is a test run"


def test_overwrite_protection(mock_ctx):
    """Test protection against accidental overwrites."""
    cmd_new(mock_ctx, recipe="locked")

    # Try creating again without force
    with pytest.raises(typer.Exit):
        cmd_new(mock_ctx, recipe="locked")


def test_force_overwrite(mock_ctx):
    """Test forced overwrite."""
    p1 = cmd_new(mock_ctx, recipe="forced", overrides=["value=1"])
    assert read_yaml(p1)["value"] == 1

    p2 = cmd_new(mock_ctx, recipe="forced", overrides=["value=2"], force=True)
    assert read_yaml(p2)["value"] == 2


def test_sequential_naming(mock_ctx):
    """Test that default names increment."""
    p1 = cmd_new(mock_ctx)
    assert p1.name == "exp_001.yaml"

    p2 = cmd_new(mock_ctx)
    assert p2.name == "exp_002.yaml"