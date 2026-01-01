from __future__ import annotations
import pytest
from unittest.mock import MagicMock
import typer
from pydantic import BaseModel, ConfigDict
from stryx.commands import recipe_new
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
    assert not c.configs_dir.exists()

    out_path = recipe_new(mock_ctx)

    assert out_path.name == "exp_001.yaml"
    assert out_path.exists()
    assert c.configs_dir.exists()

    data = read_yaml(out_path)
    assert data["name"] == "default_name"
    assert data["value"] == 1
    assert data["__stryx__"]["type"] == "canonical"


def test_new_named(mock_ctx):
    """Test creating a named recipe."""
    out_path = recipe_new(mock_ctx, name="my_exp")
    assert out_path.name == "my_exp.yaml"
    assert out_path.exists()


def test_new_smart_extension(mock_ctx):
    """Test extension handling."""
    p1 = recipe_new(mock_ctx, name="exp1")
    assert p1.name == "exp1.yaml"

    p2 = recipe_new(mock_ctx, name="exp2.yml")
    assert p2.name == "exp2.yml"


def test_new_with_overrides(mock_ctx):
    """Test applying overrides."""
    out_path = recipe_new(
        mock_ctx, name="overridden", overrides=["value=99", "name=custom"]
    )
    data = read_yaml(out_path)

    assert data["value"] == 99
    assert data["name"] == "custom"
    assert data["__stryx__"]["overrides"] == ["value=99", "name=custom"]


def test_new_implicit_overrides(mock_ctx):
    """Test `new value=99` (implicit name)."""
    # recipe_new(name="value=99") -> name=None, overrides=["value=99"]
    out_path = recipe_new(mock_ctx, name="value=99")
    
    assert out_path.name == "exp_001.yaml"
    data = read_yaml(out_path)
    assert data["value"] == 99
    assert data["__stryx__"]["overrides"] == ["value=99"]


def test_new_nested_overrides(mock_ctx):
    """Test creating a recipe with nested overrides."""
    class DeepConfig(BaseModel):
        model_config = ConfigDict(extra="forbid")
        nested: dict[str, int] = {}

    mock_ctx.obj.schema = DeepConfig
    out_path = recipe_new(mock_ctx, name="deep", overrides=["nested.steps=50"])
    data = read_yaml(out_path)
    assert data["nested"]["steps"] == 50


def test_new_invalid_overrides(mock_ctx):
    """Test that invalid overrides raise SystemExit."""
    with pytest.raises(SystemExit):
        recipe_new(mock_ctx, name="invalid", overrides=["non_existent=1"])


def test_new_with_message(mock_ctx):
    """Test adding metadata message."""
    out_path = recipe_new(mock_ctx, name="desc_test", message="This is a test run")
    data = read_yaml(out_path)
    assert data["__stryx__"]["description"] == "This is a test run"


def test_overwrite_protection(mock_ctx):
    """Test protection against accidental overwrites."""
    recipe_new(mock_ctx, name="locked")

    with pytest.raises(typer.Exit):
        recipe_new(mock_ctx, name="locked")


def test_force_overwrite(mock_ctx):
    """Test forced overwrite."""
    p1 = recipe_new(mock_ctx, name="forced", overrides=["value=1"])
    assert read_yaml(p1)["value"] == 1

    p2 = recipe_new(mock_ctx, name="forced", overrides=["value=2"], force=True)
    assert read_yaml(p2)["value"] == 2
