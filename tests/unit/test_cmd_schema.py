from __future__ import annotations
import pytest
from unittest.mock import MagicMock
import typer
from pydantic import BaseModel, ConfigDict, Field
from stryx.commands import recipe_schema
from stryx.utils import Ctx

class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(default="default", description="Experiment name")
    value: int = 1

@pytest.fixture
def mock_ctx(tmp_path):
    c = Ctx(
        schema=Config,
        configs_dir=tmp_path / "configs",
        runs_dir=tmp_path / "runs",
        func=lambda x: None
    )
    ctx = MagicMock(spec=typer.Context)
    ctx.obj = c
    return ctx

def test_schema_printing(mock_ctx, capsys):
    recipe_schema(mock_ctx)
    captured = capsys.readouterr()
    assert "Schema: test_cmd_schema:Config" in captured.out
    assert "name: str = \"default\"" in captured.out

def test_schema_json(mock_ctx, capsys):
    recipe_schema(mock_ctx, json_out=True)
    captured = capsys.readouterr()
    import json
    data = json.loads(captured.out)
    assert data["title"] == "Config"
    assert "name" in data["properties"]
    assert "__stryx__" in data["properties"]
    assert data["properties"]["__stryx__"]["title"] == "Stryx Metadata"