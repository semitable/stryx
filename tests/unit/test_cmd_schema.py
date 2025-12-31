from __future__ import annotations
import argparse
import pytest
from pydantic import BaseModel, ConfigDict, Field
from stryx.commands import cmd_schema

class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(default="default", description="Experiment name")
    value: int = 1

@pytest.fixture
def base_ns(tmp_path):
    """Create a base namespace with context fields."""
    return argparse.Namespace(
        stryx_schema=Config,
        configs_dir=tmp_path / "configs",
        runs_dir=tmp_path / "runs",
        stryx_func=lambda x: None
    )

def test_schema_printing(base_ns, capsys):
    """Test that schema fields and descriptions are printed."""
    # Add command-specific args
    ns = argparse.Namespace(**vars(base_ns))
    ns.json = False
    
    cmd_schema(ns)
    
    captured = capsys.readouterr()
    assert "Schema: test_cmd_schema:Config" in captured.out
    assert "Fields:" in captured.out
    assert "name: str = \"default\"" in captured.out
    assert "# Experiment name" in captured.out
    assert "value: int = 1" in captured.out

def test_schema_json(base_ns, capsys):
    """Test that schema can be printed as JSON."""
    ns = argparse.Namespace(**vars(base_ns))
    ns.json = True
    
    cmd_schema(ns)
    
    captured = capsys.readouterr()
    # It should be valid JSON and contain our fields
    import json
    data = json.loads(captured.out)
    assert data["title"] == "Config"
    assert "name" in data["properties"]
    assert "value" in data["properties"]