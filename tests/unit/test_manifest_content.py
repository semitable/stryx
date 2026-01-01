from __future__ import annotations
import os
import getpass
import socket
import sys
from unittest.mock import MagicMock
from stryx.lifecycle import record_run_manifest
from stryx.utils import read_yaml, Ctx

def test_manifest_content_fields(tmp_path):
    """Test that manifest contains new fields and excludes resolved_config_path."""
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    
    c = Ctx(
        schema=MagicMock(),
        configs_dir=tmp_path / "configs",
        runs_dir=runs_dir,
        func=lambda x: None
    )
    c.schema.__module__ = "mod"
    c.schema.__name__ = "Cls"
    
    cfg = MagicMock()
    cfg.model_dump.return_value = {}
    
    record_run_manifest(c, cfg, "run_id", {}, [])
    
    manifest_path = runs_dir / "run_id" / "stryx.manifest.yaml"
    data = read_yaml(manifest_path)
    
    # Check new fields
    assert data["user"] == getpass.getuser()
    assert data["host"] == socket.gethostname()
    assert data["pid"] == os.getpid()
    assert data["command"] == sys.argv
    assert data["run_dir"] == str((runs_dir / "run_id").absolute())
    
    # Check removed field
    assert "resolved_config_path" not in data
