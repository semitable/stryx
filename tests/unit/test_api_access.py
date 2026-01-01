import stryx
import pytest
from pathlib import Path
from stryx.lifecycle import RunContext

def test_run_path_outside_context():
    """Accessing run_path outside a run should raise RuntimeError."""
    with pytest.raises(RuntimeError, match="outside of an active experiment"):
        _ = stryx.run_path

def test_run_path_inside_context(tmp_path):
    """Accessing run_path inside a run should return the run dir."""
    manifest = tmp_path / "run_1" / "stryx.manifest.yaml"
    manifest.parent.mkdir()
    
    with RunContext(manifest, rank=0):
        path = stryx.run_path
        assert isinstance(path, Path)
        assert path == manifest.parent
        assert path.name == "run_1"
