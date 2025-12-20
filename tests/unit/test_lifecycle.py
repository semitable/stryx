from __future__ import annotations
import pytest
from unittest.mock import MagicMock, patch
from stryx.lifecycle import RunContext, TeeStream, record_run_manifest
from stryx.context import Ctx
from stryx.utils import read_yaml, write_yaml

@pytest.fixture
def mock_ctx(tmp_path):
    """Create a mock context."""
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    
    mock_schema = MagicMock()
    mock_schema.__module__ = "test_module"
    mock_schema.__name__ = "TestSchema"
    
    return Ctx(
        schema=mock_schema,
        configs_dir=tmp_path / "configs",
        runs_dir=runs_dir,
        func=lambda x: None
    )

def test_tee_stream(tmp_path):
    """Test that TeeStream writes to both original stream and file."""
    log_file = tmp_path / "log.txt"
    with open(log_file, "w") as f:
        mock_stdout = MagicMock()
        tee = TeeStream(mock_stdout, f)
        
        tee.write("Hello")
        tee.flush()
        
    # Check mock stdout
    mock_stdout.write.assert_called_with("Hello")
    mock_stdout.flush.assert_called()
    
    # Check file
    assert log_file.read_text() == "Hello"

def test_run_context_lifecycle(tmp_path):
    """Test RunContext status updates and logging."""
    manifest_path = tmp_path / "run_1" / "manifest.yaml"
    manifest_path.parent.mkdir()
    
    # Init manifest
    write_yaml(manifest_path, {"status": "PENDING"})
    
    # Run
    with RunContext(manifest_path, rank=0) as ctx:
        # Check running state
        data = read_yaml(manifest_path)
        assert data["status"] == "RUNNING"
        assert "started_at" in data
        
        print("Log message") # Should go to log file
        ctx.record_result("My Result")
        
    # Check finished state
    data = read_yaml(manifest_path)
    assert data["status"] == "COMPLETED"
    assert data["result"] == "My Result"
    assert "finished_at" in data
    
    # Check log file
    log_file = tmp_path / "run_1" / "stdout.log"
    assert log_file.exists()
    assert "Log message" in log_file.read_text()

def test_run_context_failure(tmp_path):
    """Test RunContext handles exceptions."""
    manifest_path = tmp_path / "run_fail" / "manifest.yaml"
    manifest_path.parent.mkdir()
    
    with pytest.raises(ValueError):
        with RunContext(manifest_path, rank=0):
            raise ValueError("Boom")
            
    data = read_yaml(manifest_path)
    assert data["status"] == "FAILED"
    assert "Boom" in data["error"]
    assert "traceback" in data

def test_run_context_non_zero_rank(tmp_path):
    """Test that non-zero ranks log to separate files and don't touch manifest."""
    manifest_path = tmp_path / "run_dist" / "manifest.yaml"
    manifest_path.parent.mkdir()
    
    # Simulate distributed env
    with patch.dict("os.environ", {"WORLD_SIZE": "2"}):
        with RunContext(manifest_path, rank=1):
            print("Rank 1 working")
            
    # Manifest shouldn't exist/be created by rank 1
    assert not manifest_path.exists()
    
    # Log should be in logs/rank_1.log
    log_file = tmp_path / "run_dist" / "logs" / "rank_1.log"
    assert log_file.exists()
    assert "Rank 1 working" in log_file.read_text()

@patch("stryx.lifecycle._run_cmd")
def test_record_manifest_git_info(mock_run_cmd, mock_ctx):
    """Test that git info is captured in manifest."""
    # Mock git responses
    def side_effect(cmd, **kwargs):
        cmd_str = " ".join(cmd)
        if "rev-parse HEAD" in cmd_str:
            return "abcdef123"
        if "status --porcelain" in cmd_str:
            return " M dirty.py"
        if "rev-parse --is-inside-work-tree" in cmd_str:
            return "true"
        if "ls-files --others" in cmd_str:
            return "untracked.py"
        if "diff --patch" in cmd_str:
            return "diff content"
        return None
    mock_run_cmd.side_effect = side_effect
    
    cfg = MagicMock()
    cfg.model_dump.return_value = {"param": 1}
    
    record_run_manifest(mock_ctx, cfg, "run_git", {"name": "src"}, [])
    
    manifest = read_yaml(mock_ctx.runs_dir / "run_git" / "manifest.yaml")
    
    assert manifest["git"]["sha"] == "abcdef123"
    assert manifest["git"]["dirty"] is True
    assert manifest["git"]["untracked"] == ["untracked.py"]
    assert "git.patch" in manifest["git"]["patch_file"]
    
    # Verify patch file created
    patch_path = mock_ctx.runs_dir / "run_git" / "git.patch"
    assert patch_path.read_text().strip() == "diff content"

@patch("stryx.lifecycle._run_cmd")
def test_record_manifest_no_git(mock_run_cmd, mock_ctx):
    """Test manifest creation outside git repo."""
    mock_run_cmd.return_value = None # Git commands fail/return empty
    
    cfg = MagicMock()
    cfg.model_dump.return_value = {}
    
    record_run_manifest(mock_ctx, cfg, "run_no_git", {}, [])
    
    manifest = read_yaml(mock_ctx.runs_dir / "run_no_git" / "manifest.yaml")
    assert manifest["git"]["sha"] is None
    assert "patch_file" not in manifest["git"]
