import subprocess
import sys
import os
import pytest
import yaml
import importlib.util


def run_torch(args, cwd):
    """Run command with torchrun using current python environment."""
    cmd = [sys.executable, "-m", "torch.distributed.run"] + args
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)

def test_distributed_failure_no_id(app_script, tmp_path):
    """Fails if torchrun used without shared ID."""
    if not importlib.util.find_spec("torch"):
        pytest.skip("torch not installed")

    # Run 'try' without explicit ID
    result = run_torch(["--nproc_per_node=2", str(app_script), "try"], cwd=tmp_path)
    
    assert result.returncode != 0
    assert "Distributed environment detected" in result.stderr
    assert "no shared Run ID" in result.stderr

def test_distributed_success_with_id(app_script, tmp_path):
    """Succeeds if ID provided via --run-id."""
    if not importlib.util.find_spec("torch"):
        pytest.skip("torch not installed")

    run_id = "dist_test_01"
    # Use 'try' with explicit ID
    result = run_torch(
        ["--nproc_per_node=2", str(app_script), "try", "--run-id", run_id], 
        cwd=tmp_path
    )
    
    if result.returncode != 0:
        pytest.fail(f"Run failed: {result.stderr}")
    
    # Check artifacts
    run_dir = tmp_path / "runs" / run_id
    assert run_dir.exists()
    
    # Manifest
    manifest = yaml.safe_load((run_dir / "manifest.yaml").read_text())
    assert manifest["status"] == "COMPLETED"
    assert manifest["result"]["rank"] == "0" # Only rank 0 writes result
    
    # Logs
    log_dir = run_dir / "logs"
    assert log_dir.exists()
    assert (log_dir / "rank_0.log").exists()
    assert (log_dir / "rank_1.log").exists()
    
    # Verify content of logs
    log0 = (log_dir / "rank_0.log").read_text()
    assert "Rank 0 running" in log0

def test_lifecycle_failure(app_script, tmp_path):
    """Captures exception in manifest on crash."""
    if not importlib.util.find_spec("torch"):
        pytest.skip("torch not installed")

    run_id = "crash_test"
    # Force crash
    result = run_torch(
        ["--nproc_per_node=1", str(app_script), "try", "--run-id", run_id, "crash=true"],
        cwd=tmp_path
    )
    
    assert result.returncode != 0
    
    run_dir = tmp_path / "runs" / run_id
    manifest = yaml.safe_load((run_dir / "manifest.yaml").read_text())
    
    assert manifest["status"] == "FAILED"
    assert "ValueError: Boom" in manifest["traceback"]

def test_strict_run_overrides(app_script, tmp_path):
    """'run' command should reject overrides."""
    # First create a canonical recipe
    subprocess.run(
        [sys.executable, str(app_script), "new", "canonical"], 
        cwd=tmp_path, check=True
    )
    
    # Try to run it with overrides
    result = subprocess.run(
        [sys.executable, str(app_script), "run", "canonical", "crash=true"],
        cwd=tmp_path, capture_output=True, text=True
    )
    
    assert result.returncode != 0
    assert "strict" in result.stderr
    assert "does not accept overrides" in result.stderr

def test_fork_vs_try_paths(app_script, tmp_path):
    """Verify 'try' saves to scratches and 'fork' saves to root."""
    # 1. Try
    subprocess.run(
        [sys.executable, str(app_script), "try", "message=scratch"],
        cwd=tmp_path, check=True
    )
    scratches = list((tmp_path / "configs" / "scratches").glob("*.yaml"))
    assert len(scratches) == 1
    
    # 2. Fork
    subprocess.run(
        [sys.executable, str(app_script), "fork", "defaults", "canonical_fork", "message=forked"],
        cwd=tmp_path, check=True
    )
    canonical = tmp_path / "configs" / "canonical_fork.yaml"
    assert canonical.exists()
    assert canonical.parent.name == "configs"

def test_manual_distributed_check(app_script, tmp_path):
    """Simulate manual distributed launch (no torchrun) via env vars."""
    env = os.environ.copy()
    env["RANK"] = "0"
    env["WORLD_SIZE"] = "2"
    # Remove launcher vars if present
    env.pop("TORCHELASTIC_RUN_ID", None)
    
    # No ID source -> Should fail
    result = subprocess.run(
        [sys.executable, str(app_script), "try"],
        cwd=tmp_path, env=env, capture_output=True, text=True
    )
    
    assert result.returncode != 0
    assert "Distributed environment detected" in result.stderr
    
    # With manual ID -> Should succeed
    env["STRYX_RUN_ID"] = "manual_id"
    result = subprocess.run(
        [sys.executable, str(app_script), "try"],
        cwd=tmp_path, env=env, capture_output=True, text=True
    )
    assert result.returncode == 0
    assert (tmp_path / "runs" / "manual_id").exists()