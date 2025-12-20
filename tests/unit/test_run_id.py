"""Tests for run_id derivation logic."""
import pytest
from stryx.run_id import derive_run_id

def test_explicit_override():
    """--run-id flag takes precedence."""
    assert derive_run_id(run_id_override="my-id") == "my-id"

def test_env_override(monkeypatch):
    """STRYX_RUN_ID env var takes precedence."""
    monkeypatch.setenv("STRYX_RUN_ID", "env-id")
    assert derive_run_id() == "env-id"

def test_override_beats_env(monkeypatch):
    """Flag beats Env."""
    monkeypatch.setenv("STRYX_RUN_ID", "env-id")
    assert derive_run_id(run_id_override="flag-id") == "flag-id"

def test_torchrun_ignored(monkeypatch):
    """TORCHELASTIC_RUN_ID is explicitly ignored due to unreliability."""
    monkeypatch.setenv("TORCHELASTIC_RUN_ID", "12345")
    # It should fall back to generating a petname (unless RANK set)
    assert derive_run_id().startswith("run_")

def test_normalization():
    """Ids are normalized."""
    assert derive_run_id(run_id_override="Bad ID/Value") == "Bad-ID-Value"

def test_local_fallback():
    """Generates petname locally."""
    id1 = derive_run_id()
    assert id1.startswith("run_")
    
    # Check label
    id2 = derive_run_id(label="My Exp")
    assert "My-Exp" in id2

def test_distributed_safety_check(monkeypatch):
    """Raises SystemExit if RANK is set but no ID source."""
    monkeypatch.setenv("RANK", "0")
    # No ID override
    
    with pytest.raises(SystemExit) as exc:
        derive_run_id()
    
    msg = str(exc.value)
    assert "Distributed environment detected" in msg
    assert "no shared Run ID" in msg

def test_slurm_id_accepted(monkeypatch):
    """SLURM_JOB_ID is accepted as a trusted shared ID."""
    monkeypatch.setenv("SLURM_JOB_ID", "999")
    assert derive_run_id() == "999"
