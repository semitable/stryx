from __future__ import annotations

import getpass
import os
import platform
import socket
import subprocess
import sys
import traceback
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO, TYPE_CHECKING

from .utils import read_yaml, write_yaml

if TYPE_CHECKING:
    from .utils import Ctx

_current_context: ContextVar[RunContext | None] = ContextVar("run_context", default=None)


def current_run() -> RunContext | None:
    """Get the active RunContext, or None if not running inside a stryx run."""
    return _current_context.get()


def resolve_run_path() -> Path:
    """Get the active run directory. Raises RuntimeError if not in a run."""
    ctx = current_run()
    if ctx is None:
        raise RuntimeError(
            "stryx.run_path was accessed outside of an active experiment run.\n"
            "Ensure your function is decorated with @stryx.cli and executed via the CLI."
        )
    return ctx.manifest_path.parent


class TeeStream:
    """Redirects writes to both an original stream and a file."""

    def __init__(self, original_stream: TextIO, file_handle: TextIO):
        self.original = original_stream
        self.file = file_handle

    def write(self, data: str) -> None:
        self.original.write(data)
        self.file.write(data)
        self.file.flush()  # Ensure logs are written immediately

    def flush(
        self,
    ) -> None:
        self.original.flush()
        self.file.flush()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.original, name)


class RunContext:
    """Manages execution lifecycle: logging, status updates, and result capture."""

    def __init__(self, manifest_path: Path, rank: int):
        self.manifest_path = manifest_path
        self.rank = rank
        self.log_file: TextIO | None = None
        self.old_stdout: TextIO | None = None
        self.old_stderr: TextIO | None = None
        self._token: Any = None

    def __enter__(self) -> RunContext:
        self._token = _current_context.set(self)

        # Determine log path
        run_root = self.manifest_path.parent
        is_distributed = os.getenv("WORLD_SIZE") is not None

        if is_distributed:
            log_dir = run_root / "logs"
            try:
                log_dir.mkdir(parents=True, exist_ok=True)
            except FileExistsError:
                pass  # Race condition safety
            log_path = log_dir / f"rank_{self.rank}.log"
        else:
            # Single process - simple log in root
            run_root.mkdir(parents=True, exist_ok=True)
            log_path = run_root / "stdout.log"

        self.log_file = open(log_path, "w", encoding="utf-8")

        # Setup Tee
        self.old_stdout = sys.stdout
        self.old_stderr = sys.stderr

        sys.stdout = TeeStream(self.old_stdout, self.log_file)  # type: ignore
        sys.stderr = TeeStream(self.old_stderr, self.log_file)  # type: ignore

        # Initial status (only rank 0)
        if self.rank == 0:
            self._update_manifest(
                status="RUNNING", started_at=datetime.now(tz=timezone.utc).isoformat()
            )
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._token:
            _current_context.reset(self._token)

        # Restore streams
        if self.old_stdout:
            sys.stdout = self.old_stdout
        if self.old_stderr:
            sys.stderr = self.old_stderr

        # Close log
        if self.log_file:
            self.log_file.close()

        # Update manifest (only rank 0)
        if self.rank != 0:
            return

        if exc_type:
            # Job Failed
            tb = "".join(traceback.format_exception(exc_type, exc_val, exc_tb))
            self._update_manifest(
                status="FAILED",
                error=str(exc_val),
                traceback=tb,
                finished_at=datetime.now(tz=timezone.utc).isoformat(),
            )
        else:
            # If successful exit
            self._update_manifest(finished_at=datetime.now(tz=timezone.utc).isoformat())

    def record_result(self, result: Any) -> None:
        """Record the execution result and mark as COMPLETED."""
        if self.rank != 0:
            return

        self._update_manifest(
            status="COMPLETED",
            result=result,
            # We don't set finished_at here because __exit__ will handle it
        )

    def _update_manifest(self, **kwargs: Any) -> None:
        """Update manifest.yaml with new fields."""
        try:
            if self.manifest_path.exists():
                data = read_yaml(self.manifest_path)
            else:
                data = {}

            data.update(kwargs)
            write_yaml(self.manifest_path, data)
        except Exception:
            # Don't crash the run just because we couldn't update status
            pass


def get_rank() -> int:
    """Get current process rank (0 if not distributed)."""
    for var in ("RANK", "LOCAL_RANK", "SLURM_PROCID", "OMPI_COMM_WORLD_RANK"):
        val = os.getenv(var)
        if val is not None:
            return int(val)
    return 0


def record_run_manifest(
    c: Ctx,
    cfg: Any,
    run_id: str,
    source: dict[str, Any],
    overrides: list[str],
) -> Path:
    """Write a per-run manifest with resolved config and metadata."""
    run_root = c.runs_dir / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    manifest_path = run_root / "manifest.yaml"
    resolved_path = run_root / "config.yaml"

    patch_path = _write_git_patch(run_root)

    # Write resolved config
    try:
        write_yaml(resolved_path, cfg.model_dump(mode="python"))
    except Exception as exc:
        print(f"Warning: failed to write resolved config: {exc}", file=sys.stderr)

    manifest = {
        "run_id": run_id,
        "run_dir": str(run_root.absolute()),
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
        "schema": f"{c.schema.__module__}:{c.schema.__name__}",
        "config_source": source,
        "overrides": overrides or [],
        "command": sys.argv,
        "user": getpass.getuser(),
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "git": _git_info(),
        "python": {"version": platform.python_version()},
    }
    manifest["git"]["untracked"] = _git_untracked_files()
    if patch_path:
        manifest["git"]["patch_file"] = str(patch_path)

    try:
        write_yaml(manifest_path, manifest)
    except Exception as exc:
        print(f"Warning: failed to write run manifest: {exc}", file=sys.stderr)

    return manifest_path


def _run_cmd(cmd: list[str], timeout: float = 2.0) -> str | None:
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def _git_info() -> dict[str, Any]:
    sha = _run_cmd(["git", "rev-parse", "HEAD"])
    dirty = False
    if sha:
        status = _run_cmd(["git", "status", "--porcelain"])
        dirty = bool(status)
    return {"sha": sha, "dirty": dirty}


def _git_untracked_files() -> list[str]:
    if not _run_cmd(["git", "rev-parse", "--is-inside-work-tree"]):
        return []
    output = _run_cmd(["git", "ls-files", "--others", "--exclude-standard"])
    if not output:
        return []
    return [line for line in output.splitlines() if line.strip()]





def _write_git_patch(run_root: Path) -> Path | None:
    if not _run_cmd(["git", "rev-parse", "--is-inside-work-tree"]):
        return None
    patch = _run_cmd(["git", "diff", "--patch", "HEAD"])
    if not patch:
        return None
    out_path = run_root / "git.patch"
    try:
        out_path.write_text(patch + "\n", encoding="utf-8")
        return out_path
    except OSError:
        return None
