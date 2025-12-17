from __future__ import annotations

import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

from .utils import read_yaml, write_yaml


class TeeStream:
    """Redirects writes to both an original stream and a file."""

    def __init__(self, original_stream: TextIO, file_handle: TextIO):
        self.original = original_stream
        self.file = file_handle

    def write(self, data: str) -> None:
        self.original.write(data)
        self.file.write(data)
        self.file.flush()  # Ensure logs are written immediately

    def flush(self) -> None:
        self.original.flush()
        self.file.flush()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.original, name)


class RunContext:
    """Manages execution lifecycle: logging, status updates, and result capture."""

    def __init__(self, manifest_path: Path, is_rank_zero: bool):
        self.manifest_path = manifest_path
        self.is_rank_zero = is_rank_zero
        self.log_file: TextIO | None = None
        self.old_stdout: TextIO | None = None
        self.old_stderr: TextIO | None = None

    def __enter__(self) -> RunContext:
        if not self.is_rank_zero:
            return self

        # Ensure directory exists (it should, but be safe)
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)

        # Open log file
        log_path = self.manifest_path.parent / "stdout.log"
        self.log_file = open(log_path, "w", encoding="utf-8")

        # Setup Tee
        self.old_stdout = sys.stdout
        self.old_stderr = sys.stderr
        
        # Redirect both stdout and stderr to the same log file for a unified timeline
        sys.stdout = TeeStream(self.old_stdout, self.log_file) # type: ignore
        sys.stderr = TeeStream(self.old_stderr, self.log_file) # type: ignore

        # Initial status
        self._update_manifest(
            status="RUNNING",
            started_at=datetime.now(tz=timezone.utc).isoformat()
        )
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if not self.is_rank_zero:
            return

        # Restore streams
        if self.old_stdout:
            sys.stdout = self.old_stdout
        if self.old_stderr:
            sys.stderr = self.old_stderr

        if exc_type:
            # Job Failed
            tb = "".join(traceback.format_exception(exc_type, exc_val, exc_tb))
            self._update_manifest(
                status="FAILED",
                error=str(exc_val),
                traceback=tb,
                finished_at=datetime.now(tz=timezone.utc).isoformat()
            )
        else:
            # If successful exit
            self._update_manifest(finished_at=datetime.now(tz=timezone.utc).isoformat())

        if self.log_file:
            self.log_file.close()

    def record_result(self, result: Any) -> None:
        """Record the execution result and mark as COMPLETED."""
        if not self.is_rank_zero:
            return
            
        self._update_manifest(
            status="COMPLETED",
            result=result,
            # finished_at is handled in __exit__
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