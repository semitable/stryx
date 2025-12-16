"""Run ID generation and normalization for Stryx."""

from __future__ import annotations

import logging
import os
import re
import secrets
from datetime import datetime, timezone
from typing import Iterable

logger = logging.getLogger("stryx.run_id")


def parse_run_id_options(argv: list[str]) -> tuple[str | None, list[str]]:
    """Extract run id options from argv and return (run_id, remaining_argv)."""
    run_id_override: str | None = None
    remaining: list[str] = []

    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--run-id":
            if i + 1 >= len(argv):
                raise SystemExit("--run-id requires a value")
            run_id_override = argv[i + 1]
            i += 2
            continue
        if arg.startswith("--run-id="):
            run_id_override = arg.split("=", 1)[1]
            i += 1
            continue

        remaining.append(arg)
        i += 1

    return run_id_override, remaining


def derive_run_id(
    label: str | None = None,
    run_id_override: str | None = None,
) -> str:
    """Select or generate a run id with best-effort stability.

    Priority:
        1) User-provided run_id_override (flag) — conflicts with STRYX_RUN_ID.
        2) STRYX_RUN_ID env var.
        3) Launcher-provided IDs (TORCHELASTIC_RUN_ID, SLURM_JOB_ID, PBS_JOBID, LSB_JOBID).
        4) Generated timestamped petname id (label, if provided, becomes a slug prefix).
    """
    env_run_id = os.getenv("STRYX_RUN_ID")
    if run_id_override and env_run_id:
        raise SystemExit("Cannot use --run-id and STRYX_RUN_ID together; pick one.")

    if run_id_override:
        normalized = _normalize(run_id_override)
        if normalized != run_id_override:
            logger.warning("--run-id contained unsupported characters; using normalized id: %s", normalized)
        else:
            logger.info("Using run id from --run-id: %s", normalized)
        return normalized

    if env_run_id:
        normalized = _normalize(env_run_id)
        if normalized != env_run_id:
            logger.warning("STRYX_RUN_ID contained unsupported characters; using normalized id: %s", normalized)
        else:
            logger.info("Using STRYX_RUN_ID from environment: %s", normalized)
        return normalized

    launcher = _launcher_id()
    if launcher:
        value, source, severity = launcher
        normalized = _normalize(value)
        msg = f"Using {source} from environment for run id: {normalized}"
        if severity == "warning":
            logger.warning("%s (override with --run-id or STRYX_RUN_ID if this launcher runs multiple jobs).", msg)
        else:
            logger.info(msg)
        return normalized

    if _looks_distributed():
        logger.warning(
            "Distributed environment detected but no launcher run id found; run ids may diverge per rank. "
            "Provide --run-id or STRYX_RUN_ID to enforce a shared id."
        )

    run_id = _generate(label=label)
    logger.info("Generated run id (petname): %s", run_id)
    return run_id


def _launcher_id() -> tuple[str, str, str] | None:
    """Return a launcher-provided id and its source."""
    torchelastic = os.getenv("TORCHELASTIC_RUN_ID")
    if torchelastic:
        return torchelastic, "TORCHELASTIC_RUN_ID", "info"

    slurm_job = os.getenv("SLURM_JOB_ID")
    if slurm_job:
        array_task = os.getenv("SLURM_ARRAY_TASK_ID")
        label = f"{slurm_job}-{array_task}" if array_task else slurm_job
        return label, "SLURM_JOB_ID", "warning"

    for key in ("PBS_JOBID", "LSB_JOBID"):
        value = os.getenv(key)
        if value:
            return value, key, "warning"

    return None


def _looks_distributed() -> bool:
    """Heuristic: check env vars set by common launchers."""
    keys = (
        "RANK",
        "LOCAL_RANK",
        "WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
        "TORCHELASTIC_RUN_ID",
        "SLURM_JOB_ID",
        "PBS_JOBID",
        "LSB_JOBID",
    )
    return any(os.getenv(k) for k in keys)


def _generate(label: str | None) -> str:
    """Generate a timestamped run id with a petname and optional label prefix."""
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    base = _petname()
    if label:
        base = f"{_normalize(label)}-{base}"

    return f"run_{ts}_{base}"


def _normalize(raw: str) -> str:
    """Convert arbitrary text into a filesystem-friendly slug."""
    cleaned = re.sub(r"[^A-Za-z0-9]+", "-", raw).strip("-").lower()
    return cleaned[:64] or "run"


def _petname() -> str:
    """Generate a human-friendly petname."""
    try:
        import petname

        return petname.Generate(2, separator="-")
    except Exception:  # pragma: no cover - optional dependency failure
        token = secrets.token_hex(2)
        logger.warning("petname library unavailable; falling back to token '%s'", token)
        return token
