"""Run ID generation and normalization for Stryx.

Policy:
1. User overrides (--run-id, STRYX_RUN_ID) take absolute precedence.
2. Slurm Job ID (SLURM_JOB_ID) is trusted as a shared ID.
3. If distributed environment detected (RANK set) and no ID found -> Error.
4. Local fallback -> Timestamp + Petname.
"""

from __future__ import annotations

import logging
import os
import re
import secrets
from datetime import datetime, timezone

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
    """Select or generate a run id with strict distributed safety."""
    
    # 1. Explicit User Override
    if run_id_override:
        return _validate_and_log(run_id_override, "user flag")

    # 2. Environment Override
    env_run_id = os.getenv("STRYX_RUN_ID")
    if env_run_id:
        return _validate_and_log(env_run_id, "STRYX_RUN_ID")

    # 3. Slurm (Trusted Shared ID)
    slurm_id = os.getenv("SLURM_JOB_ID")
    if slurm_id:
        task_id = os.getenv("SLURM_ARRAY_TASK_ID")
        full_id = f"{slurm_id}_{task_id}" if task_id else slurm_id
        return _validate_and_log(full_id, "SLURM_JOB_ID")

    # 4. Distributed Safety Check
    if _is_distributed_context():
        # We detected distributed execution but found no shared ID source.
        # We cannot safely auto-generate (ranks would diverge).
        raise SystemExit(
            "Error: Distributed environment detected but no shared Run ID found.\n"
            "Stryx requires a consistent ID across all ranks.\n\n"
            "Solution: Provide a run id explicitly.\n"
            "  export STRYX_RUN_ID=$(stryx create-run-id)"
            "  torchrun ...\n"
            "\n"
            "Or pass --run-id <id> to your script."
        )

    # 5. Local Fallback (Timestamp + Petname)
    run_id = _generate_local_id(label)
    logger.info(f"Generated local run id: {run_id}")
    return run_id


def _validate_and_log(raw_id: str, source: str) -> str:
    """Normalize and log the selected ID."""
    normalized = _normalize(raw_id)
    if normalized != raw_id:
        logger.warning(
            f"Run ID from {source} contained unsupported characters. Normalized: '{raw_id}' -> '{normalized}'"
        )
    else:
        logger.debug(f"Using run id from {source}: {normalized}")
    return normalized


def _is_distributed_context() -> bool:
    """Check if the environment looks distributed."""
    # Only check standard rank variables. 
    # We purposefully ignore TORCHELASTIC_RUN_ID as it can be unreliable/opaque.
    dist_vars = ["RANK", "LOCAL_RANK", "PMI_RANK", "OMPI_COMM_WORLD_RANK"]
    return any(os.getenv(k) is not None for k in dist_vars)


def _generate_local_id(label: str | None) -> str:
    """Generate a timestamped petname."""
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    base = _petname()
    if label:
        base = f"{_normalize(label)}-{base}"
    return f"run_{{ts}}_{base}"


def _normalize(raw: str) -> str:
    """Convert arbitrary text into a filesystem-friendly slug."""
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "-", raw).strip("-")
    return cleaned[:128] or "run"

def _petname() -> str:
    """Generate a human-friendly petname."""
    try:
        import petname
        return petname.generate(2, separator="-")
    except Exception:
        return secrets.token_hex(2)
