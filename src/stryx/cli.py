#!/usr/bin/env python3
"""Stryx CLI - minimal helpers and usage info."""

from __future__ import annotations

import sys
from pathlib import Path

from .run_id import derive_run_id, parse_run_id_options


def main() -> None:
    """Main CLI entry point."""
    run_id_override, argv = parse_run_id_options(sys.argv[1:])

    if argv and argv[0] in ("--help", "-h"):
        _print_help(Path(sys.argv[0]).name)
        sys.exit(0)

    if argv and argv[0] == "create-run-id":
        if any(tok in ("--help", "-h") for tok in argv[1:]):
            _print_help(Path(sys.argv[0]).name)
            sys.exit(0)
        run_id = derive_run_id(label=Path(sys.argv[0]).stem, run_id_override=run_id_override)
        print(run_id)
        sys.exit(0)

    _print_help(Path(sys.argv[0]).name)
    sys.exit(0)


def _print_help(prog: str) -> None:
    print(
        f"""Stryx - Typed Configuration Compiler for ML Experiments

Usage:
  {prog} --help                       Show this help
  {prog} create-run-id [options]      Print a generated run id

Run ID options:
  --run-id <id>              Use an explicit run id (conflicts with STRYX_RUN_ID)

Typical usage is via the @stryx.cli decorator on your script:
  python train.py                      # Run with defaults
  python train.py lr=1e-3              # Run with overrides
  python train.py new my_exp lr=1e-3   # Save recipe
  python train.py run my_exp           # Run from recipe
  python train.py edit my_exp          # Edit in TUI
"""
    )


if __name__ == "__main__":
    main()
