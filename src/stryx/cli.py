#!/usr/bin/env python3
"""Stryx CLI - shows usage information.

The main entry point for Stryx is your training script with the @stryx.cli decorator.
This standalone command provides general information and utilities.
"""
import sys


def main() -> None:
    """Main CLI entry point."""
    print("""Stryx - Typed Configuration Compiler for ML Experiments

Stryx works through a decorator on your training script:

    import stryx
    from pydantic import BaseModel

    class Config(BaseModel):
        lr: float = 1e-4
        batch_size: int = 32

    @stryx.cli(schema=Config)
    def main(cfg: Config):
        train(cfg)

    if __name__ == "__main__":
        main()

Then run your script directly:

    python train.py                      # Run with defaults
    python train.py lr=1e-3              # Run with overrides
    python train.py new my_exp lr=1e-3   # Save recipe
    python train.py run my_exp           # Run from recipe
    python train.py edit my_exp          # Edit in TUI

For more information, see: https://github.com/user/stryx
""")
    sys.exit(0)


if __name__ == "__main__":
    main()
