from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

# Reserved command names
COMMANDS = frozenset({"run", "edit", "show", "list", "schema", "try", "fork"})


Command = Literal["run", "edit", "show", "list", "help", "create-run-id", "schema", "try", "fork"]


@dataclass
class ParsedArgs:
    """Parsed CLI arguments."""

    command: Command
    recipe: str | None = None
    config_path: Path | None = None
    from_recipe: str | None = None
    overrides: list[str] = field(default_factory=list)
    run_id_override: str | None = None
    recipes_dir_override: Path | None = None
    run_dir_override: Path | None = None


def _is_path(arg: str) -> bool:
    """Check if argument looks like a file path rather than a recipe name.

    Returns True if arg contains path separators or has a YAML/JSON extension.
    """
    return "/" in arg or "\\" in arg or arg.endswith((".yaml", ".yml", ".json"))


def parse_argv(argv: list[str]) -> ParsedArgs:
    """Parse CLI arguments into structured form.

    Handles all argument parsing in one place for consistency.
    """
    # Import locally to avoid circular dependencies if run_id imports back (unlikely but safe)
    from .run_id import parse_run_id_options

    run_id_override, argv = parse_run_id_options(argv)
    run_dir_override: Path | None = None
    recipes_dir_override: Path | None = None

    # Extract dir overrides before command parsing
    filtered: list[str] = []
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg in ("--run-dir", "--runs-dir"):
            if i + 1 >= len(argv):
                raise SystemExit(f"{arg} requires a value")
            run_dir_override = Path(argv[i + 1]).expanduser()
            i += 2
            continue
        if arg.startswith("--run-dir=") or arg.startswith("--runs-dir="):
            run_dir_override = Path(arg.split("=", 1)[1]).expanduser()
            i += 1
            continue

        if arg in ("--configs-dir", "--recipes-dir"):
            if i + 1 >= len(argv):
                raise SystemExit(f"{arg} requires a value")
            recipes_dir_override = Path(argv[i + 1]).expanduser()
            i += 2
            continue
        if arg.startswith("--configs-dir=") or arg.startswith("--recipes-dir="):
            recipes_dir_override = Path(arg.split("=", 1)[1]).expanduser()
            i += 1
            continue

        filtered.append(arg)
        i += 1
    argv = filtered

    # No args → run with defaults
    if not argv:
        return ParsedArgs(
            command="try",
            run_id_override=run_id_override,
            run_dir_override=run_dir_override,
            recipes_dir_override=recipes_dir_override,
        )

    first = argv[0]

    # --help / -h
    if first in ("--help", "-h"):
        return ParsedArgs(
            command="help",
            run_id_override=run_id_override,
            run_dir_override=run_dir_override,
            recipes_dir_override=recipes_dir_override,
        )

    # list (no additional args)
    if first == "list":
        return ParsedArgs(
            command="list",
            run_id_override=run_id_override,
            run_dir_override=run_dir_override,
            recipes_dir_override=recipes_dir_override,
        )

    # create-run-id (optional run-id/style flags already parsed)
    if first == "create-run-id":
        return ParsedArgs(
            command="create-run-id",
            run_id_override=run_id_override,
            run_dir_override=run_dir_override,
            recipes_dir_override=recipes_dir_override,
        )

    # fork [source] [name] [overrides...]
    if first == "fork":
        source: str | None = None
        name: str | None = None
        overrides: list[str] = []

        i = 1
        # Arg 1: source (if not override)
        if i < len(argv) and "=" not in argv[i]:
            source = argv[i]
            i += 1

        # Arg 2: name (if not override)
        if i < len(argv) and "=" not in argv[i]:
            name = argv[i]
            i += 1

        overrides = argv[i:]

        return ParsedArgs(
            command="fork",
            from_recipe=source,
            recipe=name,
            overrides=overrides,
            run_id_override=run_id_override,
            run_dir_override=run_dir_override,
            recipes_dir_override=recipes_dir_override,
        )

    # try [source] [overrides...]
    if first == "try":
        source: str | None = None
        overrides: list[str] = []

        i = 1
        # Arg 1: source (if not override)
        if i < len(argv) and "=" not in argv[i]:
            source = argv[i]
            i += 1

        overrides = argv[i:]

        return ParsedArgs(
            command="try",
            from_recipe=source,
            overrides=overrides,
            run_id_override=run_id_override,
            run_dir_override=run_dir_override,
            recipes_dir_override=recipes_dir_override,
        )

    # run <name|path> [overrides...]
    if first == "run":
        if len(argv) < 2:
            raise SystemExit("'run' requires a recipe name or path")
        target = argv[1]
        if _is_path(target):
            return ParsedArgs(
                command="run",
                config_path=Path(target),
                overrides=argv[2:],
                run_id_override=run_id_override,
                run_dir_override=run_dir_override,
                recipes_dir_override=recipes_dir_override,
            )
        return ParsedArgs(
            command="run",
            recipe=target,
            overrides=argv[2:],
            run_id_override=run_id_override,
            run_dir_override=run_dir_override,
            recipes_dir_override=recipes_dir_override,
        )

    # edit <name>
    if first == "edit":
        if len(argv) < 2:
            raise SystemExit("'edit' requires a recipe name")
        return ParsedArgs(
            command="edit",
            recipe=argv[1],
            run_id_override=run_id_override,
            run_dir_override=run_dir_override,
            recipes_dir_override=recipes_dir_override,
        )

    # show [name|path] [overrides...]
    if first == "show":
        recipe: str | None = None
        config_path: Path | None = None
        overrides = []

        i = 1
        while i < len(argv):
            arg = argv[i]
            if "=" in arg:
                overrides.append(arg)
            elif recipe is None and config_path is None:
                # First non-override arg is the target
                if _is_path(arg):
                    config_path = Path(arg)
                else:
                    recipe = arg
            else:
                raise SystemExit(f"Unexpected argument: {arg}")
            i += 1

        return ParsedArgs(
            command="show",
            recipe=recipe,
            config_path=config_path,
            overrides=overrides,
            run_id_override=run_id_override,
            run_dir_override=run_dir_override,
            recipes_dir_override=recipes_dir_override,
        )

    # schema (no additional args)
    if first == "schema":
        return ParsedArgs(
            command="schema",
            run_id_override=run_id_override,
            run_dir_override=run_dir_override,
            recipes_dir_override=recipes_dir_override,
        )

    # Default: treat all args as overrides → try
    if "=" not in first:
        raise SystemExit(
            f"Unknown command '{first}'.\n"
            f"Did you mean: run {first}  (to run a recipe)\n"
            f"Or use overrides like: {first}=value"
        )

    return ParsedArgs(
        command="try",
        overrides=argv,
        run_id_override=run_id_override,
        run_dir_override=run_dir_override,
        recipes_dir_override=recipes_dir_override,
    )
