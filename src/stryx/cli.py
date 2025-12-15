#!/usr/bin/env python3
"""Stryx CLI - Interactive config builder for Pydantic schemas."""

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Type

from pydantic import BaseModel

from .tui import launch_tui


def load_schema_from_path(schema_path: str) -> Type[BaseModel]:
    """
    Load a Pydantic schema class from a path like 'examples/train.py:Config'.

    Args:
        schema_path: Path in format 'module/file.py:ClassName'

    Returns:
        The Pydantic BaseModel class
    """
    if ":" not in schema_path:
        raise ValueError(
            f"Invalid schema path '{schema_path}'. "
            "Expected format: 'path/to/file.py:ClassName'"
        )

    file_path_str, class_name = schema_path.split(":", 1)
    file_path = Path(file_path_str).resolve()

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Load module dynamically
    module_name = file_path.stem
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {file_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    # Get the class
    if not hasattr(module, class_name):
        raise AttributeError(f"Class '{class_name}' not found in {file_path}")

    schema_class = getattr(module, class_name)

    # Verify it's a Pydantic model
    if not isinstance(schema_class, type) or not issubclass(schema_class, BaseModel):
        raise TypeError(
            f"'{class_name}' is not a Pydantic BaseModel. Got: {type(schema_class)}"
        )

    return schema_class


def cmd_new(args: argparse.Namespace) -> None:
    """Create a new config interactively."""
    schema = load_schema_from_path(args.schema)

    # Determine config name
    if args.name:
        config_name = args.name
    else:
        # Auto-generate from schema class name
        config_name = schema.__name__.lower().replace("config", "").replace("cfg", "")
        if not config_name:
            config_name = "config"

    # Determine config directory
    config_dir = args.dir or "config"

    print(f"Launching interactive config builder for {schema.__name__}...")
    print(f"Will save to: {config_dir}/{config_name}.yaml")
    print()

    launch_tui(schema, config_name=config_name, config_dir=config_dir)


def cmd_edit(args: argparse.Namespace) -> None:
    """Edit an existing config."""
    schema = load_schema_from_path(args.schema)
    config_path = Path(args.config)

    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    # Extract name and directory from path
    config_name = config_path.stem
    config_dir = str(config_path.parent)

    print(f"Editing config: {config_path}")
    print()

    launch_tui(schema, config_name=config_name, config_dir=config_dir)


def cmd_show(args: argparse.Namespace) -> None:
    """Display a config file."""
    config_path = Path(args.config)

    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    print(config_path.read_text())


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="stryx",
        description="Interactive config builder for Pydantic schemas",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create a new config interactively
  stryx new examples/train.py:Config

  # Create with specific name and directory
  stryx new examples/train.py:Config --name my_exp --dir configs

  # Edit an existing config
  stryx edit config/my_exp.yaml examples/train.py:Config

  # Show a config file
  stryx show config/my_exp.yaml
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # 'new' command
    parser_new = subparsers.add_parser(
        "new",
        help="Create a new config interactively",
    )
    parser_new.add_argument(
        "schema",
        help="Path to schema in format 'path/to/file.py:ClassName'",
    )
    parser_new.add_argument(
        "--name",
        "-n",
        help="Config name (default: auto-generated from schema)",
    )
    parser_new.add_argument(
        "--dir",
        "-d",
        help="Config directory (default: 'config')",
    )
    parser_new.set_defaults(func=cmd_new)

    # 'edit' command
    parser_edit = subparsers.add_parser(
        "edit",
        help="Edit an existing config",
    )
    parser_edit.add_argument(
        "config",
        help="Path to existing config file",
    )
    parser_edit.add_argument(
        "schema",
        help="Path to schema in format 'path/to/file.py:ClassName'",
    )
    parser_edit.set_defaults(func=cmd_edit)

    # 'show' command
    parser_show = subparsers.add_parser(
        "show",
        help="Display a config file",
    )
    parser_show.add_argument(
        "config",
        help="Path to config file",
    )
    parser_show.set_defaults(func=cmd_show)

    # Parse and execute
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    try:
        args.func(args)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
