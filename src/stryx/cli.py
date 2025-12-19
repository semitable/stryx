import argparse
import sys
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel

from stryx.context import Ctx
from stryx.commands import cmd_new, cmd_fork


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="stryx")
    # global overrides (no manual scanning loop)
    p.add_argument("--runs-dir", dest="runs_dir", type=Path)
    p.add_argument("--configs-dir", dest="configs_dir", type=Path)
    # add your run-id options here too

    sub = p.add_subparsers(dest="cmd", required=True)

    # list <configs|runs>
    p_list = sub.add_parser("list")
    sub_list = p_list.add_subparsers(dest="what", required=True)

    p_list_cfg = sub_list.add_parser("configs")
    p_list_cfg.set_defaults(handler=cmd_list_configs)

    p_list_runs = sub_list.add_parser("runs")
    p_list_runs.add_argument("--status", default="any", choices=["any", "ok", "failed"])
    p_list_runs.set_defaults(handler=cmd_list_configs)

    p_new = sub.add_parser("new")
    p_new.add_argument("recipe", nargs="?")
    p_new.add_argument("-m", "--message", help="Description of the experiment")
    p_new.add_argument("--force", action="store_true", help="Overwrite existing recipe")
    p_new.add_argument("overrides", nargs=argparse.REMAINDER)
    p_new.set_defaults(handler=cmd_new)

    # try [recipe] -- overrides...
    p_try = sub.add_parser("try")
    p_try.add_argument("recipe", nargs="?")
    p_try.add_argument("overrides", nargs=argparse.REMAINDER)
    p_try.set_defaults(handler=cmd_list_configs)

    # run <recipe|path>
    p_run = sub.add_parser("run")
    p_run.add_argument("target")
    p_run.set_defaults(handler=cmd_list_configs)

    # fork <source> <name> -- overrides...
    p_fork = sub.add_parser("fork")
    p_fork.add_argument("source")
    p_fork.add_argument("name")
    p_fork.add_argument("-m", "--message", help="Description of the experiment")
    p_fork.add_argument("--force", action="store_true", help="Overwrite existing recipe")
    p_fork.add_argument("overrides", nargs=argparse.REMAINDER)
    p_fork.set_defaults(handler=cmd_fork)

    # ... add edit/show/diff/schema/new similarly

    return p


def normalize_overrides(tokens: list[str]) -> list[str]:
    return tokens[1:] if tokens[:1] == ["--"] else tokens


def dispatch(ctx: Ctx, argv: list[str]) -> Any:
    parser = build_parser()
    ns = parser.parse_args(argv)

    # resolve effective dirs once
    if ns.runs_dir:
        ctx.runs_dir = ns.runs_dir.expanduser()
    if ns.configs_dir:
        ctx.configs_dir = ns.configs_dir.expanduser()

    # normalize override lists if this command has them
    if hasattr(ns, "overrides"):
        ns.overrides = normalize_overrides(ns.overrides)

    return ns.handler(ctx, ns)


def cmd_list_configs(ctx: Ctx, ns: argparse.Namespace) -> int:
    if not ctx.configs_dir.exists():
        print(f"(no configs dir) {ctx.configs_dir}", file=sys.stderr)
        return 1

    for p in sorted(ctx.configs_dir.glob("*.y*ml")):
        print(p.stem)
    return 0


if __name__ == "__main__":
    dispatch(
        ctx=Ctx(schema=BaseModel, configs_dir=Path("configs/"), runs_dir=Path("runs/")),
        argv=sys.argv[1:],
    )
    parser = build_parser()
    ns = parser.parse_args(sys.argv[1:])
