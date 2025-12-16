# Repository Guidelines

## Project Structure & Module Organization
- Core package code lives in `src/stryx`. Key modules: `decorator.py` (CLI decorator and command parsing), `config.py` (config manager + schema introspection), `schema.py` (Pydantic type utilities), `utils.py` (YAML I/O and nested path helpers), and `tui.py` for interactive editing. The `cli.py` entrypoint backs the installed `stryx` script.
- Tests sit in `tests/` with fast unit tests at the root and broader CLI flows in `tests/integration/`. Keep new fixtures local to the suite unless shared widely.
- Examples for experimentation are in `examples/`; `train.py` and `sft.py` assume the `examples` extra is installed.

## Build, Test, and Development Commands
- Run all tests: `uv run pytest`. Target individual tests with `uv run pytest tests/test_cli.py::TestCLIDecorator::test_run_with_defaults`.
- Run examples: `uv run examples/train.py lr=1e-3` (or `uv run examples/sft.py` if extras are synced).
- Dependency changes: use `uv add <pkg>` then `uv lock` and `uv sync`. Avoid manual `pip install`; rely on the lockfile.
- Lint/formatting: no enforced tool yet; prefer `ruff`/`black` defaults if you introduce them and keep diffs minimal.
- Package build: `uv build` mirrors the production build backend.

## Coding Style & Naming Conventions
- Python 3.11+, typed throughout; keep function signatures annotated and return concrete types instead of `Any` where possible.
- Use 4-space indentation, module-level docstrings, and short, imperative helper names (e.g., `_build_config`, `_parse_argv`). Modules and files are snake_case.
- Handle user-facing errors with clear `SystemExit` messages; avoid bare exceptions in CLI paths.
- YAML I/O goes through `read_yaml`/`write_yaml` for atomic writes; prefer `Path` objects over raw strings.

## Testing Guidelines
- Tests use `pytest`. Follow the existing naming: classes prefixed with `Test`, test methods as `test_*`, and docstrings describing behavior.
- Favor temporary directories (`tmp_path`/`TemporaryDirectory`) for filesystem work to keep runs hermetic.
- Add regression tests alongside the module they cover; integration-style CLI behaviors belong under `tests/integration/`.

## Commit & Pull Request Guidelines
- Write commit subjects in imperative mood (`Add config defaults`), ~50 chars if possible; include a brief body when behavior changes or migrations occur.
- Keep PRs scoped; describe motivation, approach, and testing performed. Link issues when applicable and note any user-facing CLI changes. Include screenshots only when touching TUI output.
