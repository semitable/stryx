# CLI Reference

## `try`
Run an experiment with overrides.
- **Auto-saves** to `configs/scratches/`.
- **Arguments**: `[source] [overrides...]`

## `new`
Create a fresh canonical recipe from defaults.
- **Saves** to `configs/`.
- **Arguments**: `[name] [overrides...]`

## `fork`
Branch an existing recipe.
- **Saves** to `configs/`.
- **Arguments**: `<source> [new_name] [overrides...]`

## `run`
Run a recipe exactly as saved.
- **Strict**: No overrides allowed.
- **Arguments**: `<recipe_name>`

## `diff`
Compare two configurations.
- **Arguments**: `<recipe_A> [recipe_B]` (B defaults to schema defaults if omitted).

## `list`
List all recipes and scratches in a table.

## `show`
Print the resolved configuration.

## `edit`
Open the interactive TUI editor.

## `schema`
Print the configuration schema and fields.
