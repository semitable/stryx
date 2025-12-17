# Best Practices

## Organize your recipes
- Use descriptive names for canonical recipes (`baseline_resnet50`, `ablation_dropout`).
- Use `try` for everything else. Let Stryx manage the mess in `scratches/`.

## Commit your recipes
- Commit `configs/*.yaml`.
- Ignore `configs/scratches/` (add to `.gitignore`).
- Ignore `runs/` (add to `.gitignore`).

## Clean up
- Periodically delete `configs/scratches/` or `runs/` that failed.
- (Future: `stryx clean` command).
