# Execution Tracking

Stryx automatically captures the lifecycle of every run.

## Artifacts
Every run creates a folder in `runs/<run_id>/`:

- **`manifest.yaml`**: Metadata, status, and result.
- **`config.yaml`**: The exact resolved configuration.
- **`stdout.log`**: Captured stdout/stderr (Rank 0).
- **`git.patch`**: Git diff of the working tree (if dirty).

## Status Tracking
The `manifest.yaml` updates in real-time:
1.  **RUNNING**: When the job starts.
2.  **COMPLETED**: When `main()` returns successfully. Includes the return value in `result`.
3.  **FAILED**: If an exception occurs. Includes the full traceback.

## Logs
Output is automatically tee-d to `stdout.log` (for single process) or `logs/rank_<N>.log` (for distributed).
