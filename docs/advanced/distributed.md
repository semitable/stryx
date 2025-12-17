# Distributed Training

Stryx enforces safety for distributed runs (DDP, Slurm, etc.).

## The Shared ID Requirement
In a distributed setting, multiple processes (ranks) run the same script. To prevent them from writing to different folders (e.g., `runs/petname-1` vs `runs/petname-2`), Stryx requires a **shared Run ID**.

If Stryx detects a distributed environment (`RANK` is set) but cannot find a trusted shared ID, it will **error out**.

## Solutions

### 1. Use `torchrun` (Recommended)
`torchrun` sets `TORCHELASTIC_RUN_ID` (or you can set `rdzv_id`), which Stryx uses (sometimes). 
*Note: Standalone torchrun often sets ID to "none". Stryx considers "none" unsafe.*

### 2. Manual ID (Robust)
Explicitly generate and pass an ID.

```bash
export STRYX_RUN_ID=$(stryx create-run-id)
torchrun --nproc_per_node=8 train.py try
```

### 3. Slurm
Stryx automatically trusts `SLURM_JOB_ID` as a shared ID.
