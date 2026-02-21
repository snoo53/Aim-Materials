# Public Release Notes

This repository was audited for public release hygiene.

## What was checked

- No embedded API keys, tokens, or passwords in tracked files.
- No hardcoded personal filesystem paths in released scripts.
- No user-specific QE binary paths in released scripts.
- Shell script syntax check (`bash -n`) for all QE runner scripts.
- Python syntax check (`py_compile`) for core analysis/build scripts.

## Changes made in this release pass

- Converted QE orchestration scripts to portable defaults:
  - `qe_campaign_v1_local/check_qe_status.sh`
  - `qe_campaign_v1_local/run_final_analysis_after_pipeline.sh`
  - `qe_campaign_v1_local/run_followup_strict.sh`
  - `qe_campaign_v1_local/run_post_rescue_pipeline.sh`
  - `qe_campaign_v1_local/run_relax_rescue_nonconverged.sh`
- Removed hardcoded local paths (e.g., `/mnt/c/Users/...`, `/home/...`).
- Switched defaults to:
  - script directory for campaign root
  - parent directory for repository root
  - `pw.x` (or `QE_BIN` env/arg) for QE executable
  - `python3` (or `PYTHON_BIN` env) for analysis scripts

## Operator notes

- For production runs, explicitly pass `ROOT`, `QE_BIN`, and `NPROC` arguments.
- Keep credentials in environment variables only (never commit them).

