# Essential Fileset Summary

This trimmed repository was created from `aim-materials` by keeping:

- Core source code required for model training/inference and DFT orchestration.
- Small, final-stage artifacts needed for reporting and website/MPContrib integration.

## Excluded on purpose

- Raw/full datasets (hundreds of MB scale)
- Virtual environments (`venv`, `.venv`)
- Model checkpoints (`*.pt`)
- Massive generated/intermediate JSON files
- Long-running QE working directories and logs

## Current footprint

- Approximate total size: ~6.6 MB
- File count: 50+ (code + compact outputs)

