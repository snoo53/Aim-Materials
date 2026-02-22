# Contributing

Thanks for contributing to Aim Materials.

## Scope

This repository is a curated, compact release of the larger project. Keep contributions focused on:

- correctness and reproducibility of pipelines,
- portability of scripts (especially QE workflow scripts),
- clarity of data exports and documentation.

## Development Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Before Opening a Pull Request

1. Run Python syntax checks:

```bash
python -m compileall -q .
```

2. Run shell syntax checks:

```bash
bash -n qe_campaign_v1_local/*.sh
```

3. Update docs when behavior changes:
- `README.md`
- `PUBLIC_RELEASE_NOTES.md`
- `docs/publishable_pipeline.md` (if workflow changes)

## Pull Request Guidance

- Keep PRs small and single-purpose.
- Include rationale and expected behavior changes.
- Do not commit secrets, machine-specific absolute paths, or large local artifacts.
