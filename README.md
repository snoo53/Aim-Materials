# AIM Materials (Essential Repo)

Minimal, publication-facing subset of the AIM Materials workspace.

This repository contains only the core code and compact result artifacts needed to:

- train/infer candidate materials,
- prepare and analyze DFT campaigns (QE),
- export MP-like outputs for website/MPContrib workflows.

Large raw datasets, virtual environments, model checkpoints, and bulky intermediate outputs are intentionally excluded.

## Included

- Core model code: `aim_models/`, `utils/`
- Main pipeline scripts:
  - `train.py`
  - `generate_structures.py`
  - `relax_candidates_chgnet.py`
  - `infer_properties_ensemble.py`
  - `prepare_dft_shortlist.py`
  - `build_qe_campaign.py`
  - `analyze_qe_campaign_results.py`
  - `run_publishable_pipeline.py`
- Essential docs: `docs/`
- QE campaign essentials and compact results: `qe_campaign_v1_local/`
  - run scripts (`run_relax.sh`, `run_scf.sh`, `run_elastic.sh`, etc.)
  - parsed summaries (`qe_results_parsed.csv`, `qe_results_summary.json`)
  - website/MP-like data (`mp_combined_data_candidates_mixed51_with_qe_status.json`)

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Typical Workflow

1. Train / update model: `python train.py ...`
2. Generate candidates: `python generate_structures.py ...`
3. Pre-relax/screen: `python relax_candidates_chgnet.py ...`
4. Build QE campaign: `python build_qe_campaign.py ...`
5. Run QE stages in `qe_campaign_v1_local/`:
   - `bash run_relax.sh ...`
   - `bash run_scf.sh ...`
   - `bash run_elastic.sh ...`
6. Analyze results:
   - `python analyze_qe_campaign_results.py ...`

## Notes

- This repo is intentionally compact and reproducible-focused.
- If full reproducibility from raw MP dumps is required, use the original full workspace (not this trimmed repo).
- QE execution assumes Linux/WSL environment.

