# Publishable Pipeline (Strict)

This pipeline is designed to increase structural/mechanical plausibility before candidate reporting.

## 1) Train with stronger tensor constraints

```bash
python train.py --data processed_data_filtered/all_materials_data_normalized.json --checkpoint aim_publishable.pt --loss_profile strict_tensor --hard_symmetry_mask
```

## 2) Run strict 2/3/4-element generation + validation

```bash
python run_publishable_pipeline.py --ckpt aim_publishable.pt --sets 2,3,4 --n_per_set 300 --tag publishable_v1
```

Outputs per set:
- `generated_materials_<N>el_<tag>.json`
- `generated_cifs_<N>el_<tag>/`
- `generated_materials_<N>el_<tag>_with_predictions_real.json`
- `validation_<N>el_<tag>_deep_summary.json`
- `candidates_<N>el_<tag>_strict_novel_unique.csv`

Aggregate summary:
- `validation_<tag>_aggregate_summary.json`

## 3) Optional uncertainty with checkpoint ensemble

```bash
python infer_properties_ensemble.py --data generated_materials_2el_publishable_v1_featurized_real.json --ckpts aim_publishable.pt aim_spd_tuned_voigt_sym40.pt --out_csv predictions/ensemble_2el_publishable_v1.csv
```

## 4) CHGNet screen on shortlisted candidates

```bash
python screen_shortlist_chgnet.py \
  --manifest_csv dft_shortlist_v3_geomfix/shortlist_manifest_with_uncertainty.csv \
  --out_csv dft_shortlist_v3_geomfix/shortlist_chgnet_screen.csv \
  --out_summary_json dft_shortlist_v3_geomfix/shortlist_chgnet_screen_summary.json \
  --force_max 0.25 --stress_fro_max 50
```

## 5) Build VASP campaign package

```bash
python build_vasp_campaign.py \
  --manifest_csv dft_shortlist_v3_geomfix/shortlist_chgnet_screen.csv \
  --out_dir dft_campaign_v1 \
  --top_per_set 24 \
  --max_per_formula 2 \
  --max_scalar_std 0.05 \
  --max_voigt_std 0.05 \
  --max_force 0.25
```

Generated files include per-candidate POSCAR/CIF, INCAR templates (relax/static/elastic),
KPOINTS templates, POTCAR.spec, metadata, and runner scripts for Windows/Linux.

## 6) Analyze completed DFT outputs (elastic agreement + optional hull)

```bash
python analyze_dft_campaign_results.py \
  --campaign_manifest dft_campaign_v1_strict/campaign_manifest.csv \
  --out_csv dft_campaign_v1_strict/dft_results_parsed.csv \
  --out_summary_json dft_campaign_v1_strict/dft_results_summary.json \
  --out_validated_csv dft_campaign_v1_strict/dft_validated_top.csv \
  --force_tol 0.05 \
  --eig_tol 1e-6
```

Optional MP hull query (requires `MP_API_KEY`):

```bash
python analyze_dft_campaign_results.py \
  --campaign_manifest dft_campaign_v1_strict/campaign_manifest.csv \
  --out_csv dft_campaign_v1_strict/dft_results_parsed.csv \
  --out_summary_json dft_campaign_v1_strict/dft_results_summary.json \
  --query_mp_hull \
  --mp_api_key_env MP_API_KEY \
  --hull_tol 0.1
```

## 7) Prepare SLURM handoff bundles (cluster submission)

```bash
python prepare_slurm_campaign.py \
  --campaign_dir dft_campaign_v1_strict \
  --n_tasks 32 \
  --time_relax 24:00:00 \
  --time_static 08:00:00 \
  --time_elastic 12:00:00 \
  --module_cmds "module purge\nmodule load vasp" \
  --vasp_cmd vasp_std \
  --make_zip
```

This generates `submit_*_array.sbatch`, `submit_all.sh`, and a zip archive
(`*_handoff.zip`) for transfer to your cluster.

## 8) Local/Open-source DFT route (no VASP/HPC required): QE campaign

Build a QE-ready campaign directly from the screened shortlist:

```bash
python build_qe_campaign.py \
  --manifest_csv dft_shortlist_v3_geomfix/shortlist_chgnet_screen.csv \
  --out_dir qe_campaign_v1_local \
  --top_per_set 18 \
  --max_per_formula 2 \
  --max_scalar_std 0.05 \
  --max_voigt_std 0.05 \
  --max_force 0.25 \
  --strain_amp 0.005 \
  --ecutwfc 80 \
  --ecutrho 640
```

Generated files include per-candidate `qe_relax.in`, `qe_scf.in`, finite-strain
elastic inputs (`03_elastic/strain_*/qe_scf.in`), and runners:
`run_relax(.sh/.ps1)`, `run_scf(.sh/.ps1)`, `run_elastic(.sh/.ps1)`, `run_all(.sh/.ps1)`.

## 9) Run QE calculations locally

Linux/macOS:

```bash
cd qe_campaign_v1_local
bash run_relax.sh pw.x 4
bash run_scf.sh pw.x 4
bash run_elastic.sh pw.x 4
```

PowerShell:

```powershell
cd qe_campaign_v1_local
.\run_relax.ps1 -QeExe pw.x -NProc 4
.\run_scf.ps1 -QeExe pw.x -NProc 4
.\run_elastic.ps1 -QeExe pw.x -NProc 4
```

## 10) Analyze QE outputs for tensor consistency + shortlist ranking

```bash
python analyze_qe_campaign_results.py \
  --campaign_manifest qe_campaign_v1_local/campaign_manifest.csv \
  --out_csv qe_campaign_v1_local/qe_results_parsed.csv \
  --out_summary_json qe_campaign_v1_local/qe_results_summary.json \
  --out_validated_csv qe_campaign_v1_local/qe_validated_top.csv \
  --eig_tol 1e-6 \
  --fit_rms_tol 5.0 \
  --qe_stress_sign -1
```

Optional MP hull query (requires `MP_API_KEY`):

```bash
python analyze_qe_campaign_results.py \
  --campaign_manifest qe_campaign_v1_local/campaign_manifest.csv \
  --out_csv qe_campaign_v1_local/qe_results_parsed.csv \
  --out_summary_json qe_campaign_v1_local/qe_results_summary.json \
  --query_mp_hull \
  --mp_api_key_env MP_API_KEY \
  --hull_tol 0.1
```
