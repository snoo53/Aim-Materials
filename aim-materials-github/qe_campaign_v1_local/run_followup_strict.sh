#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${1:-$SCRIPT_DIR}"
REPO="${2:-$(cd "$ROOT/.." && pwd)}"
QE_BIN="${3:-${QE_BIN:-pw.x}}"
NPROC="${4:-1}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

cd "$ROOT"
ts=$(date +%Y%m%d_%H%M%S)

echo "[INFO] followup started ts=$ts"
for f in elastic_passed.txt elastic_failed.txt elastic_stage.log; do
  [ -f "$f" ] && cp "$f" "${f}.before_longrerun_${ts}"
done

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export QE_TIMEOUT_SEC=21600

echo "[STEP] long-timeout elastic rerun"
bash run_elastic.sh "$QE_BIN" "$NPROC" scf_passed.txt

cd "$REPO"
echo "[STEP] analyze strict latest"
"$PYTHON_BIN" analyze_qe_campaign_results.py \
  --campaign_manifest qe_campaign_v1_local/campaign_manifest.csv \
  --out_csv qe_campaign_v1_local/analysis_strict_latest.csv \
  --out_summary_json qe_campaign_v1_local/analysis_strict_latest_summary.json \
  --out_validated_csv qe_campaign_v1_local/analysis_strict_latest_validated.csv \
  --norm_stats_npz normalization_stats_fixed.npz \
  --pred_voigt_is_normalized

cd "$ROOT"
for f in relax_passed.txt relax_failed.txt relax_stage.log; do
  [ -f "$f" ] && cp "$f" "${f}.before_nextbatch_${ts}"
done

export RELAX_REQUIRE_CONVERGENCE=1
export QE_TIMEOUT_SEC=43200
export RELAX_ENABLE_RESCUE=1
export QE_RESCUE_TIMEOUT_SEC=21600
export RELAX_RESCUE_NSTEP_ADD=180
export RELAX_RESCUE_ELECTRON_MAXSTEP=300
export RELAX_RESCUE_MIXING_BETA=0.20

echo "[STEP] next strict relax batch"
bash run_relax.sh "$QE_BIN" "$NPROC" candidate_paths_next_strict_batch.txt

cat "relax_failed.txt.before_nextbatch_${ts}" relax_failed.txt 2>/dev/null | sed '/^$/d' > relax_failed_merged_latest.txt || true
cat "relax_passed.txt.before_nextbatch_${ts}" relax_passed.txt 2>/dev/null | sed '/^$/d' | sort -u > relax_passed_merged_latest.txt || true

echo "[DONE] follow-up strict workflow complete"
