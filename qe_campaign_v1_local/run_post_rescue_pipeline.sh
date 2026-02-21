#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${1:-$SCRIPT_DIR}"
QE_BIN="${2:-${QE_BIN:-pw.x}}"
NPROC="${3:-1}"
QE_SCF_TIMEOUT_SEC="${QE_SCF_TIMEOUT_SEC:-21600}"
QE_ELASTIC_TIMEOUT_SEC="${QE_ELASTIC_TIMEOUT_SEC:-21600}"

cd "$ROOT"
ts="$(date +%Y%m%d_%H%M%S)"

touch relax_passed.txt relax_passed_rescue_nonconv.txt scf_passed.txt elastic_passed.txt

cat relax_passed.txt relax_passed_rescue_nonconv.txt 2>/dev/null | sed '/^[[:space:]]*$/d' | sort -u > relax_passed_merged_latest.txt
relax_merged_n="$(wc -l < relax_passed_merged_latest.txt)"
echo "[INFO] relax_merged_count=$relax_merged_n"

comm -23 <(sort -u relax_passed_merged_latest.txt) <(sort -u scf_passed.txt) > scf_pending_post_rescue.txt
scf_pending_n="$(wc -l < scf_pending_post_rescue.txt)"
echo "[INFO] scf_pending_count=$scf_pending_n"

if [[ "$scf_pending_n" -gt 0 ]]; then
  cp scf_passed.txt "scf_passed.before_post_rescue_${ts}.txt" 2>/dev/null || true
  cp scf_failed.txt "scf_failed.before_post_rescue_${ts}.txt" 2>/dev/null || true
  cp scf_stage.log "scf_stage.before_post_rescue_${ts}.log" 2>/dev/null || true

  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
  export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
  export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
  export QE_TIMEOUT_SEC="$QE_SCF_TIMEOUT_SEC"

  echo "[STEP] run_scf on pending"
  bash run_scf.sh "$QE_BIN" "$NPROC" scf_pending_post_rescue.txt

  cat "scf_passed.before_post_rescue_${ts}.txt" scf_passed.txt 2>/dev/null | sed '/^[[:space:]]*$/d' | sort -u > scf_passed_merged_latest.txt
else
  cp scf_passed.txt scf_passed_merged_latest.txt
fi

comm -23 <(sort -u scf_passed_merged_latest.txt) <(sort -u elastic_passed.txt) > elastic_pending_post_rescue.txt
elastic_pending_n="$(wc -l < elastic_pending_post_rescue.txt)"
echo "[INFO] elastic_pending_count=$elastic_pending_n"

if [[ "$elastic_pending_n" -gt 0 ]]; then
  cp elastic_passed.txt "elastic_passed.before_post_rescue_${ts}.txt" 2>/dev/null || true
  cp elastic_failed.txt "elastic_failed.before_post_rescue_${ts}.txt" 2>/dev/null || true
  cp elastic_stage.log "elastic_stage.before_post_rescue_${ts}.log" 2>/dev/null || true

  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
  export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
  export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
  export QE_TIMEOUT_SEC="$QE_ELASTIC_TIMEOUT_SEC"

  echo "[STEP] run_elastic on pending"
  bash run_elastic.sh "$QE_BIN" "$NPROC" elastic_pending_post_rescue.txt

  cat "elastic_passed.before_post_rescue_${ts}.txt" elastic_passed.txt 2>/dev/null | sed '/^[[:space:]]*$/d' | sort -u > elastic_passed_merged_latest.txt
else
  cp elastic_passed.txt elastic_passed_merged_latest.txt
fi

echo "[DONE] post-rescue pipeline complete"
echo "[INFO] relax_merged=$(wc -l < relax_passed_merged_latest.txt) scf_merged=$(wc -l < scf_passed_merged_latest.txt) elastic_merged=$(wc -l < elastic_passed_merged_latest.txt)"
