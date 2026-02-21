#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-$PWD}"
QE_BIN="${2:-${QE_BIN:-pw.x}}"
NPROC="${3:-1}"
FAILED_FILE="${4:-$ROOT/relax_failed.txt}"
INPUT_LIST="${5:-$ROOT/candidate_paths_relax_nonconverged_retry.txt}"
STAGE_TAG="${RELAX_STAGE_TAG:-rescue_nonconv}"

cd "$ROOT"

if [[ ! -f "$FAILED_FILE" ]]; then
  echo "[ERROR] failed list not found: $FAILED_FILE"
  exit 2
fi

awk -F',' 'NF > 0 {print $1}' "$FAILED_FILE" | sed '/^[[:space:]]*$/d' | sort -u > "$INPUT_LIST"

tag_suffix="$STAGE_TAG"
if [[ -n "$tag_suffix" ]] && [[ "$tag_suffix" != _* ]]; then
  tag_suffix="_$tag_suffix"
fi
PASS_HIST="$ROOT/relax_passed${tag_suffix}.txt"
FAIL_HIST="$ROOT/relax_failed${tag_suffix}.txt"

tmp_skip="$(mktemp)"
if [[ -f "$PASS_HIST" ]]; then
  cat "$PASS_HIST" >> "$tmp_skip"
fi
if [[ -f "$FAIL_HIST" ]]; then
  awk -F',' 'NF > 0 {print $1}' "$FAIL_HIST" >> "$tmp_skip"
fi
if [[ -s "$tmp_skip" ]]; then
  sort -u "$tmp_skip" -o "$tmp_skip"
  grep -Fvxf "$tmp_skip" "$INPUT_LIST" > "${INPUT_LIST}.tmp" || true
  mv "${INPUT_LIST}.tmp" "$INPUT_LIST"
fi
rm -f "$tmp_skip"

retry_n=$(wc -l < "$INPUT_LIST")
if [[ "$retry_n" -eq 0 ]]; then
  echo "[INFO] no pending non-converged candidates found in $FAILED_FILE after removing already-attempted entries"
  exit 0
fi

export RELAX_REQUIRE_CONVERGENCE=1
export RELAX_ENABLE_RESCUE=1
export QE_TIMEOUT_SEC="${QE_TIMEOUT_SEC:-43200}"
export QE_RESCUE_TIMEOUT_SEC="${QE_RESCUE_TIMEOUT_SEC:-21600}"
export RELAX_RESCUE_NSTEP_ADD="${RELAX_RESCUE_NSTEP_ADD:-180}"
export RELAX_RESCUE_ELECTRON_MAXSTEP="${RELAX_RESCUE_ELECTRON_MAXSTEP:-300}"
export RELAX_RESCUE_MIXING_BETA="${RELAX_RESCUE_MIXING_BETA:-0.20}"
export RELAX_STAGE_TAG="$STAGE_TAG"
export RELAX_RESCUE_ONLY="${RELAX_RESCUE_ONLY:-1}"
export RELAX_APPEND_STAGE="${RELAX_APPEND_STAGE:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

echo "[INFO] retry_candidates=$retry_n input_list=$INPUT_LIST"
echo "[INFO] QE_BIN=$QE_BIN NPROC=$NPROC timeout=$QE_TIMEOUT_SEC rescue_timeout=$QE_RESCUE_TIMEOUT_SEC"
echo "[INFO] RELAX_STAGE_TAG=$RELAX_STAGE_TAG rescue_only=$RELAX_RESCUE_ONLY"

bash run_relax.sh "$QE_BIN" "$NPROC" "$INPUT_LIST"
