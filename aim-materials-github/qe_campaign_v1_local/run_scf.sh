#!/usr/bin/env bash
set -uo pipefail
QE_CMD="${1:-pw.x}"
NPROC="${2:-1}"
INPUT_LIST="${3:-}"
QE_TIMEOUT_SEC="${QE_TIMEOUT_SEC:-0}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

PASS_FILE="$PWD/scf_passed.txt"
FAIL_FILE="$PWD/scf_failed.txt"
LOG_FILE="$PWD/scf_stage.log"

if [[ -z "$INPUT_LIST" ]]; then
  if [[ -f "$PWD/relax_passed.txt" ]]; then
    INPUT_LIST="$PWD/relax_passed.txt"
  else
    INPUT_LIST="$PWD/candidate_paths.txt"
  fi
fi

: > "$PASS_FILE"
: > "$FAIL_FILE"
: > "$LOG_FILE"

echo "[INFO] QE_CMD=$QE_CMD NPROC=$NPROC input=$INPUT_LIST timeout_sec=$QE_TIMEOUT_SEC" | tee -a "$LOG_FILE"
if [[ ! -f "$INPUT_LIST" ]]; then
  echo "[ERROR] input list not found: $INPUT_LIST" | tee -a "$LOG_FILE"
  exit 2
fi

while IFS= read -r rel; do
  rel="${rel%$'\r'}"
  [[ -z "${rel}" ]] && continue
  d="$PWD/$rel"
  echo "[RUN ] $rel" | tee -a "$LOG_FILE"
  (
    cd "$d/02_scf" || exit 10
    mkdir -p tmp
    rm -f qe_scf.out
    if [[ "$NPROC" -gt 1 ]]; then
      if [[ "$QE_TIMEOUT_SEC" -gt 0 ]]; then
        timeout "$QE_TIMEOUT_SEC" mpirun -np "$NPROC" "$QE_CMD" -in qe_scf.in > qe_scf.out 2>&1
      else
        mpirun -np "$NPROC" "$QE_CMD" -in qe_scf.in > qe_scf.out 2>&1
      fi
    else
      if [[ "$QE_TIMEOUT_SEC" -gt 0 ]]; then
        timeout "$QE_TIMEOUT_SEC" "$QE_CMD" -in qe_scf.in > qe_scf.out 2>&1
      else
        "$QE_CMD" -in qe_scf.in > qe_scf.out 2>&1
      fi
    fi
  )
  rc=$?
  if [[ $rc -ne 0 ]]; then
    if [[ $rc -eq 124 ]]; then
      echo "$rel,exit_code=$rc,timeout_sec=$QE_TIMEOUT_SEC" >> "$FAIL_FILE"
      echo "[FAIL] $rel timeout_sec=$QE_TIMEOUT_SEC" | tee -a "$LOG_FILE"
      continue
    fi
    echo "$rel,exit_code=$rc" >> "$FAIL_FILE"
    echo "[FAIL] $rel exit_code=$rc" | tee -a "$LOG_FILE"
    continue
  fi
  if ! grep -q "JOB DONE" "$d/02_scf/qe_scf.out"; then
    echo "$rel,exit_code=0,no_job_done=1" >> "$FAIL_FILE"
    echo "[FAIL] $rel no JOB DONE marker" | tee -a "$LOG_FILE"
    continue
  fi
  echo "$rel" >> "$PASS_FILE"
  echo "[PASS] $rel" | tee -a "$LOG_FILE"
done < "$INPUT_LIST"

pass_n=$(wc -l < "$PASS_FILE")
fail_n=$(wc -l < "$FAIL_FILE")
echo "[DONE] scf complete passed=$pass_n failed=$fail_n" | tee -a "$LOG_FILE"
