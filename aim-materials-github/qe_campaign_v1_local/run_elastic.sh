#!/usr/bin/env bash
set -uo pipefail
QE_CMD="${1:-pw.x}"
NPROC="${2:-1}"
INPUT_LIST="${3:-}"
QE_TIMEOUT_SEC="${QE_TIMEOUT_SEC:-0}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

PASS_FILE="$PWD/elastic_passed.txt"
FAIL_FILE="$PWD/elastic_failed.txt"
LOG_FILE="$PWD/elastic_stage.log"

if [[ -z "$INPUT_LIST" ]]; then
  if [[ -f "$PWD/scf_passed.txt" ]]; then
    INPUT_LIST="$PWD/scf_passed.txt"
  elif [[ -f "$PWD/relax_passed.txt" ]]; then
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
  d="$PWD/$rel/03_elastic"
  if [[ ! -f "$d/strain_manifest.csv" ]]; then
    echo "$rel,missing_strain_manifest=1" >> "$FAIL_FILE"
    echo "[FAIL] $rel missing strain_manifest.csv" | tee -a "$LOG_FILE"
    continue
  fi

  cand_fail=0
  cand_total=0
  cand_done=0
  echo "[RUN ] $rel" | tee -a "$LOG_FILE"

  while IFS=, read -r sid e1 e2 e3 e4 e5 e6 inp outp; do
    [[ "$sid" == "strain_id" ]] && continue
    dd="$d/strain_${sid}"
    cand_total=$((cand_total + 1))
    (
      cd "$dd" || exit 10
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
    if [[ $rc -eq 124 ]]; then
      cand_fail=1
      echo "$rel,strain_id=$sid,exit_code=$rc,timeout_sec=$QE_TIMEOUT_SEC" >> "$FAIL_FILE"
      echo "[FAIL] $rel strain_id=$sid timeout_sec=$QE_TIMEOUT_SEC" | tee -a "$LOG_FILE"
    elif [[ $rc -ne 0 ]] || [[ ! -f "$dd/qe_scf.out" ]] || ! grep -q "JOB DONE" "$dd/qe_scf.out"; then
      cand_fail=1
      echo "$rel,strain_id=$sid,exit_code=$rc" >> "$FAIL_FILE"
      echo "[FAIL] $rel strain_id=$sid exit_code=$rc" | tee -a "$LOG_FILE"
    else
      cand_done=$((cand_done + 1))
    fi
  done < "$d/strain_manifest.csv"

  if [[ $cand_fail -eq 0 ]]; then
    echo "$rel" >> "$PASS_FILE"
    echo "[PASS] $rel strains=$cand_done/$cand_total" | tee -a "$LOG_FILE"
  else
    echo "[FAIL] $rel strains=$cand_done/$cand_total" | tee -a "$LOG_FILE"
  fi
done < "$INPUT_LIST"

pass_n=$(wc -l < "$PASS_FILE")
fail_n=$(wc -l < "$FAIL_FILE")
echo "[DONE] elastic complete passed=$pass_n failed=$fail_n" | tee -a "$LOG_FILE"
