#!/usr/bin/env bash
set -uo pipefail
QE_CMD="${1:-pw.x}"
NPROC="${2:-1}"
INPUT_LIST="${3:-$PWD/candidate_paths.txt}"
QE_TIMEOUT_SEC="${QE_TIMEOUT_SEC:-0}"
RELAX_REQUIRE_CONVERGENCE="${RELAX_REQUIRE_CONVERGENCE:-1}"
CAMPAIGN_ROOT="$PWD"
RELAX_STAGE_TAG="${RELAX_STAGE_TAG:-}"

# Optional rescue retry for non-converged relax (keeps strict gate on by default).
RELAX_ENABLE_RESCUE="${RELAX_ENABLE_RESCUE:-1}"
QE_RESCUE_TIMEOUT_SEC="${QE_RESCUE_TIMEOUT_SEC:-$QE_TIMEOUT_SEC}"
RELAX_RESCUE_NSTEP_ADD="${RELAX_RESCUE_NSTEP_ADD:-180}"
RELAX_RESCUE_ELECTRON_MAXSTEP="${RELAX_RESCUE_ELECTRON_MAXSTEP:-300}"
RELAX_RESCUE_MIXING_BETA="${RELAX_RESCUE_MIXING_BETA:-0.20}"
RELAX_RESCUE_SCRIPT="${RELAX_RESCUE_SCRIPT:-$CAMPAIGN_ROOT/prepare_relax_retry_input.py}"
RELAX_RESCUE_ONLY="${RELAX_RESCUE_ONLY:-0}"
RELAX_APPEND_STAGE="${RELAX_APPEND_STAGE:-0}"

# Keep memory predictable on WSL/desktop runs.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

if [[ -n "$RELAX_STAGE_TAG" ]] && [[ "$RELAX_STAGE_TAG" != _* ]]; then
  RELAX_STAGE_TAG="_$RELAX_STAGE_TAG"
fi

PASS_FILE="$PWD/relax_passed${RELAX_STAGE_TAG}.txt"
FAIL_FILE="$PWD/relax_failed${RELAX_STAGE_TAG}.txt"
LOG_FILE="$PWD/relax_stage${RELAX_STAGE_TAG}.log"

if [[ "$RELAX_APPEND_STAGE" -eq 1 ]]; then
  touch "$PASS_FILE" "$FAIL_FILE" "$LOG_FILE"
else
  : > "$PASS_FILE"
  : > "$FAIL_FILE"
  : > "$LOG_FILE"
fi

if [[ ! -f "$INPUT_LIST" ]]; then
  echo "[ERROR] input list not found: $INPUT_LIST" | tee -a "$LOG_FILE"
  exit 2
fi

echo "[INFO] QE_CMD=$QE_CMD NPROC=$NPROC input=$INPUT_LIST timeout_sec=$QE_TIMEOUT_SEC" | tee -a "$LOG_FILE"
echo "[INFO] OMP_NUM_THREADS=$OMP_NUM_THREADS OPENBLAS_NUM_THREADS=$OPENBLAS_NUM_THREADS MKL_NUM_THREADS=$MKL_NUM_THREADS" | tee -a "$LOG_FILE"
echo "[INFO] RELAX_REQUIRE_CONVERGENCE=$RELAX_REQUIRE_CONVERGENCE" | tee -a "$LOG_FILE"
echo "[INFO] RELAX_ENABLE_RESCUE=$RELAX_ENABLE_RESCUE rescue_timeout_sec=$QE_RESCUE_TIMEOUT_SEC rescue_nstep_add=$RELAX_RESCUE_NSTEP_ADD rescue_electron_maxstep=$RELAX_RESCUE_ELECTRON_MAXSTEP rescue_mixing_beta=$RELAX_RESCUE_MIXING_BETA" | tee -a "$LOG_FILE"
echo "[INFO] RELAX_RESCUE_ONLY=$RELAX_RESCUE_ONLY" | tee -a "$LOG_FILE"
echo "[INFO] RELAX_APPEND_STAGE=$RELAX_APPEND_STAGE" | tee -a "$LOG_FILE"
echo "[INFO] RELAX_STAGE_TAG=${RELAX_STAGE_TAG:-<default>}" | tee -a "$LOG_FILE"

run_relax_once() {
  local run_dir="$1"
  local in_file="$2"
  local out_file="$3"
  (
    cd "$run_dir" || exit 10
    mkdir -p tmp
    rm -f "$out_file"
    if [[ "$NPROC" -gt 1 ]]; then
      if [[ "$QE_TIMEOUT_SEC" -gt 0 ]]; then
        timeout "$QE_TIMEOUT_SEC" mpirun -np "$NPROC" "$QE_CMD" -in "$in_file" > "$out_file" 2>&1
      else
        mpirun -np "$NPROC" "$QE_CMD" -in "$in_file" > "$out_file" 2>&1
      fi
    else
      if [[ "$QE_TIMEOUT_SEC" -gt 0 ]]; then
        timeout "$QE_TIMEOUT_SEC" "$QE_CMD" -in "$in_file" > "$out_file" 2>&1
      else
        "$QE_CMD" -in "$in_file" > "$out_file" 2>&1
      fi
    fi
  )
}

run_relax_rescue_once() {
  local run_dir="$1"
  local in_file="$2"
  local out_file="$3"
  (
    cd "$run_dir" || exit 10
    mkdir -p tmp
    rm -f "$out_file"
    if [[ "$NPROC" -gt 1 ]]; then
      if [[ "$QE_RESCUE_TIMEOUT_SEC" -gt 0 ]]; then
        timeout "$QE_RESCUE_TIMEOUT_SEC" mpirun -np "$NPROC" "$QE_CMD" -in "$in_file" > "$out_file" 2>&1
      else
        mpirun -np "$NPROC" "$QE_CMD" -in "$in_file" > "$out_file" 2>&1
      fi
    else
      if [[ "$QE_RESCUE_TIMEOUT_SEC" -gt 0 ]]; then
        timeout "$QE_RESCUE_TIMEOUT_SEC" "$QE_CMD" -in "$in_file" > "$out_file" 2>&1
      else
        "$QE_CMD" -in "$in_file" > "$out_file" 2>&1
      fi
    fi
  )
}

while IFS= read -r rel; do
  rel="${rel%$'\r'}"
  [[ -z "${rel}" ]] && continue
  d="$PWD/$rel"
  echo "[RUN ] $rel" | tee -a "$LOG_FILE"

  if [[ "$RELAX_RESCUE_ONLY" -eq 1 ]]; then
    if [[ "$RELAX_ENABLE_RESCUE" -ne 1 ]] || [[ ! -f "$RELAX_RESCUE_SCRIPT" ]]; then
      echo "$rel,exit_code=0,rescue_only_requested_but_unavailable=1" >> "$FAIL_FILE"
      echo "[FAIL] $rel rescue-only requested but rescue script unavailable" | tee -a "$LOG_FILE"
      continue
    fi

    echo "[INFO] $rel rescue-only: building qe_relax_retry.in from existing qe_relax.out/qe_relax.in" | tee -a "$LOG_FILE"
    (
      cd "$d/01_relax" || exit 10
      python3 "$RELAX_RESCUE_SCRIPT" \
        --infile qe_relax.in \
        --outfile qe_relax_retry.in \
        --outlog qe_relax.out \
        --nstep_add "$RELAX_RESCUE_NSTEP_ADD" \
        --electron_maxstep "$RELAX_RESCUE_ELECTRON_MAXSTEP" \
        --mixing_beta "$RELAX_RESCUE_MIXING_BETA" \
        > qe_relax_retry_prep.log 2>&1
    )
    prep_rc=$?
    if [[ $prep_rc -ne 0 ]]; then
      echo "$rel,exit_code=0,rescue_prepare_failed=1,rescue_only=1" >> "$FAIL_FILE"
      echo "[FAIL] $rel rescue-only prepare failed" | tee -a "$LOG_FILE"
      continue
    fi

    run_relax_rescue_once "$d/01_relax" "qe_relax_retry.in" "qe_relax_retry.out"
    rc_retry=$?
    if [[ $rc_retry -eq 0 ]] && grep -q "JOB DONE" "$d/01_relax/qe_relax_retry.out" && ! grep -q "convergence NOT achieved" "$d/01_relax/qe_relax_retry.out"; then
      echo "$rel" >> "$PASS_FILE"
      echo "[PASS] $rel rescue_only_converged=1" | tee -a "$LOG_FILE"
    elif [[ $rc_retry -eq 124 ]]; then
      echo "$rel,exit_code=$rc_retry,rescue_timeout_sec=$QE_RESCUE_TIMEOUT_SEC,rescue_only=1" >> "$FAIL_FILE"
      echo "[FAIL] $rel rescue-only timeout_sec=$QE_RESCUE_TIMEOUT_SEC" | tee -a "$LOG_FILE"
    else
      echo "$rel,exit_code=$rc_retry,rescue_non_converged=1,rescue_only=1" >> "$FAIL_FILE"
      echo "[FAIL] $rel convergence NOT achieved (rescue-only)" | tee -a "$LOG_FILE"
    fi
    continue
  fi

  run_relax_once "$d/01_relax" "qe_relax.in" "qe_relax.out"
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
  if ! grep -q "JOB DONE" "$d/01_relax/qe_relax.out"; then
    echo "$rel,exit_code=0,no_job_done=1" >> "$FAIL_FILE"
    echo "[FAIL] $rel no JOB DONE marker" | tee -a "$LOG_FILE"
    continue
  fi
  if [[ "$RELAX_REQUIRE_CONVERGENCE" -eq 1 ]] && grep -q "convergence NOT achieved" "$d/01_relax/qe_relax.out"; then
    if [[ "$RELAX_ENABLE_RESCUE" -eq 1 ]] && [[ -f "$RELAX_RESCUE_SCRIPT" ]]; then
      echo "[INFO] $rel rescue retry: building qe_relax_retry.in from previous run output" | tee -a "$LOG_FILE"
      (
        cd "$d/01_relax" || exit 10
        python3 "$RELAX_RESCUE_SCRIPT" \
          --infile qe_relax.in \
          --outfile qe_relax_retry.in \
          --outlog qe_relax.out \
          --nstep_add "$RELAX_RESCUE_NSTEP_ADD" \
          --electron_maxstep "$RELAX_RESCUE_ELECTRON_MAXSTEP" \
          --mixing_beta "$RELAX_RESCUE_MIXING_BETA" \
          > qe_relax_retry_prep.log 2>&1
      )
      prep_rc=$?
      if [[ $prep_rc -eq 0 ]]; then
        run_relax_rescue_once "$d/01_relax" "qe_relax_retry.in" "qe_relax_retry.out"
        rc_retry=$?
        if [[ $rc_retry -eq 0 ]] && grep -q "JOB DONE" "$d/01_relax/qe_relax_retry.out" && ! grep -q "convergence NOT achieved" "$d/01_relax/qe_relax_retry.out"; then
          echo "$rel" >> "$PASS_FILE"
          echo "[PASS] $rel rescue_converged=1" | tee -a "$LOG_FILE"
          continue
        fi
        if [[ $rc_retry -eq 124 ]]; then
          echo "$rel,exit_code=$rc_retry,rescue_timeout_sec=$QE_RESCUE_TIMEOUT_SEC,rescue=1" >> "$FAIL_FILE"
          echo "[FAIL] $rel rescue timeout_sec=$QE_RESCUE_TIMEOUT_SEC" | tee -a "$LOG_FILE"
        else
          echo "$rel,exit_code=$rc_retry,rescue_non_converged=1,rescue=1" >> "$FAIL_FILE"
          echo "[FAIL] $rel convergence NOT achieved (after rescue)" | tee -a "$LOG_FILE"
        fi
      else
        echo "$rel,exit_code=0,non_converged=1,rescue_prepare_failed=1" >> "$FAIL_FILE"
        echo "[FAIL] $rel convergence NOT achieved (rescue prepare failed)" | tee -a "$LOG_FILE"
      fi
      continue
    fi
    echo "$rel,exit_code=0,non_converged=1" >> "$FAIL_FILE"
    echo "[FAIL] $rel convergence NOT achieved" | tee -a "$LOG_FILE"
    continue
  fi
  echo "$rel" >> "$PASS_FILE"
  echo "[PASS] $rel" | tee -a "$LOG_FILE"
done < "$INPUT_LIST"

pass_n=$(wc -l < "$PASS_FILE")
fail_n=$(wc -l < "$FAIL_FILE")
echo "[DONE] relax complete passed=$pass_n failed=$fail_n" | tee -a "$LOG_FILE"
