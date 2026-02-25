#!/usr/bin/env bash
set -euo pipefail

CAMPAIGN_DIR="/mnt/c/Users/sunwo/Desktop/aim-materials/dft_campaign_v4all220"
FULL_LIST="/mnt/c/Users/sunwo/Desktop/aim-materials/dft_shortlist_from_271/candidate_paths_tierAB_v4.txt"
PENDING_LIST="$CAMPAIGN_DIR/candidate_paths_tierAB_v4_resume_pending.txt"
PASS_FILE="$CAMPAIGN_DIR/relax_passed_tierab_v4.txt"
FAIL_FILE="$CAMPAIGN_DIR/relax_failed_tierab_v4.txt"
LOG_FILE="$CAMPAIGN_DIR/relax_stage_tierab_v4.log"
LAUNCH_LOG="$CAMPAIGN_DIR/tierab_v4_resume_launcher.log"

cd "$CAMPAIGN_DIR"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export QE_FORCE_MPIRUN=1
export QE_TIMEOUT_SEC=21600
export RELAX_REQUIRE_CONVERGENCE=1
export RELAX_ENABLE_RESCUE=1
export QE_RESCUE_TIMEOUT_SEC=7200
export RELAX_RESCUE_NSTEP_ADD=180
export RELAX_RESCUE_ELECTRON_MAXSTEP=300
export RELAX_RESCUE_MIXING_BETA=0.20
export RELAX_RESCUE_ONLY=0
export RELAX_APPEND_STAGE=1
export RELAX_STAGE_TAG=tierab_v4
export RELAX_SKIP_EXTRAORDINARY_GRID=1
export RELAX_MAX_FFT_DIM=900
export RELAX_MAX_FFT_DIM_RATIO=10
export RELAX_MAX_FFT_GRID_POINTS=70000000

echo "[LAUNCH] $(date '+%Y-%m-%d %H:%M:%S %Z') start tierab_v4 resume" >> "$LAUNCH_LOG"

# If the previous session died right after a [RUN] entry, mark that candidate
# as interrupted so this resume call moves to the next one.
if [[ -f "$LOG_FILE" ]]; then
  last_run="$(grep '^\[RUN \]' "$LOG_FILE" | tail -n 1 | sed 's/^\[RUN \] //')"
  if [[ -n "$last_run" ]]; then
    done_flag=0
    if [[ -f "$PASS_FILE" ]] && grep -Fxq "$last_run" "$PASS_FILE"; then
      done_flag=1
    fi
    if [[ "$done_flag" -eq 0 ]] && [[ -f "$FAIL_FILE" ]] && awk -F',' -v x="$last_run" '$1==x{f=1} END{exit f?0:1}' "$FAIL_FILE"; then
      done_flag=1
    fi
    if [[ "$done_flag" -eq 0 ]]; then
      echo "$last_run,exit_code=130,interrupted_previous_session=1" >> "$FAIL_FILE"
      echo "[FAIL] $last_run interrupted_previous_session=1" | tee -a "$LOG_FILE" >/dev/null
      echo "[INFO] Marked interrupted previous run as failed: $last_run" >> "$LAUNCH_LOG"
    fi
  fi
fi

pending_n="$(
python3 - "$FULL_LIST" "$PASS_FILE" "$FAIL_FILE" "$PENDING_LIST" <<'PY'
import sys
from pathlib import Path

full_list = Path(sys.argv[1])
pass_file = Path(sys.argv[2])
fail_file = Path(sys.argv[3])
pending_file = Path(sys.argv[4])

full_rows = [line.strip() for line in full_list.read_text(encoding="utf-8").splitlines() if line.strip()]
done = set()

if pass_file.exists():
    done.update(line.strip() for line in pass_file.read_text(encoding="utf-8").splitlines() if line.strip())
if fail_file.exists():
    for line in fail_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        done.add(line.split(",", 1)[0].strip())

pending = [row for row in full_rows if row not in done]
pending_file.write_text("\n".join(pending) + ("\n" if pending else ""), encoding="utf-8")
print(len(pending))
PY
)"

echo "[INFO] pending_count=$pending_n pending_list=$PENDING_LIST" >> "$LAUNCH_LOG"
if [[ "$pending_n" -le 0 ]]; then
  echo "[EXIT] $(date '+%Y-%m-%d %H:%M:%S %Z') no pending candidates" >> "$LAUNCH_LOG"
  exit 0
fi

bash run_relax.sh /home/sunwoo/miniforge3/envs/qe75/bin/pw.x 8 "$PENDING_LIST"
rc=$?
echo "[EXIT] $(date '+%Y-%m-%d %H:%M:%S %Z') run_relax.sh rc=${rc}" >> "$LAUNCH_LOG"
exit "$rc"
