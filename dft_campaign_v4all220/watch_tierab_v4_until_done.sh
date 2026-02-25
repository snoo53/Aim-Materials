#!/usr/bin/env bash
set -euo pipefail

CAMPAIGN_DIR="/mnt/c/Users/sunwo/Desktop/aim-materials/dft_campaign_v4all220"
FULL_LIST="/mnt/c/Users/sunwo/Desktop/aim-materials/dft_shortlist_from_271/candidate_paths_tierAB_v4.txt"
PASS_FILE="$CAMPAIGN_DIR/relax_passed_tierab_v4.txt"
FAIL_FILE="$CAMPAIGN_DIR/relax_failed_tierab_v4.txt"
LOG_FILE="$CAMPAIGN_DIR/relax_stage_tierab_v4.log"
WATCHDOG_LOG="$CAMPAIGN_DIR/tierab_v4_watchdog.log"
MP_SYNC_LOG="$CAMPAIGN_DIR/tierab_v4_mp_sync.log"

POLL_SEC="${POLL_SEC:-120}"
RESTART_COOLDOWN_SEC="${RESTART_COOLDOWN_SEC:-15}"
MAX_RETRIES="${MAX_RETRIES:-0}" # 0 means infinite retries.
AUTO_SYNC_MP="${AUTO_SYNC_MP:-1}"
MP_SYNC_MIN_INTERVAL_SEC="${MP_SYNC_MIN_INTERVAL_SEC:-180}"
MP_SYNC_PROJECT="${MP_SYNC_PROJECT:-aim_materials_v1}"
MP_SYNC_SCRIPT="${MP_SYNC_SCRIPT:-$CAMPAIGN_DIR/sync_mpcontribs_tierab_v4_live.py}"
MP_SYNC_PYTHON="${MP_SYNC_PYTHON:-/home/sunwoo/miniforge3/envs/qe75/bin/python}"

run_active() {
  pgrep -f "run_relax.sh /home/sunwoo/miniforge3/envs/qe75/bin/pw.x 8 /mnt/c/Users/sunwo/Desktop/aim-materials/dft_campaign_v4all220/candidate_paths_tierAB_v4_resume_pending.txt" >/dev/null 2>&1
}

pending_count() {
  python3 - "$FULL_LIST" "$PASS_FILE" "$FAIL_FILE" <<'PY'
import sys
from pathlib import Path

full = [x.strip() for x in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines() if x.strip()]
done = set()

pf = Path(sys.argv[2])
if pf.exists():
    done.update(x.strip() for x in pf.read_text(encoding="utf-8").splitlines() if x.strip())

ff = Path(sys.argv[3])
if ff.exists():
    for line in ff.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            done.add(line.split(",", 1)[0].strip())

pending = [x for x in full if x not in done]
print(len(pending))
PY
}

current_run() {
  if [[ -f "$LOG_FILE" ]]; then
    grep '^\[RUN \]' "$LOG_FILE" | tail -n 1 | sed 's/^\[RUN \] //'
  fi
}

mkdir -p "$CAMPAIGN_DIR"
touch "$WATCHDOG_LOG"
touch "$MP_SYNC_LOG"

echo "[WATCHDOG] $(date '+%Y-%m-%d %H:%M:%S %Z') start poll_sec=$POLL_SEC cooldown_sec=$RESTART_COOLDOWN_SEC max_retries=$MAX_RETRIES auto_sync_mp=$AUTO_SYNC_MP sync_interval_sec=$MP_SYNC_MIN_INTERVAL_SEC" | tee -a "$WATCHDOG_LOG"

mp_sync_once() {
  [[ "$AUTO_SYNC_MP" -eq 1 ]] || return 0

  if [[ ! -f "$MP_SYNC_SCRIPT" ]]; then
    echo "[WATCHDOG] $(date '+%Y-%m-%d %H:%M:%S %Z') MP sync skipped: missing script $MP_SYNC_SCRIPT" | tee -a "$WATCHDOG_LOG"
    return 0
  fi
  if [[ -z "${MPCONTRIBS_API_KEY:-}" && -z "${MP_API_KEY:-}" ]]; then
    echo "[WATCHDOG] $(date '+%Y-%m-%d %H:%M:%S %Z') MP sync skipped: MPCONTRIBS_API_KEY/MP_API_KEY not set" | tee -a "$WATCHDOG_LOG"
    return 0
  fi
  if [[ ! -x "$MP_SYNC_PYTHON" ]]; then
    echo "[WATCHDOG] $(date '+%Y-%m-%d %H:%M:%S %Z') MP sync skipped: python not executable at $MP_SYNC_PYTHON" | tee -a "$WATCHDOG_LOG"
    return 0
  fi

  echo "[WATCHDOG] $(date '+%Y-%m-%d %H:%M:%S %Z') MP sync start project=$MP_SYNC_PROJECT python=$MP_SYNC_PYTHON" | tee -a "$WATCHDOG_LOG"
  if (
    cd /mnt/c/Users/sunwo/Desktop/aim-materials
    "$MP_SYNC_PYTHON" "$MP_SYNC_SCRIPT" --project "$MP_SYNC_PROJECT" >> "$MP_SYNC_LOG" 2>&1
  ); then
    echo "[WATCHDOG] $(date '+%Y-%m-%d %H:%M:%S %Z') MP sync success" | tee -a "$WATCHDOG_LOG"
  else
    rc=$?
    echo "[WATCHDOG] $(date '+%Y-%m-%d %H:%M:%S %Z') MP sync failed rc=$rc" | tee -a "$WATCHDOG_LOG"
  fi
}

retry_n=0
last_sync_ts=0
while true; do
  pending_n="$(pending_count)"
  active_n=0
  if run_active; then
    active_n=1
  fi
  cur="$(current_run || true)"
  echo "[WATCHDOG] $(date '+%Y-%m-%d %H:%M:%S %Z') pending=$pending_n active=$active_n current=${cur:-<none>} retry_n=$retry_n" | tee -a "$WATCHDOG_LOG"

  now_ts="$(date +%s)"
  if [[ "$AUTO_SYNC_MP" -eq 1 ]] && (( now_ts - last_sync_ts >= MP_SYNC_MIN_INTERVAL_SEC )); then
    mp_sync_once
    last_sync_ts="$now_ts"
  fi

  if [[ "$pending_n" -le 0 ]]; then
    mp_sync_once
    echo "[WATCHDOG] $(date '+%Y-%m-%d %H:%M:%S %Z') completed all candidates; exiting." | tee -a "$WATCHDOG_LOG"
    exit 0
  fi

  if [[ "$active_n" -eq 1 ]]; then
    sleep "$POLL_SEC"
    continue
  fi

  if [[ "$MAX_RETRIES" -gt 0 ]] && [[ "$retry_n" -ge "$MAX_RETRIES" ]]; then
    echo "[WATCHDOG] $(date '+%Y-%m-%d %H:%M:%S %Z') retry limit reached ($MAX_RETRIES); exiting with failure." | tee -a "$WATCHDOG_LOG"
    exit 2
  fi

  retry_n=$((retry_n + 1))
  echo "[WATCHDOG] $(date '+%Y-%m-%d %H:%M:%S %Z') launching start_tierab_v4_resume.sh (attempt $retry_n)" | tee -a "$WATCHDOG_LOG"
  (
    cd "$CAMPAIGN_DIR"
    bash start_tierab_v4_resume.sh
  )
  rc=$?
  echo "[WATCHDOG] $(date '+%Y-%m-%d %H:%M:%S %Z') start_tierab_v4_resume.sh exited rc=$rc" | tee -a "$WATCHDOG_LOG"
  mp_sync_once
  last_sync_ts="$(date +%s)"
  sleep "$RESTART_COOLDOWN_SEC"
done
