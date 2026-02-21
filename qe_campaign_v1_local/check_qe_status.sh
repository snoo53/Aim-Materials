#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-/mnt/c/Users/sunwo/Desktop/aim-materials/qe_campaign_v1_local}"
cd "$ROOT"

echo "=== now ==="
date "+%Y-%m-%d %H:%M:%S %Z"

echo
echo "=== tmux sessions ==="
tmux ls 2>/dev/null || echo "(none)"

echo
echo "=== active processes ==="
pgrep -af "run_followup_strict.sh|run_relax_rescue_nonconverged.sh|run_relax.sh|run_scf.sh|run_elastic.sh|pw.x -in qe_relax.in|pw.x -in qe_relax_retry.in|pw.x -in qe_scf.in" || echo "(none)"

echo
echo "=== timeout remaining ==="
ps -eo pid,etimes,args | awk '
  /timeout [0-9]+ .*pw\.x -in qe_(relax|relax_retry|scf)\.in/ {
    pid=$1; et=$2;
    if (match($0, /timeout [0-9]+/)) {
      tstr = substr($0, RSTART, RLENGTH);
      split(tstr, a, " ");
      to = a[2] + 0;
      rem = to - et;
      if (rem < 0) rem = 0;
      printf "pid=%s elapsed=%ss timeout=%ss remaining=%ss\n", pid, et, to, rem;
    }
  }
' || true

echo
echo "=== stage counts ==="
for f in \
  relax_passed.txt relax_failed.txt \
  relax_passed_rescue_nonconv.txt relax_failed_rescue_nonconv.txt \
  scf_passed.txt scf_failed.txt \
  elastic_passed.txt elastic_failed.txt; do
  if [[ -f "$f" ]]; then
    printf "%5d %s\n" "$(wc -l < "$f")" "$f"
  else
    printf "%5s %s\n" "-" "$f"
  fi
done

echo
echo "=== latest relax logs ==="
tail -n 12 relax_stage.log 2>/dev/null || true
echo "---"
tail -n 12 relax_stage_rescue_nonconv.log 2>/dev/null || true

echo
echo "=== latest scf/elastic logs ==="
tail -n 12 scf_stage.log 2>/dev/null || true
echo "---"
tail -n 12 elastic_stage.log 2>/dev/null || true
