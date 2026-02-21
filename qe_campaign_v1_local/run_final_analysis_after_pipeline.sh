#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-/mnt/c/Users/sunwo/Desktop/aim-materials/qe_campaign_v1_local}"
REPO="${2:-/mnt/c/Users/sunwo/Desktop/aim-materials}"
ts="$(date +%Y%m%d_%H%M%S)"

cd "$ROOT"
echo "[INFO] final-analysis start ts=$ts"

./check_qe_status.sh > "final_status_snapshot_${ts}.txt" || true

cd "$REPO"
if [[ -f "analyze_qe_campaign_results.py" ]]; then
  python analyze_qe_campaign_results.py \
    --campaign_manifest qe_campaign_v1_local/campaign_manifest.csv \
    --out_csv "qe_campaign_v1_local/analysis_strict_latest_${ts}.csv" \
    --out_summary_json "qe_campaign_v1_local/analysis_strict_latest_summary_${ts}.json" \
    --out_validated_csv "qe_campaign_v1_local/analysis_strict_latest_validated_${ts}.csv" \
    --norm_stats_npz normalization_stats_fixed.npz \
    --pred_voigt_is_normalized
  echo "[INFO] analysis outputs generated with ts=$ts"
else
  echo "[WARN] analyze_qe_campaign_results.py not found, skipped"
fi

cd "$ROOT"

python3 - "$ts" << 'PY'
import csv
import pathlib
import sys

ts = sys.argv[1]
root = pathlib.Path(".")

def read_set(name):
    p = root / name
    if not p.exists():
        return set()
    out = set()
    for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        out.add(line.split(",")[0].strip())
    return out

cands = set()
for fname in [
    "candidate_paths_relax_nonconverged_retry.txt",
    "candidate_paths_rerun_top10_strict_pending.txt",
    "candidate_paths_strict_all_for_scf.txt",
]:
    p = root / fname
    if p.exists():
        for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
            s = line.strip()
            if s:
                cands.add(s)

relax_pass = read_set("relax_passed.txt") | read_set("relax_passed_rescue_nonconv.txt")
relax_fail = read_set("relax_failed.txt") | read_set("relax_failed_rescue_nonconv.txt")
scf_pass = read_set("scf_passed.txt")
scf_fail = read_set("scf_failed.txt")
elastic_pass = read_set("elastic_passed.txt")
elastic_fail = read_set("elastic_failed.txt")

rows = []
for c in sorted(cands):
    rows.append(
        {
            "candidate": c,
            "relax_status": "pass" if c in relax_pass else ("fail" if c in relax_fail else "pending"),
            "scf_status": "pass" if c in scf_pass else ("fail" if c in scf_fail else "pending"),
            "elastic_status": "pass" if c in elastic_pass else ("fail" if c in elastic_fail else "pending"),
        }
    )

out_csv = root / f"final_pipeline_candidate_status_{ts}.csv"
with out_csv.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["candidate", "relax_status", "scf_status", "elastic_status"])
    w.writeheader()
    w.writerows(rows)

print(f"[INFO] wrote {out_csv}")
print(f"[INFO] candidates={len(rows)}")
PY

echo "[DONE] final-analysis finished ts=$ts"

