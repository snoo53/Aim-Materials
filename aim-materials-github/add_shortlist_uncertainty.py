"""
Attach ensemble uncertainty summaries to shortlist manifest.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Dict, List

import numpy as np


def read_csv(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_ensemble_csv(path: str) -> Dict[str, dict]:
    rows = read_csv(path)
    out = {}
    for r in rows:
        mid = str(r.get("material_id", ""))
        if not mid:
            continue
        s_std = []
        c_std = []
        for k, v in r.items():
            if k.endswith("_std"):
                try:
                    fv = float(v)
                except Exception:
                    continue
                if k.startswith("s"):
                    s_std.append(fv)
                elif k.startswith("c"):
                    c_std.append(fv)
        rec = {
            "scalar_std_mean": float(np.mean(s_std)) if s_std else 0.0,
            "scalar_std_max": float(np.max(s_std)) if s_std else 0.0,
            "voigt_std_mean": float(np.mean(c_std)) if c_std else 0.0,
            "voigt_std_max": float(np.max(c_std)) if c_std else 0.0,
        }
        out[mid] = rec
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest_csv", required=True)
    ap.add_argument("--ensemble_2el_csv", required=True)
    ap.add_argument("--ensemble_3el_csv", required=True)
    ap.add_argument("--ensemble_4el_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_summary_json", required=True)
    args = ap.parse_args()

    manifest = read_csv(args.manifest_csv)
    ens_by_set = {
        "2el": load_ensemble_csv(args.ensemble_2el_csv),
        "3el": load_ensemble_csv(args.ensemble_3el_csv),
        "4el": load_ensemble_csv(args.ensemble_4el_csv),
    }

    out_rows = []
    missing = 0
    for r in manifest:
        s = str(r.get("set", ""))
        mid = str(r.get("material_id", ""))
        rec = dict(r)
        u = ens_by_set.get(s, {}).get(mid)
        if u is None:
            missing += 1
            u = {
                "scalar_std_mean": 0.0,
                "scalar_std_max": 0.0,
                "voigt_std_mean": 0.0,
                "voigt_std_max": 0.0,
            }
        rec.update(u)
        out_rows.append(rec)

    fieldnames = list(out_rows[0].keys()) if out_rows else []
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True) if os.path.dirname(args.out_csv) else None
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)

    sv = [float(r["scalar_std_mean"]) for r in out_rows] if out_rows else [0.0]
    cv = [float(r["voigt_std_mean"]) for r in out_rows] if out_rows else [0.0]
    summary = {
        "n_rows": len(out_rows),
        "missing_uncertainty_rows": int(missing),
        "scalar_std_mean_mean": float(np.mean(sv)),
        "scalar_std_mean_max": float(np.max(sv)),
        "voigt_std_mean_mean": float(np.mean(cv)),
        "voigt_std_mean_max": float(np.max(cv)),
        "out_csv": args.out_csv,
    }
    with open(args.out_summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"wrote {args.out_csv}")
    print(f"wrote {args.out_summary_json}")
    print(f"missing_uncertainty_rows={missing}")


if __name__ == "__main__":
    main()

