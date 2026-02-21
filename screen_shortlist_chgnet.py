"""
Screen shortlist candidates with CHGNet single-point predictions.

Outputs:
- CSV with per-candidate CHGNet energy/force/stress indicators.
- JSON summary with pass rates and distribution stats.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import numpy as np
from chgnet.model import CHGNet
from pymatgen.core import Structure


def read_csv(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: str, rows: List[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    fieldnames = list(rows[0].keys()) if rows else []
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def safe_float(x, default: float = 0.0) -> float:
    try:
        v = float(x)
        if np.isfinite(v):
            return float(v)
        return float(default)
    except Exception:
        return float(default)


def _stats(vals: List[float]) -> Dict[str, float]:
    if not vals:
        return {}
    a = np.array(vals, dtype=float)
    return {
        "min": float(np.min(a)),
        "p10": float(np.percentile(a, 10)),
        "p25": float(np.percentile(a, 25)),
        "median": float(np.median(a)),
        "mean": float(np.mean(a)),
        "p75": float(np.percentile(a, 75)),
        "p90": float(np.percentile(a, 90)),
        "max": float(np.max(a)),
    }


def evaluate_row(model: CHGNet, row: dict, force_max: float, stress_max: float) -> Tuple[dict, dict]:
    out = dict(row)
    cif_path = os.path.normpath(str(row.get("cif_path", "")).strip())
    out["chgnet_status"] = "ok"
    out["chgnet_error"] = ""
    out["chgnet_n_sites"] = 0
    out["chgnet_e_per_atom"] = ""
    out["chgnet_force_max"] = ""
    out["chgnet_force_mean"] = ""
    out["chgnet_stress_fro"] = ""
    out["chgnet_stress_trace"] = ""
    out["chgnet_pass_force"] = ""
    out["chgnet_pass_stress"] = ""
    out["chgnet_pass"] = ""

    info = {"ok": False}
    if not cif_path or not os.path.exists(cif_path):
        out["chgnet_status"] = "missing_cif"
        out["chgnet_error"] = cif_path
        return out, info

    try:
        s = Structure.from_file(cif_path)
        pred = model.predict_structure(s)

        e = float(np.asarray(pred["e"], dtype=float).reshape(-1)[0])
        f = np.asarray(pred["f"], dtype=float)
        st = np.asarray(pred["s"], dtype=float)

        if f.ndim == 1:
            f = f.reshape(-1, 3)
        if st.ndim == 1 and st.size == 9:
            st = st.reshape(3, 3)

        f_norm = np.linalg.norm(f, axis=1) if f.size else np.zeros((0,), dtype=float)
        fmax = float(np.max(f_norm)) if f_norm.size else 0.0
        fmean = float(np.mean(f_norm)) if f_norm.size else 0.0
        sfro = float(np.linalg.norm(st)) if st.size else 0.0
        strace = float(np.trace(st)) if st.ndim == 2 and st.shape[0] == st.shape[1] else 0.0

        pass_force = bool(fmax <= float(force_max))
        pass_stress = bool(sfro <= float(stress_max))
        pass_all = bool(pass_force and pass_stress)

        out["chgnet_n_sites"] = int(len(s))
        out["chgnet_e_per_atom"] = e
        out["chgnet_force_max"] = fmax
        out["chgnet_force_mean"] = fmean
        out["chgnet_stress_fro"] = sfro
        out["chgnet_stress_trace"] = strace
        out["chgnet_pass_force"] = pass_force
        out["chgnet_pass_stress"] = pass_stress
        out["chgnet_pass"] = pass_all

        info = {
            "ok": True,
            "set": str(row.get("set", "")),
            "e": e,
            "fmax": fmax,
            "fmean": fmean,
            "sfro": sfro,
            "pass_force": pass_force,
            "pass_stress": pass_stress,
            "pass_all": pass_all,
        }
        return out, info
    except Exception as exc:
        out["chgnet_status"] = "predict_fail"
        out["chgnet_error"] = str(exc)
        return out, info


def summarize(screen_info: List[dict], force_max: float, stress_max: float) -> dict:
    n = len(screen_info)
    ok = [x for x in screen_info if x.get("ok")]
    by_set = defaultdict(list)
    for x in ok:
        by_set[x.get("set", "")].append(x)

    def pack(items: List[dict]) -> dict:
        e = [float(x["e"]) for x in items]
        fmax = [float(x["fmax"]) for x in items]
        fmean = [float(x["fmean"]) for x in items]
        sfro = [float(x["sfro"]) for x in items]
        n_items = len(items)
        p_force = sum(1 for x in items if x["pass_force"])
        p_stress = sum(1 for x in items if x["pass_stress"])
        p_all = sum(1 for x in items if x["pass_all"])
        return {
            "n_ok": n_items,
            "force_max_threshold": float(force_max),
            "stress_fro_threshold": float(stress_max),
            "pass_force": {"count": int(p_force), "total": int(n_items), "rate": float(p_force / n_items) if n_items else None},
            "pass_stress": {"count": int(p_stress), "total": int(n_items), "rate": float(p_stress / n_items) if n_items else None},
            "pass_all": {"count": int(p_all), "total": int(n_items), "rate": float(p_all / n_items) if n_items else None},
            "e_per_atom_stats": _stats(e),
            "force_max_stats": _stats(fmax),
            "force_mean_stats": _stats(fmean),
            "stress_fro_stats": _stats(sfro),
        }

    out = {
        "n_total_rows": int(n),
        "n_ok": int(len(ok)),
        "n_failed": int(n - len(ok)),
        "overall": pack(ok),
        "by_set": {k: pack(v) for k, v in by_set.items()},
    }
    return out


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest_csv", required=True, help="Shortlist manifest CSV.")
    ap.add_argument("--out_csv", required=True, help="Output CSV with CHGNet metrics.")
    ap.add_argument("--out_summary_json", required=True, help="Output JSON summary.")
    ap.add_argument("--force_max", type=float, default=0.25, help="Force threshold (eV/A).")
    ap.add_argument("--stress_fro_max", type=float, default=50.0, help="Stress Frobenius threshold (GPa-ish scale).")
    return ap.parse_args()


def main():
    args = parse_args()
    rows = read_csv(args.manifest_csv)
    if not rows:
        raise RuntimeError(f"No rows found: {args.manifest_csv}")

    model = CHGNet.load()

    out_rows = []
    info_rows = []
    status_counter = Counter()
    for i, row in enumerate(rows, start=1):
        out, info = evaluate_row(
            model=model,
            row=row,
            force_max=float(args.force_max),
            stress_max=float(args.stress_fro_max),
        )
        out_rows.append(out)
        info_rows.append(info)
        status_counter[out.get("chgnet_status", "unknown")] += 1
        if i == 1 or i % 20 == 0:
            print(f"[{i}/{len(rows)}] status={out.get('chgnet_status')}, id={row.get('material_id')}")

    write_csv(args.out_csv, out_rows)
    summary = summarize(info_rows, force_max=float(args.force_max), stress_max=float(args.stress_fro_max))
    summary["status_counts"] = dict(status_counter)
    with open(args.out_summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=" * 72)
    print("CHGNET SHORTLIST SCREEN COMPLETE")
    print("=" * 72)
    print(f"in_manifest={args.manifest_csv}")
    print(f"out_csv={args.out_csv}")
    print(f"out_summary={args.out_summary_json}")
    print(f"status_counts={dict(status_counter)}")
    p_all = summary["overall"]["pass_all"]
    if p_all["rate"] is not None:
        print(f"pass_all={p_all['count']}/{p_all['total']} ({100.0*p_all['rate']:.2f}%)")


if __name__ == "__main__":
    main()

