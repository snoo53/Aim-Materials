"""
Select top-K rerun candidates from relax_failed.txt.

Ranking uses campaign_manifest metadata:
- primary: selection_score (desc)
- tie breaks: quality_score (desc), scalar_std_mean (asc),
              voigt_std_mean (asc), chgnet_force_max (asc)

Outputs:
- text list of candidate relpaths for rerun scripts
- csv with ranking metadata
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from typing import Dict, List, Tuple


def _as_float(x: str, default: float) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _norm_relpath(p: str) -> str:
    return p.replace("\\", "/").strip()


def read_failed_relpaths(path: str) -> List[str]:
    if not os.path.exists(path):
        return []
    out: List[str] = []
    seen = set()
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            rel = _norm_relpath(s.split(",", 1)[0])
            if not rel or rel.startswith("#"):
                continue
            if rel in seen:
                continue
            seen.add(rel)
            out.append(rel)
    return out


def read_manifest(path: str) -> Dict[str, dict]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    out: Dict[str, dict] = {}
    for r in rows:
        rel = _norm_relpath(r.get("candidate_relpath", ""))
        if rel:
            out[rel] = r
    return out


def row_key(r: dict) -> Tuple[float, float, float, float]:
    # sort ascending on tuple; negate desc metrics
    sel = -_as_float(r.get("selection_score", ""), -1e18)
    q = -_as_float(r.get("quality_score", ""), -1e18)
    us = _as_float(r.get("scalar_std_mean", ""), 1e18)
    uv = _as_float(r.get("voigt_std_mean", ""), 1e18)
    ff = _as_float(r.get("chgnet_force_max", ""), 1e18)
    # Include force only as weak tie-breaker by adding to uv-scale tuple.
    return (sel, q, us, uv + 1e-6 * ff)


def select_topk(
    failed_relpaths: List[str],
    manifest_map: Dict[str, dict],
    top_k: int,
    max_per_formula: int,
    balance_sets: bool,
) -> List[dict]:
    candidates: List[dict] = []
    for rel in failed_relpaths:
        r = manifest_map.get(rel)
        if r is None:
            continue
        rr = dict(r)
        rr["candidate_relpath"] = rel
        candidates.append(rr)

    if not candidates:
        return []

    ranked = sorted(candidates, key=row_key)

    if not balance_sets:
        out: List[dict] = []
        formula_count = defaultdict(int)
        for r in ranked:
            rf = (r.get("reduced_formula") or "").strip()
            if max_per_formula > 0 and formula_count[rf] >= max_per_formula:
                continue
            formula_count[rf] += 1
            out.append(r)
            if len(out) >= top_k:
                break
        return out

    # Balanced selection across 2el/3el/4el (or arbitrary set labels).
    by_set: Dict[str, List[dict]] = defaultdict(list)
    for r in ranked:
        by_set[(r.get("set") or "").strip()].append(r)
    set_names = sorted(k for k in by_set.keys() if k)
    if not set_names:
        set_names = [""]

    out = []
    formula_count = defaultdict(int)
    idx = {s: 0 for s in set_names}

    while len(out) < top_k:
        progressed = False
        for s in set_names:
            arr = by_set.get(s, [])
            while idx[s] < len(arr):
                r = arr[idx[s]]
                idx[s] += 1
                rf = (r.get("reduced_formula") or "").strip()
                if max_per_formula > 0 and formula_count[rf] >= max_per_formula:
                    continue
                formula_count[rf] += 1
                out.append(r)
                progressed = True
                break
            if len(out) >= top_k:
                break
        if not progressed:
            break
    return out


def write_outputs(rows: List[dict], out_list: str, out_csv: str) -> None:
    os.makedirs(os.path.dirname(out_list), exist_ok=True) if os.path.dirname(out_list) else None
    os.makedirs(os.path.dirname(out_csv), exist_ok=True) if os.path.dirname(out_csv) else None

    with open(out_list, "w", encoding="utf-8", newline="") as f:
        for r in rows:
            f.write(_norm_relpath(r.get("candidate_relpath", "")) + "\n")

    fields = [
        "set",
        "rank_in_set",
        "material_id",
        "reduced_formula",
        "selection_score",
        "quality_score",
        "scalar_std_mean",
        "voigt_std_mean",
        "chgnet_force_max",
        "candidate_relpath",
    ]
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign_dir", default="qe_campaign_v1_local")
    ap.add_argument("--failed_file", default="")
    ap.add_argument("--manifest_file", default="")
    ap.add_argument("--top_k", type=int, default=10)
    ap.add_argument("--max_per_formula", type=int, default=2)
    ap.add_argument("--balance_sets", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--out_list", default="")
    ap.add_argument("--out_csv", default="")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    campaign_dir = os.path.normpath(args.campaign_dir)
    failed_file = os.path.normpath(args.failed_file) if args.failed_file else os.path.join(campaign_dir, "relax_failed.txt")
    manifest_file = os.path.normpath(args.manifest_file) if args.manifest_file else os.path.join(campaign_dir, "campaign_manifest.csv")
    out_list = os.path.normpath(args.out_list) if args.out_list else os.path.join(campaign_dir, f"candidate_paths_rerun_top{int(args.top_k)}.txt")
    out_csv = os.path.normpath(args.out_csv) if args.out_csv else os.path.join(campaign_dir, f"rerun_top{int(args.top_k)}_from_relax_failed.csv")

    failed = read_failed_relpaths(failed_file)
    manifest_map = read_manifest(manifest_file)
    selected = select_topk(
        failed_relpaths=failed,
        manifest_map=manifest_map,
        top_k=int(args.top_k),
        max_per_formula=int(args.max_per_formula),
        balance_sets=bool(args.balance_sets),
    )
    write_outputs(selected, out_list, out_csv)

    print(f"failed_count={len(failed)}")
    print(f"selected_count={len(selected)}")
    print(f"out_list={out_list}")
    print(f"out_csv={out_csv}")
    if selected:
        print("top_selected_preview:")
        for r in selected[:10]:
            rel = _norm_relpath(r.get("candidate_relpath", ""))
            ss = r.get("selection_score", "")
            rf = r.get("reduced_formula", "")
            st = r.get("set", "")
            print(f"  {st} {rel} score={ss} formula={rf}")


if __name__ == "__main__":
    main()

