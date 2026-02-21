"""
Build a non-redundant DFT backup queue from a campaign manifest.

This is intended for deadline mode where one relax batch is already running and
you want the next batch ready without redoing CHGNet filtering on the same IDs.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from collections import defaultdict
from typing import Dict, List, Tuple


_FORMULA_RE = re.compile(r"([A-Z][a-z]*)([0-9.]*)")


def _norm_relpath(p: str) -> str:
    return p.replace("\\", "/").strip()


def _as_float(x: str, default: float) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _formula_atom_count(formula: str) -> float:
    total = 0.0
    for _el, num in _FORMULA_RE.findall(formula or ""):
        if not num:
            total += 1.0
            continue
        try:
            total += float(num)
        except Exception:
            total += 1.0
    return total if total > 0.0 else 99.0


def read_manifest(path: str) -> Dict[str, dict]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    out: Dict[str, dict] = {}
    for r in rows:
        rel = _norm_relpath(r.get("candidate_relpath", ""))
        if not rel:
            continue
        rr = dict(r)
        rr["candidate_relpath"] = rel
        rr["formula_atom_count"] = _formula_atom_count(rr.get("reduced_formula", ""))
        out[rel] = rr
    return out


def read_relpath_lines(path: str) -> List[str]:
    if not path or not os.path.exists(path):
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


def _score_row(
    r: dict,
    w_sel: float,
    w_quality: float,
    w_atoms: float,
    w_scalar_unc: float,
    w_voigt_unc: float,
    w_force: float,
) -> float:
    sel = _as_float(r.get("selection_score", ""), 0.0)
    q = _as_float(r.get("quality_score", ""), 0.0)
    n_atoms = _as_float(str(r.get("formula_atom_count", "")), 99.0)
    us = _as_float(r.get("scalar_std_mean", ""), 0.0)
    uv = _as_float(r.get("voigt_std_mean", ""), 0.0)
    ff = _as_float(r.get("chgnet_force_max", ""), 0.0)
    # Higher is better.
    return (
        w_sel * sel
        + w_quality * q
        - w_atoms * n_atoms
        - w_scalar_unc * us
        - w_voigt_unc * uv
        - w_force * ff
    )


def select_backup(
    pending_relpaths: List[str],
    manifest: Dict[str, dict],
    exclude_relpaths: List[str],
    top_k: int,
    max_per_formula: int,
    balance_sets: bool,
    w_sel: float,
    w_quality: float,
    w_atoms: float,
    w_scalar_unc: float,
    w_voigt_unc: float,
    w_force: float,
) -> List[dict]:
    exclude = {_norm_relpath(x) for x in exclude_relpaths}
    rows = []
    for rel in pending_relpaths:
        rel = _norm_relpath(rel)
        if not rel or rel in exclude:
            continue
        r = manifest.get(rel)
        if r is None:
            continue
        rr = dict(r)
        rr["candidate_relpath"] = rel
        rr["backup_score"] = _score_row(
            rr,
            w_sel=w_sel,
            w_quality=w_quality,
            w_atoms=w_atoms,
            w_scalar_unc=w_scalar_unc,
            w_voigt_unc=w_voigt_unc,
            w_force=w_force,
        )
        rows.append(rr)

    rows.sort(key=lambda x: float(x.get("backup_score", -1e18)), reverse=True)

    if not balance_sets:
        out = []
        by_formula = defaultdict(int)
        for r in rows:
            rf = (r.get("reduced_formula") or "").strip()
            if max_per_formula > 0 and by_formula[rf] >= max_per_formula:
                continue
            by_formula[rf] += 1
            out.append(r)
            if len(out) >= top_k:
                break
        return out

    by_set = defaultdict(list)
    for r in rows:
        by_set[(r.get("set") or "").strip()].append(r)
    set_names = sorted([s for s in by_set.keys() if s])
    if not set_names:
        set_names = [""]

    out = []
    by_formula = defaultdict(int)
    idx = {s: 0 for s in set_names}
    while len(out) < top_k:
        progressed = False
        for s in set_names:
            arr = by_set[s]
            while idx[s] < len(arr):
                r = arr[idx[s]]
                idx[s] += 1
                rf = (r.get("reduced_formula") or "").strip()
                if max_per_formula > 0 and by_formula[rf] >= max_per_formula:
                    continue
                by_formula[rf] += 1
                out.append(r)
                progressed = True
                break
            if len(out) >= top_k:
                break
        if not progressed:
            break
    return out


def write_outputs(rows: List[dict], out_list: str, out_csv: str) -> None:
    if os.path.dirname(out_list):
        os.makedirs(os.path.dirname(out_list), exist_ok=True)
    if os.path.dirname(out_csv):
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    with open(out_list, "w", encoding="utf-8", newline="") as f:
        for r in rows:
            f.write(_norm_relpath(r.get("candidate_relpath", "")) + "\n")

    fields = [
        "set",
        "material_id",
        "reduced_formula",
        "candidate_relpath",
        "selection_score",
        "quality_score",
        "scalar_std_mean",
        "voigt_std_mean",
        "chgnet_force_max",
        "formula_atom_count",
        "backup_score",
    ]
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign_dir", default="qe_campaign_v1_local")
    ap.add_argument("--manifest_file", default="")
    ap.add_argument("--pending_list", default="")
    ap.add_argument(
        "--exclude_files",
        nargs="*",
        default=[],
        help="Files containing relpaths to exclude (supports CSV-style lines).",
    )
    ap.add_argument("--top_k", type=int, default=12)
    ap.add_argument("--max_per_formula", type=int, default=1)
    ap.add_argument("--balance_sets", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--w_sel", type=float, default=2.5)
    ap.add_argument("--w_quality", type=float, default=0.8)
    ap.add_argument("--w_atoms", type=float, default=0.45)
    ap.add_argument("--w_scalar_unc", type=float, default=15.0)
    ap.add_argument("--w_voigt_unc", type=float, default=10.0)
    ap.add_argument("--w_force", type=float, default=3.0)
    ap.add_argument("--out_list", default="")
    ap.add_argument("--out_csv", default="")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    campaign_dir = os.path.normpath(args.campaign_dir)
    manifest_file = (
        os.path.normpath(args.manifest_file)
        if args.manifest_file
        else os.path.join(campaign_dir, "campaign_manifest.csv")
    )
    pending_list = (
        os.path.normpath(args.pending_list)
        if args.pending_list
        else os.path.join(campaign_dir, "candidate_paths_march1_pending.txt")
    )
    out_list = (
        os.path.normpath(args.out_list)
        if args.out_list
        else os.path.join(campaign_dir, f"candidate_paths_march1_backup_fast{int(args.top_k)}.txt")
    )
    out_csv = (
        os.path.normpath(args.out_csv)
        if args.out_csv
        else os.path.join(campaign_dir, f"march1_backup_fast{int(args.top_k)}.csv")
    )

    manifest = read_manifest(manifest_file)
    pending = read_relpath_lines(pending_list)

    exclude_relpaths: List[str] = []
    for ef in args.exclude_files:
        exclude_relpaths.extend(read_relpath_lines(os.path.normpath(ef)))

    selected = select_backup(
        pending_relpaths=pending,
        manifest=manifest,
        exclude_relpaths=exclude_relpaths,
        top_k=int(args.top_k),
        max_per_formula=int(args.max_per_formula),
        balance_sets=bool(args.balance_sets),
        w_sel=float(args.w_sel),
        w_quality=float(args.w_quality),
        w_atoms=float(args.w_atoms),
        w_scalar_unc=float(args.w_scalar_unc),
        w_voigt_unc=float(args.w_voigt_unc),
        w_force=float(args.w_force),
    )

    write_outputs(selected, out_list, out_csv)

    print(f"manifest_rows={len(manifest)}")
    print(f"pending_count={len(pending)}")
    print(f"exclude_count={len(set(map(_norm_relpath, exclude_relpaths)))}")
    print(f"selected_count={len(selected)}")
    print(f"out_list={out_list}")
    print(f"out_csv={out_csv}")
    if selected:
        print("preview:")
        for r in selected[: min(12, len(selected))]:
            print(
                "  "
                f"{r.get('set','')} "
                f"{r.get('candidate_relpath','')} "
                f"score={float(r.get('backup_score',0.0)):.4f} "
                f"formula={r.get('reduced_formula','')}"
            )


if __name__ == "__main__":
    main()

