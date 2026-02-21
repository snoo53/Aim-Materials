"""
Prepare DFT-ready shortlist package from strict candidate files.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from dataclasses import dataclass
from typing import Dict, List

from pymatgen.core import Structure


@dataclass
class SetConfig:
    name: str
    candidates_csv: str
    pred_json: str
    cif_dir: str


def read_csv(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def as_bool(x) -> bool:
    if isinstance(x, bool):
        return x
    if x is None:
        return False
    s = str(x).strip().lower()
    return s in {"1", "true", "t", "yes", "y"}


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return float(default)


def load_set(cfg: SetConfig) -> Dict[str, dict]:
    rows = read_json(cfg.pred_json)
    return {str(r.get("material_id")): r for r in rows}


def choose_rows(rows: List[dict], top_n: int, require_novel: bool) -> List[dict]:
    out = []
    seen = set()
    for r in rows:
        if not as_bool(r.get("strict_pass")):
            continue
        if require_novel and not as_bool(r.get("formula_novel_vs_train")):
            continue
        mid = str(r.get("material_id", ""))
        if not mid or mid in seen:
            continue
        seen.add(mid)
        out.append(r)
        if len(out) >= int(top_n):
            break
    return out


def write_manifest(rows: List[dict], out_csv: str):
    cols = [
        "set",
        "rank_in_set",
        "material_id",
        "reduced_formula",
        "quality_score",
        "strict_pass",
        "formula_novel_vs_train",
        "spacegroup_symbol",
        "spacegroup_number",
        "crystal_system",
        "density",
        "volume_per_atom",
        "min_distance",
        "B_H",
        "G_H",
        "E_H",
        "nu_H",
        "A_U",
        "cif_path",
        "poscar_path",
        "json_record_path",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in cols})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="densityaware_strict_v3_retrain")
    ap.add_argument("--top_n_per_set", type=int, default=40)
    ap.add_argument("--require_novel", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--out_dir", default="dft_shortlist_v1")
    args = ap.parse_args()

    sets = ["2el", "3el", "4el"]
    cfgs = [
        SetConfig(
            name=s,
            candidates_csv=f"candidates_{s}_{args.tag}_strict_novel_unique.csv",
            pred_json=f"generated_materials_{s}_{args.tag}_with_predictions_real.json",
            cif_dir=f"generated_cifs_{s}_{args.tag}",
        )
        for s in sets
    ]

    ensure_dir(args.out_dir)
    manifest_rows = []
    all_json_rows = []
    per_set_counts = {}

    for cfg in cfgs:
        if not (os.path.exists(cfg.candidates_csv) and os.path.exists(cfg.pred_json) and os.path.isdir(cfg.cif_dir)):
            print(f"[skip] missing inputs for {cfg.name}")
            per_set_counts[cfg.name] = 0
            continue

        candidates = read_csv(cfg.candidates_csv)
        pred_by_id = load_set(cfg)
        picked = choose_rows(candidates, top_n=int(args.top_n_per_set), require_novel=bool(args.require_novel))

        set_dir = os.path.join(args.out_dir, cfg.name)
        cif_out_dir = os.path.join(set_dir, "cifs")
        poscar_out_dir = os.path.join(set_dir, "poscars")
        json_out_dir = os.path.join(set_dir, "json")
        ensure_dir(cif_out_dir)
        ensure_dir(poscar_out_dir)
        ensure_dir(json_out_dir)

        count = 0
        for idx, row in enumerate(picked, start=1):
            mid = str(row.get("material_id", ""))
            if mid not in pred_by_id:
                continue
            src_cif = os.path.join(cfg.cif_dir, f"{mid}.cif")
            if not os.path.exists(src_cif):
                continue

            prefix = f"{cfg.name}_{idx:03d}_{mid}"
            dst_cif = os.path.join(cif_out_dir, f"{prefix}.cif")
            dst_poscar = os.path.join(poscar_out_dir, f"{prefix}.vasp")
            dst_json = os.path.join(json_out_dir, f"{prefix}.json")

            shutil.copy2(src_cif, dst_cif)
            try:
                s = Structure.from_file(src_cif)
                s.to(fmt="poscar", filename=dst_poscar)
            except Exception:
                dst_poscar = ""

            rec = pred_by_id[mid]
            with open(dst_json, "w", encoding="utf-8") as f:
                json.dump(rec, f, indent=2)

            out_row = {
                "set": cfg.name,
                "rank_in_set": idx,
                "material_id": mid,
                "reduced_formula": row.get("reduced_formula"),
                "quality_score": safe_float(row.get("quality_score"), 0.0),
                "strict_pass": row.get("strict_pass"),
                "formula_novel_vs_train": row.get("formula_novel_vs_train"),
                "spacegroup_symbol": row.get("spacegroup_symbol"),
                "spacegroup_number": row.get("spacegroup_number"),
                "crystal_system": row.get("crystal_system"),
                "density": safe_float(row.get("density"), 0.0),
                "volume_per_atom": safe_float(row.get("volume_per_atom"), 0.0),
                "min_distance": safe_float(row.get("min_distance"), 0.0),
                "B_H": safe_float(row.get("B_H"), 0.0),
                "G_H": safe_float(row.get("G_H"), 0.0),
                "E_H": safe_float(row.get("E_H"), 0.0),
                "nu_H": safe_float(row.get("nu_H"), 0.0),
                "A_U": safe_float(row.get("A_U"), 0.0),
                "cif_path": dst_cif,
                "poscar_path": dst_poscar,
                "json_record_path": dst_json,
            }
            manifest_rows.append(out_row)
            all_json_rows.append(rec)
            count += 1

        per_set_counts[cfg.name] = count
        print(f"[{cfg.name}] shortlisted={count}")

    manifest_csv = os.path.join(args.out_dir, "shortlist_manifest.csv")
    manifest_json = os.path.join(args.out_dir, "shortlist_records.json")
    summary_json = os.path.join(args.out_dir, "shortlist_summary.json")
    write_manifest(manifest_rows, manifest_csv)
    with open(manifest_json, "w", encoding="utf-8") as f:
        json.dump(all_json_rows, f, indent=2)
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "tag": args.tag,
                "top_n_per_set": int(args.top_n_per_set),
                "require_novel": bool(args.require_novel),
                "counts": per_set_counts,
                "total": int(len(manifest_rows)),
                "manifest_csv": manifest_csv,
                "manifest_json": manifest_json,
            },
            f,
            indent=2,
        )

    print("=" * 72)
    print("DFT SHORTLIST PACKAGE COMPLETE")
    print("=" * 72)
    print(f"out_dir={args.out_dir}")
    print(f"total_shortlisted={len(manifest_rows)}")
    print(f"manifest={manifest_csv}")
    print(f"summary={summary_json}")


if __name__ == "__main__":
    main()

