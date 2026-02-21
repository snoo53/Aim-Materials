"""
Relax shortlisted generated structures with CHGNet and rebuild geometry fields.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np
from chgnet.model.dynamics import StructOptimizer
from pymatgen.core import Structure


def build_periodic_graph(struct: Structure, cutoff: float = 5.0, max_neighbors: int = 32):
    frac = np.array(struct.frac_coords, dtype=float)
    lat = np.array(struct.lattice.matrix, dtype=float)
    n = int(frac.shape[0])
    if n <= 0:
        return [[0], [0]], [[0.0]]

    diff = frac[:, None, :] - frac[None, :, :]
    diff = diff - np.round(diff)
    dc = np.einsum("ijk,kl->ijl", diff, lat)
    dist = np.linalg.norm(dc + 1e-12, axis=-1)

    edge_i: List[int] = []
    edge_j: List[int] = []
    edge_attr: List[List[float]] = []
    for i in range(n):
        order = np.argsort(dist[i])
        cnt = 0
        for j in order.tolist():
            if i == j:
                continue
            d = float(dist[i, j])
            if d > cutoff:
                break
            edge_i.append(int(i))
            edge_j.append(int(j))
            edge_attr.append([d])
            cnt += 1
            if cnt >= max_neighbors:
                break
    if not edge_i:
        return [[0], [0]], [[0.0]]
    return [edge_i, edge_j], edge_attr


def min_distance(struct: Structure) -> float:
    dm = np.array(struct.distance_matrix, dtype=float)
    np.fill_diagonal(dm, np.inf)
    return float(np.min(dm)) if dm.size else float("inf")


def load_candidates(path: str, target_set: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    out = [r for r in rows if str(r.get("set")) == str(target_set)]
    return out


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates_json", required=True, help="candidates_all_strict_novel_unique_top200.json")
    ap.add_argument("--target_set", required=True, choices=["2el", "3el", "4el"])
    ap.add_argument("--source_json", required=True, help="Source generated materials JSON (with predictions).")
    ap.add_argument("--default_cif_dir", required=True, help="Default CIF directory when candidate path is missing.")
    ap.add_argument("--out_json", required=True, help="Relaxed subset JSON output.")
    ap.add_argument("--out_cif_dir", required=True, help="Relaxed CIF output dir.")
    ap.add_argument("--out_meta_json", required=True, help="Per-candidate relaxation metadata.")
    ap.add_argument("--device", default="", help="CHGNet device: cuda/cpu/auto(empty).")
    ap.add_argument("--fmax", type=float, default=0.08)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--no_relax_cell", action="store_true")
    ap.add_argument("--edge_cutoff", type=float, default=5.0)
    ap.add_argument("--max_neighbors", type=int, default=32)
    return ap.parse_args()


def main():
    args = parse_args()
    candidates = load_candidates(args.candidates_json, args.target_set)
    source_rows = json.load(open(args.source_json, "r", encoding="utf-8"))
    source_by_id: Dict[str, dict] = {str(r.get("material_id")): r for r in source_rows}

    os.makedirs(args.out_cif_dir, exist_ok=True)

    use_device = args.device.strip() if args.device.strip() else None
    relaxer = StructOptimizer(use_device=use_device)

    out_rows = []
    meta_rows = []

    n_ok = 0
    n_fail = 0
    n_missing_src = 0
    n_missing_cif = 0

    for i, c in enumerate(candidates):
        mid = str(c.get("material_id"))
        rec = {"set": args.target_set, "material_id": mid}

        src = source_by_id.get(mid)
        if src is None:
            n_missing_src += 1
            rec["status"] = "missing_source_row"
            meta_rows.append(rec)
            continue

        cif_path = c.get("cif_path")
        if not isinstance(cif_path, str) or not os.path.exists(cif_path):
            cif_path = os.path.join(args.default_cif_dir, f"{mid}.cif")
        if not os.path.exists(cif_path):
            n_missing_cif += 1
            rec["status"] = "missing_cif"
            rec["cif_path"] = cif_path
            meta_rows.append(rec)
            continue

        try:
            s0 = Structure.from_file(cif_path)
            r = relaxer.relax(
                s0,
                fmax=float(args.fmax),
                steps=int(args.steps),
                relax_cell=not bool(args.no_relax_cell),
                verbose=False,
            )
            s1 = r["final_structure"]
        except Exception as exc:
            n_fail += 1
            rec["status"] = "relax_fail"
            rec["error"] = str(exc)
            rec["cif_path"] = cif_path
            meta_rows.append(rec)
            continue

        before_md = min_distance(s0)
        after_md = min_distance(s1)
        before_v = float(s0.volume)
        after_v = float(s1.volume)
        before_rho = float(s0.density)
        after_rho = float(s1.density)

        out_cif = os.path.join(args.out_cif_dir, f"{mid}.cif")
        s1.to(fmt="cif", filename=out_cif)

        row = dict(src)
        row["positions"] = np.array(s1.frac_coords, dtype=float).tolist()
        edge_index, edge_attr = build_periodic_graph(
            s1,
            cutoff=float(args.edge_cutoff),
            max_neighbors=int(args.max_neighbors),
        )
        row["edge_index"] = edge_index
        row["edge_attr"] = edge_attr
        row["num_nodes"] = int(len(s1))
        row["num_edges"] = int(len(edge_attr))
        row["num_atoms"] = int(len(s1))
        row["lattice_matrix"] = np.array(s1.lattice.matrix, dtype=float).tolist()
        row["lattice_abc"] = [float(s1.lattice.a), float(s1.lattice.b), float(s1.lattice.c)]
        row["lattice_angles"] = [float(s1.lattice.alpha), float(s1.lattice.beta), float(s1.lattice.gamma)]
        row["lattice_volume"] = float(s1.volume)
        row["estimated_density"] = float(s1.density)
        row["min_distance"] = float(after_md)
        out_rows.append(row)

        rec.update(
            {
                "status": "ok",
                "cif_path_in": cif_path,
                "cif_path_out": out_cif,
                "before_min_distance": before_md,
                "after_min_distance": after_md,
                "before_volume": before_v,
                "after_volume": after_v,
                "volume_ratio": after_v / max(before_v, 1e-12),
                "before_density": before_rho,
                "after_density": after_rho,
            }
        )
        meta_rows.append(rec)
        n_ok += 1

        if n_ok == 1 or n_ok % 10 == 0:
            print(
                f"[{n_ok}/{len(candidates)}] {mid} "
                f"min_d {before_md:.3f}->{after_md:.3f} A, "
                f"V {before_v:.1f}->{after_v:.1f} A^3, rho {before_rho:.2f}->{after_rho:.2f}"
            )

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out_rows, f)
    with open(args.out_meta_json, "w", encoding="utf-8") as f:
        json.dump(meta_rows, f, indent=2)

    print("=" * 72)
    print("CHGNET RELAXATION COMPLETE")
    print("=" * 72)
    print(f"target_set={args.target_set}")
    print(f"candidates_total={len(candidates)}")
    print(f"relaxed_ok={n_ok}")
    print(f"relax_fail={n_fail}")
    print(f"missing_source={n_missing_src}")
    print(f"missing_cif={n_missing_cif}")
    print(f"out_json={args.out_json}")
    print(f"out_meta_json={args.out_meta_json}")
    print(f"out_cif_dir={args.out_cif_dir}")


if __name__ == "__main__":
    main()

