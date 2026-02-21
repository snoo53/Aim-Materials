"""
Geometric post-relaxation fallback when ML force-field relaxers are unavailable.

For each generated structure:
- load CIF,
- iteratively isotropically expand cell when local coordination is too high,
- preserve fractional coordinates,
- rebuild periodic graph edges/distances for downstream featurization.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np
from pymatgen.core import Structure


def structure_metrics(struct: Structure, cn_cutoff: float = 3.0) -> Dict[str, float]:
    dm = np.array(struct.distance_matrix, dtype=float)
    np.fill_diagonal(dm, np.inf)
    min_dist = float(np.min(dm)) if dm.size else float("inf")
    n = len(struct)
    cn = [len(struct.get_neighbors(struct[i], cn_cutoff)) for i in range(n)]
    cn_mean = float(np.mean(cn)) if cn else 0.0
    cn_max = int(np.max(cn)) if cn else 0
    return {
        "n_sites": int(n),
        "min_dist": min_dist,
        "cn_mean": cn_mean,
        "cn_max": cn_max,
        "density": float(struct.density),
        "volume": float(struct.volume),
    }


def isotropic_scale_structure(struct: Structure, scale_factor: float) -> Structure:
    out = struct.copy()
    new_volume = float(out.volume) * float(scale_factor) ** 3
    out.scale_lattice(new_volume)
    return out


def choose_scale_factor(cn_mean: float, cn_lower: float, cn_upper: float) -> float:
    # Over-coordinated: expand
    if cn_mean > cn_upper:
        ratio = max(cn_mean / max(cn_upper, 1e-6), 1.0)
        return float(min(1.25, max(1.03, ratio ** 0.35)))
    # Under-coordinated: compress
    if cn_mean < cn_lower:
        if cn_mean <= 1e-8:
            return 0.90
        ratio = max(cn_lower / max(cn_mean, 1e-6), 1.0)
        return float(max(0.80, min(0.98, ratio ** (-0.30))))
    return 1.0


def relax_by_coordination(
    struct: Structure,
    cn_cutoff: float,
    cn_target_mean_min: float,
    cn_target_mean_max: float,
    cn_target_max: int,
    min_dist_target: float,
    density_floor: float,
    density_ceiling: float,
    max_iters: int,
) -> Tuple[Structure, Dict[str, float]]:
    cur = struct.copy()
    before = structure_metrics(cur, cn_cutoff=cn_cutoff)

    for _ in range(max_iters):
        m = structure_metrics(cur, cn_cutoff=cn_cutoff)
        ok_cn = (cn_target_mean_min <= m["cn_mean"] <= cn_target_mean_max) and (m["cn_max"] <= cn_target_max)
        ok_dist = m["min_dist"] >= min_dist_target
        ok_density = density_floor <= m["density"] <= density_ceiling
        if ok_cn and ok_dist and ok_density:
            break

        sf = choose_scale_factor(m["cn_mean"], cn_target_mean_min, cn_target_mean_max)
        # Always bias toward expansion when minimum distance is below target.
        if not ok_dist:
            sf = max(sf, 1.03)
        if sf <= 1.0001:
            sf = 1.03
        trial = isotropic_scale_structure(cur, sf)
        mt = structure_metrics(trial, cn_cutoff=cn_cutoff)
        if mt["density"] < density_floor or mt["density"] > density_ceiling:
            break
        cur = trial

    after = structure_metrics(cur, cn_cutoff=cn_cutoff)
    meta = {
        "before_cn_mean": before["cn_mean"],
        "before_cn_max": before["cn_max"],
        "before_density": before["density"],
        "before_min_dist": before["min_dist"],
        "before_volume": before["volume"],
        "after_cn_mean": after["cn_mean"],
        "after_cn_max": after["cn_max"],
        "after_density": after["density"],
        "after_min_dist": after["min_dist"],
        "after_volume": after["volume"],
        "scaled_volume_ratio": after["volume"] / max(before["volume"], 1e-9),
    }
    return cur, meta


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


def find_entry_map(rows: List[dict]) -> Dict[str, dict]:
    out = {}
    for r in rows:
        mid = r.get("material_id")
        if isinstance(mid, str):
            out[mid] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_json", required=True, help="Generated JSON to post-relax.")
    ap.add_argument("--in_cif_dir", required=True, help="Input CIF directory.")
    ap.add_argument("--out_json", required=True, help="Output JSON with relaxed geometry/graph.")
    ap.add_argument("--out_cif_dir", required=True, help="Output CIF directory.")
    ap.add_argument("--out_meta_json", default="", help="Optional per-material relaxation metadata.")
    ap.add_argument("--cn_cutoff", type=float, default=3.0)
    ap.add_argument("--cn_target_mean_min", type=float, default=0.5)
    ap.add_argument("--cn_target_mean_max", type=float, default=20.0)
    ap.add_argument("--cn_target_max", type=int, default=32)
    ap.add_argument("--min_distance", type=float, default=1.2)
    ap.add_argument("--density_floor", type=float, default=1.0)
    ap.add_argument("--density_ceiling", type=float, default=25.0)
    ap.add_argument("--max_iters", type=int, default=10)
    ap.add_argument("--edge_cutoff", type=float, default=5.0)
    ap.add_argument("--max_neighbors", type=int, default=32)
    args = ap.parse_args()

    with open(args.in_json, "r", encoding="utf-8") as f:
        rows = json.load(f)

    os.makedirs(args.out_cif_dir, exist_ok=True)
    meta_rows = []

    n_missing = 0
    n_parse_fail = 0
    n_adjusted = 0

    for r in rows:
        mid = r.get("material_id", "")
        cif_in = os.path.join(args.in_cif_dir, f"{mid}.cif")
        cif_out = os.path.join(args.out_cif_dir, f"{mid}.cif")
        rec = {"material_id": mid}

        if not os.path.exists(cif_in):
            n_missing += 1
            rec["status"] = "missing_cif"
            meta_rows.append(rec)
            continue

        try:
            s = Structure.from_file(cif_in)
        except Exception as exc:
            n_parse_fail += 1
            rec["status"] = "parse_fail"
            rec["error"] = str(exc)
            meta_rows.append(rec)
            continue

        s_rel, meta = relax_by_coordination(
            s,
            cn_cutoff=float(args.cn_cutoff),
            cn_target_mean_min=float(args.cn_target_mean_min),
            cn_target_mean_max=float(args.cn_target_mean_max),
            cn_target_max=int(args.cn_target_max),
            min_dist_target=float(args.min_distance),
            density_floor=float(args.density_floor),
            density_ceiling=float(args.density_ceiling),
            max_iters=int(args.max_iters),
        )
        if meta["scaled_volume_ratio"] > 1.0001:
            n_adjusted += 1
        elif meta["scaled_volume_ratio"] < 0.9999:
            n_adjusted += 1
        rec.update(meta)
        rec["status"] = "ok"
        meta_rows.append(rec)

        # Save relaxed CIF.
        s_rel.to(fmt="cif", filename=cif_out)

        # Sync JSON structure/graph fields to relaxed geometry.
        edge_index, edge_attr = build_periodic_graph(
            s_rel,
            cutoff=float(args.edge_cutoff),
            max_neighbors=int(args.max_neighbors),
        )
        r["positions"] = np.array(s_rel.frac_coords, dtype=float).tolist()
        r["edge_index"] = edge_index
        r["edge_attr"] = edge_attr
        r["num_nodes"] = int(len(s_rel))
        r["num_edges"] = int(len(edge_attr))
        r["num_atoms"] = int(len(s_rel))
        r["lattice_matrix"] = np.array(s_rel.lattice.matrix, dtype=float).tolist()
        r["lattice_abc"] = [float(s_rel.lattice.a), float(s_rel.lattice.b), float(s_rel.lattice.c)]
        r["lattice_angles"] = [float(s_rel.lattice.alpha), float(s_rel.lattice.beta), float(s_rel.lattice.gamma)]
        r["lattice_volume"] = float(s_rel.volume)
        r["estimated_density"] = float(s_rel.density)
        r["min_distance"] = float(meta["after_min_dist"])

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(rows, f)

    if args.out_meta_json:
        with open(args.out_meta_json, "w", encoding="utf-8") as f:
            json.dump(meta_rows, f, indent=2)

    print("=" * 72)
    print("GEOMETRIC POST-RELAXATION COMPLETE")
    print("=" * 72)
    print(f"input_json={args.in_json}")
    print(f"input_cif_dir={args.in_cif_dir}")
    print(f"output_json={args.out_json}")
    print(f"output_cif_dir={args.out_cif_dir}")
    if args.out_meta_json:
        print(f"output_meta={args.out_meta_json}")
    print(f"n_total={len(rows)}")
    print(f"n_adjusted={n_adjusted}")
    print(f"n_missing_cif={n_missing}")
    print(f"n_parse_fail={n_parse_fail}")


if __name__ == "__main__":
    main()
