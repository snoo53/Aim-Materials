"""
Deep validation pipeline for generated materials:
- Structural plausibility from CIF geometry
- Mechanical plausibility from predicted Voigt-21 tensors
- Scalar-vs-tensor consistency checks
- Novelty and duplicate proxies for publication readiness
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from pymatgen.core import Composition, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


VOIGT21_IDXS = [
    (0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5),
    (1, 1), (1, 2), (1, 3), (1, 4), (1, 5),
    (2, 2), (2, 3), (2, 4), (2, 5),
    (3, 3), (3, 4), (3, 5),
    (4, 4), (4, 5),
    (5, 5),
]

DEFAULT_DENSITY_RANGE = (0.5, 25.0)  # g/cc
DEFAULT_VOL_PER_ATOM_RANGE = (2.0, 80.0)  # A^3/atom
DEFAULT_MIN_DISTANCE = 1.2  # A
DEFAULT_CN_CUTOFF = 3.0  # A
DEFAULT_CN_MEAN_RANGE = (0.5, 20.0)
DEFAULT_CN_MAX = 32
DEFAULT_EIG_TOL = 1e-6
DEFAULT_RELERR_MAX = 0.50


@dataclass
class NormStats:
    scalar_mean: Optional[np.ndarray]
    scalar_std: Optional[np.ndarray]
    voigt_mean: Optional[np.ndarray]
    voigt_std: Optional[np.ndarray]


def load_norm_stats(npz_path: str) -> NormStats:
    if not npz_path:
        return NormStats(None, None, None, None)
    stats = np.load(npz_path)
    scalar_mean = stats["scalar_mean"] if "scalar_mean" in stats else None
    scalar_std = stats["scalar_std"] if "scalar_std" in stats else None
    voigt_mean = stats["voigt_mean"] if "voigt_mean" in stats else None
    voigt_std = stats["voigt_std"] if "voigt_std" in stats else None
    return NormStats(scalar_mean, scalar_std, voigt_mean, voigt_std)


def _denorm_vec(x: List[float], mean: Optional[np.ndarray], std: Optional[np.ndarray]) -> Optional[List[float]]:
    if x is None:
        return None
    arr = np.array(x, dtype=float)
    if mean is None or std is None:
        return arr.tolist()

    mean = np.array(mean, dtype=float)
    std = np.array(std, dtype=float)
    std_safe = np.where(np.abs(std) < 1e-12, 1.0, std)

    if mean.ndim == 0 and std.ndim == 0:
        out = arr * float(std_safe) + float(mean)
    else:
        if arr.shape != mean.shape:
            return arr.tolist()
        out = arr * std_safe + mean
    return out.tolist()


def voigt21_to_c6(voigt21: List[float]) -> Optional[np.ndarray]:
    if not isinstance(voigt21, list) or len(voigt21) != 21:
        return None
    c = np.zeros((6, 6), dtype=float)
    for k, (i, j) in enumerate(VOIGT21_IDXS):
        v = float(voigt21[k])
        c[i, j] = v
        c[j, i] = v
    return c


def born_checks(c: np.ndarray, crystal_system: str) -> Tuple[Optional[bool], List[str]]:
    cs = (crystal_system or "").strip().lower()
    c11, c22, c33 = c[0, 0], c[1, 1], c[2, 2]
    c12, c13, c23 = c[0, 1], c[0, 2], c[1, 2]
    c44, c55, c66 = c[3, 3], c[4, 4], c[5, 5]

    checks = []
    if cs == "cubic":
        checks = [
            ("C11-C12>0", c11 - c12 > 0.0),
            ("C11+2C12>0", c11 + 2.0 * c12 > 0.0),
            ("C44>0", c44 > 0.0),
        ]
    elif cs in ("hexagonal", "trigonal"):
        checks = [
            ("C11-|C12|>0", c11 - abs(c12) > 0.0),
            ("2C13^2<C33(C11+C12)", 2.0 * c13 * c13 < c33 * (c11 + c12)),
            ("C44>0", c44 > 0.0),
        ]
    elif cs == "tetragonal":
        checks = [
            ("C11-|C12|>0", c11 - abs(c12) > 0.0),
            ("2C13^2<C33(C11+C12)", 2.0 * c13 * c13 < c33 * (c11 + c12)),
            ("C44>0", c44 > 0.0),
            ("C66>0", c66 > 0.0),
        ]
    elif cs == "orthorhombic":
        checks = [
            ("C11>0", c11 > 0.0),
            ("C22>0", c22 > 0.0),
            ("C33>0", c33 > 0.0),
            ("C44>0", c44 > 0.0),
            ("C55>0", c55 > 0.0),
            ("C66>0", c66 > 0.0),
            ("C11+C22-2C12>0", c11 + c22 - 2.0 * c12 > 0.0),
            ("C11+C33-2C13>0", c11 + c33 - 2.0 * c13 > 0.0),
            ("C22+C33-2C23>0", c22 + c33 - 2.0 * c23 > 0.0),
            ("C11+C22+C33+2(C12+C13+C23)>0", c11 + c22 + c33 + 2.0 * (c12 + c13 + c23) > 0.0),
        ]
    else:
        return None, []

    failed = [name for name, ok in checks if not ok]
    return len(failed) == 0, failed


def mechanical_metrics(c6: np.ndarray) -> Dict[str, Optional[float]]:
    csym = 0.5 * (c6 + c6.T)
    eigvals = np.linalg.eigvalsh(csym)
    min_eig = float(np.min(eigvals))

    out: Dict[str, Optional[float]] = {
        "min_eig": min_eig,
        "c11": float(csym[0, 0]),
        "c22": float(csym[1, 1]),
        "c33": float(csym[2, 2]),
        "c44": float(csym[3, 3]),
        "c55": float(csym[4, 4]),
        "c66": float(csym[5, 5]),
    }

    c11, c22, c33 = csym[0, 0], csym[1, 1], csym[2, 2]
    c12, c13, c23 = csym[0, 1], csym[0, 2], csym[1, 2]
    c44, c55, c66 = csym[3, 3], csym[4, 4], csym[5, 5]

    bv = (c11 + c22 + c33 + 2.0 * (c12 + c13 + c23)) / 9.0
    gv = (c11 + c22 + c33 - (c12 + c13 + c23) + 3.0 * (c44 + c55 + c66)) / 15.0
    out["B_V"] = float(bv)
    out["G_V"] = float(gv)

    try:
        s = np.linalg.inv(csym)
        s11, s22, s33 = s[0, 0], s[1, 1], s[2, 2]
        s12, s13, s23 = s[0, 1], s[0, 2], s[1, 2]
        s44, s55, s66 = s[3, 3], s[4, 4], s[5, 5]

        denom_b = (s11 + s22 + s33 + 2.0 * (s12 + s13 + s23))
        denom_g = (4.0 * (s11 + s22 + s33) - 4.0 * (s12 + s13 + s23) + 3.0 * (s44 + s55 + s66))

        br = 1.0 / denom_b if abs(denom_b) > 1e-12 else np.nan
        gr = 15.0 / denom_g if abs(denom_g) > 1e-12 else np.nan
        bh = 0.5 * (bv + br)
        gh = 0.5 * (gv + gr)
        eh = (9.0 * bh * gh / (3.0 * bh + gh)) if abs(3.0 * bh + gh) > 1e-12 else np.nan
        nu = ((3.0 * bh - 2.0 * gh) / (2.0 * (3.0 * bh + gh))) if abs(3.0 * bh + gh) > 1e-12 else np.nan
        au = (5.0 * gv / gr + bv / br - 6.0) if (abs(gr) > 1e-12 and abs(br) > 1e-12) else np.nan

        out.update(
            {
                "B_R": float(br),
                "G_R": float(gr),
                "B_H": float(bh),
                "G_H": float(gh),
                "E_H": float(eh),
                "nu_H": float(nu),
                "A_U": float(au),
                "invertible": True,
            }
        )
    except np.linalg.LinAlgError:
        out.update(
            {
                "B_R": None,
                "G_R": None,
                "B_H": None,
                "G_H": None,
                "E_H": None,
                "nu_H": None,
                "A_U": None,
                "invertible": False,
            }
        )

    return out


def _reduced_formula_from_any(d: dict) -> Optional[str]:
    try:
        comp_red = d.get("composition_reduced")
        if isinstance(comp_red, dict) and comp_red:
            return Composition(comp_red).reduced_formula
    except Exception:
        pass
    try:
        if d.get("formula_pretty"):
            return Composition(d["formula_pretty"]).reduced_formula
    except Exception:
        pass
    return None


def load_training_formula_set(path: str) -> Optional[set]:
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    out = set()
    for d in rows:
        rf = _reduced_formula_from_any(d)
        if rf:
            out.add(rf)
    return out


def structure_metrics(struct: Structure, cn_cutoff: float) -> Dict[str, float]:
    dm = np.array(struct.distance_matrix, dtype=float)
    np.fill_diagonal(dm, np.inf)
    min_dist = float(np.min(dm)) if dm.size else float("inf")

    n = len(struct)
    # Simple geometric CN proxy at fixed cutoff (fast and deterministic).
    cn_list = [len(struct.get_neighbors(struct[i], cn_cutoff)) for i in range(n)]
    cn_mean = float(np.mean(cn_list)) if cn_list else 0.0
    cn_max = int(np.max(cn_list)) if cn_list else 0

    lat = struct.lattice
    vol = float(struct.volume)
    density = float(struct.density)
    vpa = vol / max(n, 1)

    return {
        "n_sites": int(n),
        "volume": vol,
        "density": density,
        "volume_per_atom": float(vpa),
        "min_distance": min_dist,
        "a": float(lat.a),
        "b": float(lat.b),
        "c": float(lat.c),
        "alpha": float(lat.alpha),
        "beta": float(lat.beta),
        "gamma": float(lat.gamma),
        "cn_mean_r3": cn_mean,
        "cn_max_r3": int(cn_max),
    }


def _symmetry_info(struct: Structure) -> Dict[str, Optional[object]]:
    try:
        sga = SpacegroupAnalyzer(struct, symprec=0.1, angle_tolerance=5.0)
        return {
            "spacegroup_symbol": sga.get_space_group_symbol(),
            "spacegroup_number": int(sga.get_space_group_number()),
            "crystal_system": sga.get_crystal_system(),
        }
    except Exception:
        return {
            "spacegroup_symbol": None,
            "spacegroup_number": None,
            "crystal_system": None,
        }


def _safe_rel_err(x: Optional[float], y: Optional[float], floor: float = 1e-6) -> Optional[float]:
    if x is None or y is None:
        return None
    if not np.isfinite(x) or not np.isfinite(y):
        return None
    return float(abs(x - y) / max(abs(y), floor))


def _composition_neutrality_guess(struct: Structure) -> Optional[bool]:
    # Conservative, reduced-composition check to avoid combinatorial blowups.
    try:
        guesses = struct.composition.reduced_composition.oxi_state_guesses(max_sites=8)
        return bool(len(guesses) > 0)
    except Exception:
        return None


def evaluate_dataset(
    rows: List[dict],
    cif_dir: str,
    norm: NormStats,
    train_formula_set: Optional[set],
    min_distance_threshold: float,
    density_range: Tuple[float, float],
    vpa_range: Tuple[float, float],
    cn_cutoff: float,
    cn_mean_min: float,
    cn_mean_max: float,
    cn_max_allowed: int,
    require_neutrality_guess: bool,
    consistency_relerr_max: float,
    eig_tol: float,
) -> List[dict]:
    results = []

    for m in rows:
        mid = m.get("material_id", "")
        cif_path = os.path.join(cif_dir, f"{mid}.cif") if cif_dir else ""

        rec = {
            "material_id": mid,
            "has_cif": bool(cif_path and os.path.exists(cif_path)),
            "parse_ok": False,
            "reduced_formula": None,
            "nelements": m.get("nelements"),
        }

        struct = None
        if rec["has_cif"]:
            try:
                struct = Structure.from_file(cif_path)
                rec["parse_ok"] = True
            except Exception as exc:
                rec["parse_error"] = str(exc)
        else:
            rec["parse_error"] = "missing_cif"

        if struct is not None:
            sm = structure_metrics(struct, cn_cutoff=cn_cutoff)
            rec.update(sm)
            rec["reduced_formula"] = struct.composition.reduced_formula
            rec["formula_pretty"] = struct.composition.formula
            rec["composition_neutrality_guess"] = _composition_neutrality_guess(struct)
            if require_neutrality_guess:
                rec["pass_neutrality_guess"] = bool(rec["composition_neutrality_guess"] is True)
            else:
                rec["pass_neutrality_guess"] = True
            rec.update(_symmetry_info(struct))

            rec["pass_min_distance"] = bool(rec["min_distance"] >= min_distance_threshold)
            rec["pass_density_range"] = bool(density_range[0] <= rec["density"] <= density_range[1])
            rec["pass_vpa_range"] = bool(vpa_range[0] <= rec["volume_per_atom"] <= vpa_range[1])
            rec["pass_cn_reasonable"] = bool(
                cn_mean_min <= rec["cn_mean_r3"] <= cn_mean_max and rec["cn_max_r3"] <= int(cn_max_allowed)
            )
        else:
            rec["composition_neutrality_guess"] = None
            rec["pass_neutrality_guess"] = False if require_neutrality_guess else True
            rec["spacegroup_symbol"] = None
            rec["spacegroup_number"] = None
            rec["crystal_system"] = m.get("crystal_system")
            rec["pass_min_distance"] = False
            rec["pass_density_range"] = False
            rec["pass_vpa_range"] = False
            rec["pass_cn_reasonable"] = False

        # Scalars and Voigt are expected normalized for this model; denorm if stats provided.
        scalars_denorm = _denorm_vec(m.get("targets_scalars"), norm.scalar_mean, norm.scalar_std)
        voigt_denorm = _denorm_vec(m.get("targets_voigt21"), norm.voigt_mean, norm.voigt_std)
        rec["targets_scalars_denorm"] = scalars_denorm

        c6 = voigt21_to_c6(voigt_denorm if voigt_denorm is not None else [])
        if c6 is None:
            rec["has_voigt21"] = False
            rec["is_pd"] = False
            rec["born_pass"] = None
            rec["born_failed_checks"] = []
            rec["mechanics_plausible"] = False
        else:
            rec["has_voigt21"] = True
            mm = mechanical_metrics(c6)
            rec.update(mm)
            rec["is_pd"] = bool(mm["min_eig"] is not None and mm["min_eig"] > eig_tol)
            born_ok, born_failed = born_checks(0.5 * (c6 + c6.T), rec.get("crystal_system"))
            rec["born_pass"] = born_ok
            rec["born_failed_checks"] = born_failed

            bh = mm.get("B_H")
            gh = mm.get("G_H")
            eh = mm.get("E_H")
            nuh = mm.get("nu_H")
            au = mm.get("A_U")

            plausible = True
            if bh is None or gh is None or eh is None:
                plausible = False
            else:
                plausible = plausible and (bh > 0.0 and gh > 0.0 and eh > 0.0)
                plausible = plausible and (-0.2 <= nuh <= 0.5 if nuh is not None and np.isfinite(nuh) else False)
                plausible = plausible and (au >= -1e-5 if au is not None and np.isfinite(au) else False)
                plausible = plausible and (rec["c11"] > 0 and rec["c22"] > 0 and rec["c33"] > 0)
                plausible = plausible and (rec["c44"] > 0 and rec["c55"] > 0 and rec["c66"] > 0)
            rec["mechanics_plausible"] = bool(plausible)

            # Consistency with scalar targets [bulk, shear, young, ...]
            bulk_s = scalars_denorm[0] if scalars_denorm and len(scalars_denorm) > 0 else None
            shear_s = scalars_denorm[1] if scalars_denorm and len(scalars_denorm) > 1 else None
            young_s = scalars_denorm[2] if scalars_denorm and len(scalars_denorm) > 2 else None

            rec["relerr_bulk_scalar_vs_BH"] = _safe_rel_err(bulk_s, bh)
            rec["relerr_shear_scalar_vs_GH"] = _safe_rel_err(shear_s, gh)
            rec["relerr_young_scalar_vs_EH"] = _safe_rel_err(young_s, eh)

            consistency = True
            for key in ("relerr_bulk_scalar_vs_BH", "relerr_shear_scalar_vs_GH", "relerr_young_scalar_vs_EH"):
                v = rec.get(key)
                if v is None or not np.isfinite(v):
                    consistency = False
                    break
                if v > float(consistency_relerr_max):
                    consistency = False
                    break
            rec["scalar_tensor_consistent"] = consistency

        rf = rec.get("reduced_formula")
        if rf and train_formula_set is not None:
            rec["formula_novel_vs_train"] = bool(rf not in train_formula_set)
        else:
            rec["formula_novel_vs_train"] = None

        # Publication-oriented strict pass flag.
        born_ok = rec["born_pass"]
        born_gate = (born_ok is True) if born_ok is not None else True
        rec["strict_pass"] = bool(
            rec["parse_ok"]
            and rec["pass_min_distance"]
            and rec["pass_density_range"]
            and rec["pass_vpa_range"]
            and rec["pass_cn_reasonable"]
            and rec["pass_neutrality_guess"]
            and rec["is_pd"]
            and born_gate
            and rec["mechanics_plausible"]
            and rec.get("scalar_tensor_consistent", False)
        )

        # Heuristic quality score for top-k triage.
        score = 0
        score += int(rec["parse_ok"])
        score += int(rec["pass_min_distance"])
        score += int(rec["pass_density_range"])
        score += int(rec["pass_vpa_range"])
        score += int(rec["pass_cn_reasonable"])
        score += int(rec["pass_neutrality_guess"])
        score += int(rec.get("is_pd", False))
        score += int(rec.get("mechanics_plausible", False))
        score += int(rec.get("scalar_tensor_consistent", False))
        if rec.get("formula_novel_vs_train") is True:
            score += 1
        rec["quality_score"] = int(score)

        results.append(rec)

    # Duplicate proxy using formula + rounded lattice + coarse distance fingerprint.
    fp_counter = Counter()
    for rec in results:
        if not rec.get("parse_ok"):
            continue
        fp = (
            rec.get("reduced_formula"),
            round(rec.get("a", 0.0), 2),
            round(rec.get("b", 0.0), 2),
            round(rec.get("c", 0.0), 2),
            round(rec.get("alpha", 0.0), 1),
            round(rec.get("beta", 0.0), 1),
            round(rec.get("gamma", 0.0), 1),
            round(rec.get("volume_per_atom", 0.0), 2),
        )
        fp_counter[fp] += 1

    for rec in results:
        if not rec.get("parse_ok"):
            rec["duplicate_count_proxy"] = 0
            rec["is_duplicate_proxy"] = False
            continue
        fp = (
            rec.get("reduced_formula"),
            round(rec.get("a", 0.0), 2),
            round(rec.get("b", 0.0), 2),
            round(rec.get("c", 0.0), 2),
            round(rec.get("alpha", 0.0), 1),
            round(rec.get("beta", 0.0), 1),
            round(rec.get("gamma", 0.0), 1),
            round(rec.get("volume_per_atom", 0.0), 2),
        )
        ndup = int(fp_counter[fp])
        rec["duplicate_count_proxy"] = ndup
        rec["is_duplicate_proxy"] = bool(ndup > 1)

    return results


def summarize(results: List[dict]) -> dict:
    n = len(results)
    def _rate(key: str, cond=lambda x: bool(x)):
        vals = [r.get(key) for r in results]
        known = [v for v in vals if v is not None]
        if not known:
            return {"count": 0, "total": 0, "rate": None}
        c = sum(1 for v in known if cond(v))
        return {"count": int(c), "total": int(len(known)), "rate": float(c / len(known))}

    def _num_stats(key: str):
        vals = [r.get(key) for r in results if r.get(key) is not None]
        vals = [float(v) for v in vals if np.isfinite(v)]
        if not vals:
            return None
        arr = np.array(vals, dtype=float)
        return {
            "min": float(np.min(arr)),
            "median": float(np.median(arr)),
            "mean": float(np.mean(arr)),
            "max": float(np.max(arr)),
        }

    summary = {
        "n_total": n,
        "parse_ok": _rate("parse_ok"),
        "pass_min_distance": _rate("pass_min_distance"),
        "pass_density_range": _rate("pass_density_range"),
        "pass_vpa_range": _rate("pass_vpa_range"),
        "pass_cn_reasonable": _rate("pass_cn_reasonable"),
        "pass_neutrality_guess": _rate("pass_neutrality_guess"),
        "is_pd": _rate("is_pd"),
        "born_pass": _rate("born_pass"),
        "mechanics_plausible": _rate("mechanics_plausible"),
        "scalar_tensor_consistent": _rate("scalar_tensor_consistent"),
        "strict_pass": _rate("strict_pass"),
        "formula_novel_vs_train": _rate("formula_novel_vs_train"),
        "is_duplicate_proxy": _rate("is_duplicate_proxy"),
    }

    score_vals = [r.get("quality_score", 0) for r in results]
    if score_vals:
        summary["quality_score_mean"] = float(np.mean(score_vals))
        summary["quality_score_median"] = float(np.median(score_vals))
        summary["quality_score_min"] = int(np.min(score_vals))
        summary["quality_score_max"] = int(np.max(score_vals))

    crystal_counts = Counter(str(r.get("crystal_system")) for r in results if r.get("crystal_system") is not None)
    summary["crystal_system_counts"] = dict(crystal_counts)
    sg_counts = Counter(str(r.get("spacegroup_symbol")) for r in results if r.get("spacegroup_symbol") is not None)
    summary["spacegroup_symbol_counts"] = dict(sg_counts)

    summary["numeric_stats"] = {
        "density": _num_stats("density"),
        "min_distance": _num_stats("min_distance"),
        "volume_per_atom": _num_stats("volume_per_atom"),
        "min_eig": _num_stats("min_eig"),
        "B_H": _num_stats("B_H"),
        "G_H": _num_stats("G_H"),
        "E_H": _num_stats("E_H"),
        "nu_H": _num_stats("nu_H"),
        "A_U": _num_stats("A_U"),
        "relerr_bulk_scalar_vs_BH": _num_stats("relerr_bulk_scalar_vs_BH"),
        "relerr_shear_scalar_vs_GH": _num_stats("relerr_shear_scalar_vs_GH"),
        "relerr_young_scalar_vs_EH": _num_stats("relerr_young_scalar_vs_EH"),
    }

    fail_counter = Counter()
    fail_keys = [
        "parse_ok",
        "pass_min_distance",
        "pass_density_range",
        "pass_vpa_range",
        "pass_cn_reasonable",
        "pass_neutrality_guess",
        "is_pd",
        "mechanics_plausible",
        "scalar_tensor_consistent",
    ]
    for r in results:
        if r.get("strict_pass"):
            continue
        reasons = [k for k in fail_keys if r.get(k) is False]
        fail_counter["|".join(reasons) if reasons else "unknown"] += 1
    summary["strict_fail_reason_counts"] = dict(fail_counter)

    rounded_a = [round(float(r.get("a")), 3) for r in results if r.get("a") is not None and np.isfinite(r.get("a"))]
    rounded_b = [round(float(r.get("b")), 3) for r in results if r.get("b") is not None and np.isfinite(r.get("b"))]
    rounded_c = [round(float(r.get("c")), 3) for r in results if r.get("c") is not None and np.isfinite(r.get("c"))]
    rounded_vpa = [round(float(r.get("volume_per_atom")), 4) for r in results if r.get("volume_per_atom") is not None and np.isfinite(r.get("volume_per_atom"))]
    summary["lattice_diversity_proxy"] = {
        "unique_a_rounded_1e-3": int(len(set(rounded_a))),
        "unique_b_rounded_1e-3": int(len(set(rounded_b))),
        "unique_c_rounded_1e-3": int(len(set(rounded_c))),
        "unique_vpa_rounded_1e-4": int(len(set(rounded_vpa))),
    }
    return summary


def write_topk_csv(results: List[dict], out_csv: str, k: int = 50):
    cols = [
        "material_id",
        "reduced_formula",
        "crystal_system",
        "spacegroup_symbol",
        "spacegroup_number",
        "quality_score",
        "strict_pass",
        "formula_novel_vs_train",
        "is_duplicate_proxy",
        "min_distance",
        "density",
        "volume_per_atom",
        "pass_neutrality_guess",
        "cn_mean_r3",
        "cn_max_r3",
        "min_eig",
        "B_H",
        "G_H",
        "E_H",
        "nu_H",
        "A_U",
        "relerr_bulk_scalar_vs_BH",
        "relerr_shear_scalar_vs_GH",
        "relerr_young_scalar_vs_EH",
    ]
    sorted_rows = sorted(
        results,
        key=lambda r: (
            int(bool(r.get("strict_pass"))),
            int(r.get("quality_score", 0)),
            float(r.get("min_eig", -1e9) if r.get("min_eig") is not None else -1e9),
            -float(r.get("relerr_bulk_scalar_vs_BH", 1e9) if r.get("relerr_bulk_scalar_vs_BH") is not None else 1e9),
        ),
        reverse=True,
    )
    top = sorted_rows[:k]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in top:
            w.writerow({c: r.get(c, None) for c in cols})


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_json", required=True, help="Generated materials JSON with predicted targets.")
    ap.add_argument("--cif_dir", required=True, help="Directory containing gen_XXXXX.cif files.")
    ap.add_argument("--out_json", required=True, help="Detailed per-material validation output JSON.")
    ap.add_argument("--out_summary_json", required=True, help="Summary output JSON.")
    ap.add_argument("--out_topk_csv", required=True, help="Top candidates CSV.")
    ap.add_argument("--norm_stats_npz", default="", help="normalization_stats.npz for de-normalization.")
    ap.add_argument("--train_summary_json", default="", help="mp_summary_filtered.json for novelty checks.")
    ap.add_argument("--min_distance", type=float, default=DEFAULT_MIN_DISTANCE)
    ap.add_argument("--density_min", type=float, default=DEFAULT_DENSITY_RANGE[0])
    ap.add_argument("--density_max", type=float, default=DEFAULT_DENSITY_RANGE[1])
    ap.add_argument("--vpa_min", type=float, default=DEFAULT_VOL_PER_ATOM_RANGE[0])
    ap.add_argument("--vpa_max", type=float, default=DEFAULT_VOL_PER_ATOM_RANGE[1])
    ap.add_argument("--cn_cutoff", type=float, default=DEFAULT_CN_CUTOFF)
    ap.add_argument("--cn_mean_min", type=float, default=DEFAULT_CN_MEAN_RANGE[0])
    ap.add_argument("--cn_mean_max", type=float, default=DEFAULT_CN_MEAN_RANGE[1])
    ap.add_argument("--cn_max", type=int, default=DEFAULT_CN_MAX)
    ap.add_argument(
        "--require_neutrality_guess",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require oxidation-state neutrality guess for strict pass.",
    )
    ap.add_argument(
        "--consistency_relerr_max",
        type=float,
        default=DEFAULT_RELERR_MAX,
        help="Max allowed relative error between scalar targets and tensor-derived BH/GH/EH.",
    )
    ap.add_argument("--eig_tol", type=float, default=DEFAULT_EIG_TOL)
    ap.add_argument("--topk", type=int, default=50)
    return ap.parse_args()


def main():
    args = parse_args()

    with open(args.in_json, "r", encoding="utf-8") as f:
        rows = json.load(f)

    norm = load_norm_stats(args.norm_stats_npz)
    train_set = load_training_formula_set(args.train_summary_json) if args.train_summary_json else None

    results = evaluate_dataset(
        rows=rows,
        cif_dir=args.cif_dir,
        norm=norm,
        train_formula_set=train_set,
        min_distance_threshold=float(args.min_distance),
        density_range=(float(args.density_min), float(args.density_max)),
        vpa_range=(float(args.vpa_min), float(args.vpa_max)),
        cn_cutoff=float(args.cn_cutoff),
        cn_mean_min=float(args.cn_mean_min),
        cn_mean_max=float(args.cn_mean_max),
        cn_max_allowed=int(args.cn_max),
        require_neutrality_guess=bool(args.require_neutrality_guess),
        consistency_relerr_max=float(args.consistency_relerr_max),
        eig_tol=float(args.eig_tol),
    )

    summary = summarize(results)
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True) if os.path.dirname(args.out_json) else None
    os.makedirs(os.path.dirname(args.out_summary_json), exist_ok=True) if os.path.dirname(args.out_summary_json) else None
    os.makedirs(os.path.dirname(args.out_topk_csv), exist_ok=True) if os.path.dirname(args.out_topk_csv) else None

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    with open(args.out_summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    write_topk_csv(results, args.out_topk_csv, k=int(args.topk))

    print("=" * 72)
    print("DEEP VALIDATION REPORT")
    print("=" * 72)
    print(f"Input: {args.in_json}")
    print(f"CIF dir: {args.cif_dir}")
    print(f"n_total: {summary['n_total']}")
    for key in [
        "parse_ok",
        "pass_min_distance",
        "pass_density_range",
        "pass_vpa_range",
        "pass_cn_reasonable",
        "pass_neutrality_guess",
        "is_pd",
        "born_pass",
        "mechanics_plausible",
        "scalar_tensor_consistent",
        "strict_pass",
        "formula_novel_vs_train",
        "is_duplicate_proxy",
    ]:
        s = summary[key]
        if s["rate"] is None:
            print(f"{key:28s}: N/A")
        else:
            print(f"{key:28s}: {s['count']}/{s['total']} ({100.0*s['rate']:.2f}%)")
    print(f"quality_score mean/median   : {summary.get('quality_score_mean', 0.0):.2f}/{summary.get('quality_score_median', 0.0):.2f}")
    print(f"Wrote detailed JSON         : {args.out_json}")
    print(f"Wrote summary JSON          : {args.out_summary_json}")
    print(f"Wrote top-K CSV             : {args.out_topk_csv}")


if __name__ == "__main__":
    main()
