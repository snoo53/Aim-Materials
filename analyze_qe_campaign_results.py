"""
Analyze Quantum ESPRESSO campaign outputs for elastic-tensor publication metrics.

Parses each candidate directory, extracting:
- relax/scf convergence + energies,
- stress-derived elastic tensor from finite-strain SCF outputs,
- DFT-vs-ML elastic agreement metrics,
- optional MP hull distance (if API key available).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from pymatgen.core import Structure

try:
    from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
    from pymatgen.ext.matproj import MPRester
except Exception:
    PDEntry = None
    PhaseDiagram = None
    MPRester = None


RY_TO_EV = 13.605693009

VOIGT21_IDXS = [
    (0, 0),
    (0, 1),
    (0, 2),
    (0, 3),
    (0, 4),
    (0, 5),
    (1, 1),
    (1, 2),
    (1, 3),
    (1, 4),
    (1, 5),
    (2, 2),
    (2, 3),
    (2, 4),
    (2, 5),
    (3, 3),
    (3, 4),
    (3, 5),
    (4, 4),
    (4, 5),
    (5, 5),
]
PAIR_TO_IDX = {p: i for i, p in enumerate(VOIGT21_IDXS)}


def read_csv(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: str, rows: List[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    fields = list(rows[0].keys()) if rows else []
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def as_float(x, default: Optional[float] = None) -> Optional[float]:
    try:
        v = float(x)
        if np.isfinite(v):
            return float(v)
    except Exception:
        pass
    return default


def load_norm_stats(npz_path: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if not npz_path or not os.path.exists(npz_path):
        return None, None
    stats = np.load(npz_path)
    voigt_mean = stats["voigt_mean"] if "voigt_mean" in stats else None
    voigt_std = stats["voigt_std"] if "voigt_std" in stats else None
    return voigt_mean, voigt_std


def denorm_voigt(vec21: List[float], voigt_mean: Optional[np.ndarray], voigt_std: Optional[np.ndarray]) -> List[float]:
    arr = np.array(vec21, dtype=float)
    if voigt_mean is None or voigt_std is None:
        return arr.tolist()
    if arr.shape != voigt_mean.shape:
        return arr.tolist()
    std_safe = np.where(np.abs(voigt_std) < 1e-12, 1.0, voigt_std)
    return (arr * std_safe + voigt_mean).tolist()


def voigt21_to_c6(voigt21: List[float]) -> Optional[np.ndarray]:
    if not isinstance(voigt21, list) or len(voigt21) != 21:
        return None
    c = np.zeros((6, 6), dtype=float)
    for k, (i, j) in enumerate(VOIGT21_IDXS):
        v = float(voigt21[k])
        c[i, j] = v
        c[j, i] = v
    return c


def c6_to_voigt21(c6: np.ndarray) -> List[float]:
    cs = 0.5 * (c6 + c6.T)
    return [float(cs[i, j]) for i, j in VOIGT21_IDXS]


def safe_rel_err(pred: Optional[float], tgt: Optional[float], floor: float = 1e-8) -> Optional[float]:
    if pred is None or tgt is None:
        return None
    if not np.isfinite(pred) or not np.isfinite(tgt):
        return None
    return float(abs(pred - tgt) / max(abs(tgt), floor))


def fro_rel_err(a: np.ndarray, b: np.ndarray, floor: float = 1e-8) -> float:
    na = float(np.linalg.norm(a))
    return float(np.linalg.norm(a - b) / max(na, floor))


def _stats(vals: List[float]) -> Optional[dict]:
    x = [float(v) for v in vals if v is not None and np.isfinite(v)]
    if not x:
        return None
    a = np.array(x, dtype=float)
    return {"min": float(np.min(a)), "median": float(np.median(a)), "mean": float(np.mean(a)), "max": float(np.max(a))}


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

        denom_b = s11 + s22 + s33 + 2.0 * (s12 + s13 + s23)
        denom_g = 4.0 * (s11 + s22 + s33) - 4.0 * (s12 + s13 + s23) + 3.0 * (s44 + s55 + s66)
        br = 1.0 / denom_b if abs(denom_b) > 1e-12 else np.nan
        gr = 15.0 / denom_g if abs(denom_g) > 1e-12 else np.nan

        bh = 0.5 * (bv + br)
        gh = 0.5 * (gv + gr)
        eh = (9.0 * bh * gh / (3.0 * bh + gh)) if abs(3.0 * bh + gh) > 1e-12 else np.nan
        nu = ((3.0 * bh - 2.0 * gh) / (2.0 * (3.0 * bh + gh))) if abs(3.0 * bh + gh) > 1e-12 else np.nan
        au = (5.0 * gv / gr + bv / br - 6.0) if (abs(gr) > 1e-12 and abs(br) > 1e-12) else np.nan
        out.update(
            {
                "B_H": float(bh),
                "G_H": float(gh),
                "E_H": float(eh),
                "nu_H": float(nu),
                "A_U": float(au),
                "invertible": True,
            }
        )
    except np.linalg.LinAlgError:
        out.update({"B_H": None, "G_H": None, "E_H": None, "nu_H": None, "A_U": None, "invertible": False})
    return out


def try_load_predicted_voigt(
    metadata: dict,
    voigt_mean: Optional[np.ndarray],
    voigt_std: Optional[np.ndarray],
    pred_voigt_is_normalized: bool,
) -> Optional[np.ndarray]:
    rec_path = metadata.get("json_record_path")
    if not isinstance(rec_path, str) or not rec_path.strip():
        return None
    rec_path = os.path.normpath(rec_path)
    if not os.path.exists(rec_path):
        return None
    try:
        rec = json.load(open(rec_path, "r", encoding="utf-8"))
        v = rec.get("targets_voigt21")
        if not isinstance(v, list) or len(v) != 21:
            return None
        vv = [float(x) for x in v]
        if pred_voigt_is_normalized:
            vv = denorm_voigt(vv, voigt_mean, voigt_std)
        return voigt21_to_c6(vv)
    except Exception:
        return None


def _to_float(tok: str) -> Optional[float]:
    t = tok.replace("D", "E").replace("d", "E")
    try:
        return float(t)
    except Exception:
        return None


def parse_qe_output(path: str) -> dict:
    out = {
        "exists": False,
        "job_done": False,
        "scf_converged_msg": False,
        "final_energy_ry": None,
        "final_energy_ev_cell": None,
        "stress_kbar": None,
        "total_force_ry_bohr": None,
    }
    if not path or not os.path.exists(path):
        return out
    out["exists"] = True
    try:
        txt = open(path, "r", encoding="utf-8", errors="ignore").read()
    except Exception:
        return out

    out["job_done"] = ("JOB DONE" in txt) and ("convergence NOT achieved" not in txt)
    out["scf_converged_msg"] = ("convergence has been achieved" in txt) and ("convergence NOT achieved" not in txt)

    # Final total energy.
    e_matches = re.findall(r"!\s+total energy\s*=\s*([\-+0-9.EeDd]+)\s+Ry", txt)
    if e_matches:
        e_ry = _to_float(e_matches[-1])
        if e_ry is not None:
            out["final_energy_ry"] = float(e_ry)
            out["final_energy_ev_cell"] = float(e_ry * RY_TO_EV)

    # Total force.
    f_matches = re.findall(r"Total force\s*=\s*([\-+0-9.EeDd]+)", txt)
    if f_matches:
        tf = _to_float(f_matches[-1])
        if tf is not None:
            out["total_force_ry_bohr"] = float(tf)

    # Last stress block in kbar.
    lines = txt.splitlines()
    idxs = [i for i, ln in enumerate(lines) if ("total   stress" in ln or "total stress" in ln)]
    for idx in reversed(idxs):
        rows = []
        for j in range(idx + 1, min(len(lines), idx + 12)):
            nums = re.findall(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[EeDd][-+]?\d+)?", lines[j])
            vals = [v for v in (_to_float(x) for x in nums) if v is not None]
            if len(vals) >= 6:
                rows.append(vals[-3:])
            elif len(vals) >= 3:
                rows.append(vals[:3])
            if len(rows) == 3:
                break
        if len(rows) == 3:
            out["stress_kbar"] = np.array(rows, dtype=float).tolist()
            break
    return out


def fit_elastic_from_strain_stress(points: List[Tuple[np.ndarray, np.ndarray]]) -> Tuple[Optional[np.ndarray], Optional[float]]:
    """
    Fit symmetric 6x6 C matrix (engineering-shear Voigt) from (eps6, sigma6) pairs.
    """
    if not points:
        return None, None
    a_rows = []
    b_rows = []
    for eps6, sig6 in points:
        for i in range(6):
            row = np.zeros(21, dtype=float)
            for j in range(6):
                p = (i, j) if i <= j else (j, i)
                row[PAIR_TO_IDX[p]] += float(eps6[j])
            a_rows.append(row)
            b_rows.append(float(sig6[i]))

    A = np.array(a_rows, dtype=float)
    b = np.array(b_rows, dtype=float)
    if A.shape[0] < 21:
        return None, None

    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    c6 = np.zeros((6, 6), dtype=float)
    for k, (i, j) in enumerate(VOIGT21_IDXS):
        c6[i, j] = float(x[k])
        c6[j, i] = float(x[k])

    pred = A @ x
    rms = float(np.sqrt(np.mean((pred - b) ** 2)))
    return c6, rms


def try_mp_hull(
    mpr: Optional["MPRester"],
    structure: Optional[Structure],
    final_energy_ev_cell: Optional[float],
    cache: dict,
) -> Tuple[Optional[float], Optional[str]]:
    if mpr is None or structure is None or final_energy_ev_cell is None:
        return None, None
    if PDEntry is None or PhaseDiagram is None:
        return None, "pymatgen_phase_diagram_unavailable"
    try:
        elems = tuple(sorted({str(el) for el in structure.composition.elements}))
        if elems not in cache:
            cache[elems] = mpr.get_entries_in_chemsys(list(elems))
        entries = list(cache[elems])
        if not entries:
            return None, "no_mp_entries"
        cand = PDEntry(structure.composition, float(final_energy_ev_cell), name="candidate")
        pd = PhaseDiagram(entries + [cand])
        e_hull = float(pd.get_e_above_hull(cand))
        return e_hull, None
    except Exception as exc:
        return None, str(exc)


def analyze_row(
    row: dict,
    voigt_mean: Optional[np.ndarray],
    voigt_std: Optional[np.ndarray],
    pred_voigt_is_normalized: bool,
    eig_tol: float,
    fit_rms_tol: float,
    qe_stress_sign: float,
    mpr: Optional["MPRester"],
    mp_cache: dict,
    hull_tol: float,
) -> dict:
    out = dict(row)
    campaign_dir = os.path.normpath(str(row.get("campaign_dir", "")).strip())
    out["status"] = "pending"
    out["has_relax_output"] = False
    out["has_scf_output"] = False
    out["relax_converged"] = None
    out["scf_converged"] = None
    out["final_energy_ev_cell"] = None
    out["final_energy_ev_atom"] = None
    out["total_force_ry_bohr"] = None
    out["dft_has_elastic"] = False
    out["elastic_points_total"] = 0
    out["elastic_points_ok"] = 0
    out["elastic_fit_rms_gpa"] = None
    out["dft_min_eig"] = None
    out["dft_B_H"] = None
    out["dft_G_H"] = None
    out["dft_E_H"] = None
    out["dft_nu_H"] = None
    out["dft_A_U"] = None
    out["pass_pd"] = None
    out["pass_fit_rms"] = None
    out["relerr_B_H_vs_pred"] = None
    out["relerr_G_H_vs_pred"] = None
    out["relerr_E_H_vs_pred"] = None
    out["relerr_voigt21_fro"] = None
    out["mp_e_above_hull"] = None
    out["pass_hull"] = None
    out["mp_hull_error"] = None

    if not campaign_dir or not os.path.isdir(campaign_dir):
        out["status"] = "missing_campaign_dir"
        return out

    relax_out = os.path.join(campaign_dir, "01_relax", "qe_relax.out")
    scf_out = os.path.join(campaign_dir, "02_scf", "qe_scf.out")
    elastic_manifest = os.path.join(campaign_dir, "03_elastic", "strain_manifest.csv")
    meta_path = os.path.join(campaign_dir, "metadata.json")
    struct_path = os.path.join(campaign_dir, "structure.cif")

    metadata = {}
    if os.path.exists(meta_path):
        try:
            metadata = json.load(open(meta_path, "r", encoding="utf-8"))
        except Exception:
            metadata = {}

    structure = None
    if os.path.exists(struct_path):
        try:
            structure = Structure.from_file(struct_path)
        except Exception:
            structure = None

    pr = parse_qe_output(relax_out)
    ps = parse_qe_output(scf_out)
    out["has_relax_output"] = bool(pr.get("exists"))
    out["has_scf_output"] = bool(ps.get("exists"))
    out["relax_converged"] = bool(pr.get("job_done")) if pr.get("exists") else None
    out["scf_converged"] = bool(ps.get("job_done")) if ps.get("exists") else None

    e_cell = ps.get("final_energy_ev_cell")
    if e_cell is None:
        e_cell = pr.get("final_energy_ev_cell")
    out["final_energy_ev_cell"] = e_cell
    tf = ps.get("total_force_ry_bohr")
    if tf is None:
        tf = pr.get("total_force_ry_bohr")
    out["total_force_ry_bohr"] = tf

    if e_cell is not None and structure is not None and len(structure) > 0:
        out["final_energy_ev_atom"] = float(e_cell) / float(len(structure))

    points = []
    if os.path.exists(elastic_manifest):
        try:
            sm = read_csv(elastic_manifest)
        except Exception:
            sm = []
        out["elastic_points_total"] = int(len(sm))
        for r in sm:
            sid = str(r.get("strain_id", "")).strip()
            if not sid:
                continue
            op = os.path.join(campaign_dir, "03_elastic", f"strain_{sid}", "qe_scf.out")
            po = parse_qe_output(op)
            sk = po.get("stress_kbar")
            if sk is None:
                continue
            st = np.array(sk, dtype=float) * 0.1 * float(qe_stress_sign)  # kbar -> GPa (+ sign convention switch)
            eps = np.array(
                [
                    float(r.get("eps1", 0.0)),
                    float(r.get("eps2", 0.0)),
                    float(r.get("eps3", 0.0)),
                    float(r.get("eps4", 0.0)),
                    float(r.get("eps5", 0.0)),
                    float(r.get("eps6", 0.0)),
                ],
                dtype=float,
            )
            sigma = np.array(
                [st[0, 0], st[1, 1], st[2, 2], st[1, 2], st[0, 2], st[0, 1]],
                dtype=float,
            )
            points.append((eps, sigma))
        out["elastic_points_ok"] = int(len(points))

    c6_dft = None
    if len(points) >= 8:
        c6_dft, rms = fit_elastic_from_strain_stress(points)
        out["elastic_fit_rms_gpa"] = rms
        if c6_dft is not None:
            out["dft_has_elastic"] = True
            mm = mechanical_metrics(c6_dft)
            out["dft_min_eig"] = mm.get("min_eig")
            out["dft_B_H"] = mm.get("B_H")
            out["dft_G_H"] = mm.get("G_H")
            out["dft_E_H"] = mm.get("E_H")
            out["dft_nu_H"] = mm.get("nu_H")
            out["dft_A_U"] = mm.get("A_U")

    if out["dft_min_eig"] is not None:
        out["pass_pd"] = bool(float(out["dft_min_eig"]) > float(eig_tol))
    if out["elastic_fit_rms_gpa"] is not None:
        out["pass_fit_rms"] = bool(float(out["elastic_fit_rms_gpa"]) <= float(fit_rms_tol))

    pred_bh = as_float(row.get("B_H"))
    pred_gh = as_float(row.get("G_H"))
    pred_eh = as_float(row.get("E_H"))
    out["relerr_B_H_vs_pred"] = safe_rel_err(pred_bh, out["dft_B_H"])
    out["relerr_G_H_vs_pred"] = safe_rel_err(pred_gh, out["dft_G_H"])
    out["relerr_E_H_vs_pred"] = safe_rel_err(pred_eh, out["dft_E_H"])

    c6_pred = try_load_predicted_voigt(
        metadata=metadata,
        voigt_mean=voigt_mean,
        voigt_std=voigt_std,
        pred_voigt_is_normalized=bool(pred_voigt_is_normalized),
    )
    if c6_pred is not None and c6_dft is not None:
        out["relerr_voigt21_fro"] = fro_rel_err(c6_pred, c6_dft)

    e_hull, err = try_mp_hull(mpr=mpr, structure=structure, final_energy_ev_cell=out["final_energy_ev_cell"], cache=mp_cache)
    out["mp_e_above_hull"] = e_hull
    out["mp_hull_error"] = err
    if e_hull is not None:
        out["pass_hull"] = bool(float(e_hull) <= float(hull_tol))

    if out["dft_has_elastic"]:
        out["status"] = "elastic_ready"
    elif out["has_scf_output"] and out.get("scf_converged") is True:
        out["status"] = "scf_done"
    elif out["has_relax_output"] and out.get("relax_converged") is True:
        out["status"] = "relax_done"
    elif out["has_relax_output"] or out["has_scf_output"]:
        out["status"] = "output_present_not_converged"
    else:
        out["status"] = "pending"

    return out


def summarize(rows: List[dict]) -> dict:
    status_counts = Counter(str(r.get("status")) for r in rows)

    def rate(key: str, val=True) -> dict:
        known = [r.get(key) for r in rows if r.get(key) is not None]
        if not known:
            return {"count": 0, "total": 0, "rate": None}
        c = sum(1 for x in known if x == val)
        return {"count": int(c), "total": int(len(known)), "rate": float(c / len(known))}

    by_set = defaultdict(list)
    for r in rows:
        by_set[str(r.get("set", ""))].append(r)
    by_set_summary = {}
    for k, rr in by_set.items():
        by_set_summary[k] = {
            "n_total": len(rr),
            "status_counts": dict(Counter(str(x.get("status")) for x in rr)),
            "elastic_ready": sum(1 for x in rr if x.get("status") == "elastic_ready"),
        }

    summary = {
        "n_total": int(len(rows)),
        "status_counts": dict(status_counts),
        "has_relax_output": rate("has_relax_output", True),
        "has_scf_output": rate("has_scf_output", True),
        "dft_has_elastic": rate("dft_has_elastic", True),
        "pass_pd": rate("pass_pd", True),
        "pass_fit_rms": rate("pass_fit_rms", True),
        "pass_hull": rate("pass_hull", True),
        "elastic_points_ok": _stats([r.get("elastic_points_ok") for r in rows]),
        "elastic_fit_rms_gpa": _stats([r.get("elastic_fit_rms_gpa") for r in rows]),
        "relerr_B_H_vs_pred": _stats([r.get("relerr_B_H_vs_pred") for r in rows]),
        "relerr_G_H_vs_pred": _stats([r.get("relerr_G_H_vs_pred") for r in rows]),
        "relerr_E_H_vs_pred": _stats([r.get("relerr_E_H_vs_pred") for r in rows]),
        "relerr_voigt21_fro": _stats([r.get("relerr_voigt21_fro") for r in rows]),
        "mp_e_above_hull": _stats([r.get("mp_e_above_hull") for r in rows]),
        "by_set": by_set_summary,
    }
    return summary


def build_validated_top(rows: List[dict], require_hull: bool, topk: int) -> List[dict]:
    cand = []
    for r in rows:
        if r.get("status") != "elastic_ready":
            continue
        if r.get("pass_pd") is not True:
            continue
        if r.get("pass_fit_rms") is False:
            continue
        if require_hull and r.get("pass_hull") is not True:
            continue
        cand.append(r)

    def key(r):
        bh = r.get("relerr_B_H_vs_pred")
        gh = r.get("relerr_G_H_vs_pred")
        eh = r.get("relerr_E_H_vs_pred")
        tt = r.get("relerr_voigt21_fro")
        rms = r.get("elastic_fit_rms_gpa")
        ehull = r.get("mp_e_above_hull")
        return (
            float(bh) if bh is not None else 1e9,
            float(gh) if gh is not None else 1e9,
            float(eh) if eh is not None else 1e9,
            float(tt) if tt is not None else 1e9,
            float(rms) if rms is not None else 1e9,
            float(ehull) if ehull is not None else (0.0 if not require_hull else 1e9),
        )

    cand = sorted(cand, key=key)
    out = []
    for i, r in enumerate(cand[: int(topk)], start=1):
        out.append(
            {
                "rank": i,
                "set": r.get("set"),
                "material_id": r.get("material_id"),
                "reduced_formula": r.get("reduced_formula"),
                "campaign_dir": r.get("campaign_dir"),
                "dft_B_H": r.get("dft_B_H"),
                "dft_G_H": r.get("dft_G_H"),
                "dft_E_H": r.get("dft_E_H"),
                "dft_min_eig": r.get("dft_min_eig"),
                "elastic_fit_rms_gpa": r.get("elastic_fit_rms_gpa"),
                "relerr_B_H_vs_pred": r.get("relerr_B_H_vs_pred"),
                "relerr_G_H_vs_pred": r.get("relerr_G_H_vs_pred"),
                "relerr_E_H_vs_pred": r.get("relerr_E_H_vs_pred"),
                "relerr_voigt21_fro": r.get("relerr_voigt21_fro"),
                "mp_e_above_hull": r.get("mp_e_above_hull"),
            }
        )
    return out


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign_manifest", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_summary_json", required=True)
    ap.add_argument("--out_validated_csv", default="")
    ap.add_argument("--norm_stats_npz", default="normalization_stats.npz")
    ap.add_argument("--pred_voigt_is_normalized", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--eig_tol", type=float, default=1e-6)
    ap.add_argument("--fit_rms_tol", type=float, default=5.0, help="Elastic fit RMS threshold in GPa.")
    ap.add_argument(
        "--qe_stress_sign",
        type=float,
        default=-1.0,
        help="Multiplier for QE stress before fitting (default -1 converts QE compressive-positive to conventional sign).",
    )
    ap.add_argument("--query_mp_hull", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--mp_api_key_env", default="MP_API_KEY")
    ap.add_argument("--hull_tol", type=float, default=0.1)
    ap.add_argument("--require_hull_for_validated", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--validated_topk", type=int, default=50)
    return ap.parse_args()


def main():
    args = parse_args()
    rows = read_csv(args.campaign_manifest)
    if not rows:
        raise RuntimeError(f"No rows found in {args.campaign_manifest}")

    voigt_mean, voigt_std = load_norm_stats(args.norm_stats_npz)

    mpr = None
    mp_cache = {}
    if bool(args.query_mp_hull):
        if MPRester is None:
            print("[warn] MPRester unavailable; skipping hull query.")
        else:
            key = os.getenv(args.mp_api_key_env, "").strip()
            if not key:
                print(f"[warn] env {args.mp_api_key_env} is empty; skipping hull query.")
            else:
                try:
                    mpr = MPRester(key)
                    print("[info] MP hull query enabled.")
                except Exception as exc:
                    print(f"[warn] failed to initialize MPRester: {exc}")
                    mpr = None

    out_rows = []
    for i, r in enumerate(rows, start=1):
        rec = analyze_row(
            row=r,
            voigt_mean=voigt_mean,
            voigt_std=voigt_std,
            pred_voigt_is_normalized=bool(args.pred_voigt_is_normalized),
            eig_tol=float(args.eig_tol),
            fit_rms_tol=float(args.fit_rms_tol),
            qe_stress_sign=float(args.qe_stress_sign),
            mpr=mpr,
            mp_cache=mp_cache,
            hull_tol=float(args.hull_tol),
        )
        out_rows.append(rec)
        if i == 1 or i % 20 == 0:
            print(f"[{i}/{len(rows)}] status={rec.get('status')} id={rec.get('material_id')}")

    write_csv(args.out_csv, out_rows)
    summary = summarize(out_rows)
    with open(args.out_summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if args.out_validated_csv:
        validated = build_validated_top(
            out_rows,
            require_hull=bool(args.require_hull_for_validated),
            topk=int(args.validated_topk),
        )
        write_csv(args.out_validated_csv, validated) if validated else write_csv(args.out_validated_csv, [])

    print("=" * 72)
    print("QE CAMPAIGN ANALYSIS COMPLETE")
    print("=" * 72)
    print(f"in_manifest={args.campaign_manifest}")
    print(f"out_csv={args.out_csv}")
    print(f"out_summary={args.out_summary_json}")
    if args.out_validated_csv:
        print(f"out_validated={args.out_validated_csv}")
    print(f"status_counts={summary.get('status_counts')}")


if __name__ == "__main__":
    main()
