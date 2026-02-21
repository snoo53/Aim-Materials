"""
Analyze VASP campaign outputs for elastic-tensor publication metrics.

Parses each candidate directory, extracting:
- convergence + final energy,
- force/stress proxies from vasprun,
- elastic tensor (from OUTCAR, if present),
- DFT-vs-ML elastic agreement metrics,
- optional MP hull distance (if API key available).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from pymatgen.core import Composition, Structure
from pymatgen.io.vasp.outputs import Outcar, Vasprun

try:
    from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
    from pymatgen.ext.matproj import MPRester
except Exception:
    PDEntry = None
    PhaseDiagram = None
    MPRester = None


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


def as_bool(x) -> bool:
    if isinstance(x, bool):
        return x
    if x is None:
        return False
    s = str(x).strip().lower()
    return s in {"1", "true", "t", "yes", "y"}


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
    out = []
    cs = 0.5 * (c6 + c6.T)
    for i, j in VOIGT21_IDXS:
        out.append(float(cs[i, j]))
    return out


def safe_rel_err(pred: Optional[float], tgt: Optional[float], floor: float = 1e-8) -> Optional[float]:
    if pred is None or tgt is None:
        return None
    if not np.isfinite(pred) or not np.isfinite(tgt):
        return None
    return float(abs(pred - tgt) / max(abs(tgt), floor))


def fro_rel_err(a: np.ndarray, b: np.ndarray, floor: float = 1e-8) -> float:
    na = float(np.linalg.norm(a))
    return float(np.linalg.norm(a - b) / max(na, floor))


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


def try_load_predicted_voigt(metadata: dict, voigt_mean: Optional[np.ndarray], voigt_std: Optional[np.ndarray], pred_voigt_is_normalized: bool) -> Optional[np.ndarray]:
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


def _stats(vals: List[float]) -> Optional[dict]:
    x = [float(v) for v in vals if v is not None and np.isfinite(v)]
    if not x:
        return None
    a = np.array(x, dtype=float)
    return {
        "min": float(np.min(a)),
        "median": float(np.median(a)),
        "mean": float(np.mean(a)),
        "max": float(np.max(a)),
    }


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
    force_tol: float,
    mpr: Optional["MPRester"],
    mp_cache: dict,
    hull_tol: float,
) -> dict:
    out = dict(row)
    campaign_dir = os.path.normpath(str(row.get("campaign_dir", "")).strip())
    out["status"] = "pending"
    out["has_outcar"] = False
    out["has_vasprun"] = False
    out["has_contcar"] = False
    out["converged"] = None
    out["converged_electronic"] = None
    out["converged_ionic"] = None
    out["final_energy_ev_cell"] = None
    out["final_energy_ev_atom"] = None
    out["dft_force_max"] = None
    out["dft_stress_fro"] = None
    out["dft_has_elastic"] = False
    out["dft_min_eig"] = None
    out["dft_B_H"] = None
    out["dft_G_H"] = None
    out["dft_E_H"] = None
    out["dft_nu_H"] = None
    out["dft_A_U"] = None
    out["pass_force_tol"] = None
    out["pass_pd"] = None
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

    outcar_path = os.path.join(campaign_dir, "OUTCAR")
    vasprun_path = os.path.join(campaign_dir, "vasprun.xml")
    contcar_path = os.path.join(campaign_dir, "CONTCAR")
    poscar_path = os.path.join(campaign_dir, "POSCAR")
    meta_path = os.path.join(campaign_dir, "metadata.json")

    out["has_outcar"] = bool(os.path.exists(outcar_path))
    out["has_vasprun"] = bool(os.path.exists(vasprun_path))
    out["has_contcar"] = bool(os.path.exists(contcar_path))

    structure = None
    for p in (contcar_path, poscar_path):
        if os.path.exists(p):
            try:
                structure = Structure.from_file(p)
                break
            except Exception:
                continue

    metadata = {}
    if os.path.exists(meta_path):
        try:
            metadata = json.load(open(meta_path, "r", encoding="utf-8"))
        except Exception:
            metadata = {}

    # Parse vasprun if available.
    if out["has_vasprun"]:
        try:
            vr = Vasprun(vasprun_path, parse_dos=False, parse_eigen=False, parse_projected_eigen=False)
            out["converged"] = bool(vr.converged)
            out["converged_electronic"] = bool(vr.converged_electronic)
            out["converged_ionic"] = bool(vr.converged_ionic)
            out["final_energy_ev_cell"] = as_float(getattr(vr, "final_energy", None))

            # Final forces/stress from last ionic step if available.
            fmax = None
            sfro = None
            if getattr(vr, "ionic_steps", None):
                last = vr.ionic_steps[-1]
                f = np.array(last.get("forces", []), dtype=float)
                st = np.array(last.get("stress", []), dtype=float)
                if f.size:
                    if f.ndim == 1:
                        f = f.reshape(-1, 3)
                    fmax = float(np.max(np.linalg.norm(f, axis=1)))
                if st.size:
                    sfro = float(np.linalg.norm(st))
            out["dft_force_max"] = fmax
            out["dft_stress_fro"] = sfro
        except Exception:
            pass

    # Parse outcar for fallback energy + elastic tensor.
    c6_dft = None
    if out["has_outcar"]:
        try:
            oc = Outcar(outcar_path)
            if out["final_energy_ev_cell"] is None:
                out["final_energy_ev_cell"] = as_float(getattr(oc, "final_energy", None))
            try:
                oc.read_elastic_tensor()
                et = oc.data.get("elastic_tensor")
                if isinstance(et, list) and len(et) == 6 and all(isinstance(r, list) and len(r) == 6 for r in et):
                    c6_dft = np.array(et, dtype=float) * 0.1  # kBar -> GPa
                    out["dft_has_elastic"] = True
            except Exception:
                pass
        except Exception:
            pass

    # Energy per atom.
    if out["final_energy_ev_cell"] is not None and structure is not None and len(structure) > 0:
        out["final_energy_ev_atom"] = float(out["final_energy_ev_cell"]) / float(len(structure))

    # DFT mechanics.
    if c6_dft is not None:
        mm = mechanical_metrics(c6_dft)
        out["dft_min_eig"] = mm.get("min_eig")
        out["dft_B_H"] = mm.get("B_H")
        out["dft_G_H"] = mm.get("G_H")
        out["dft_E_H"] = mm.get("E_H")
        out["dft_nu_H"] = mm.get("nu_H")
        out["dft_A_U"] = mm.get("A_U")

    # Pass/fail gates.
    if out["dft_force_max"] is not None:
        out["pass_force_tol"] = bool(float(out["dft_force_max"]) <= float(force_tol))
    if out["dft_min_eig"] is not None:
        out["pass_pd"] = bool(float(out["dft_min_eig"]) > float(eig_tol))

    # DFT-vs-ML scalar consistency.
    pred_bh = as_float(row.get("B_H"))
    pred_gh = as_float(row.get("G_H"))
    pred_eh = as_float(row.get("E_H"))
    out["relerr_B_H_vs_pred"] = safe_rel_err(pred_bh, out["dft_B_H"])
    out["relerr_G_H_vs_pred"] = safe_rel_err(pred_gh, out["dft_G_H"])
    out["relerr_E_H_vs_pred"] = safe_rel_err(pred_eh, out["dft_E_H"])

    # DFT-vs-ML tensor consistency (if predicted voigt available).
    c6_pred = try_load_predicted_voigt(
        metadata=metadata,
        voigt_mean=voigt_mean,
        voigt_std=voigt_std,
        pred_voigt_is_normalized=bool(pred_voigt_is_normalized),
    )
    if c6_pred is not None and c6_dft is not None:
        out["relerr_voigt21_fro"] = fro_rel_err(c6_pred, c6_dft)

    # Optional MP hull metric.
    e_hull, err = try_mp_hull(
        mpr=mpr,
        structure=structure,
        final_energy_ev_cell=out["final_energy_ev_cell"],
        cache=mp_cache,
    )
    out["mp_e_above_hull"] = e_hull
    out["mp_hull_error"] = err
    if e_hull is not None:
        out["pass_hull"] = bool(float(e_hull) <= float(hull_tol))

    # Status inference.
    if out["dft_has_elastic"]:
        out["status"] = "elastic_ready"
    elif out["has_vasprun"] and out.get("converged") is True:
        out["status"] = "relax_or_static_done"
    elif out["has_outcar"] or out["has_vasprun"]:
        out["status"] = "output_present_not_converged"
    else:
        out["status"] = "pending"

    return out


def summarize(rows: List[dict]) -> dict:
    n = len(rows)
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
        "n_total": int(n),
        "status_counts": dict(status_counts),
        "has_vasprun": rate("has_vasprun", True),
        "has_outcar": rate("has_outcar", True),
        "dft_has_elastic": rate("dft_has_elastic", True),
        "converged": rate("converged", True),
        "pass_force_tol": rate("pass_force_tol", True),
        "pass_pd": rate("pass_pd", True),
        "pass_hull": rate("pass_hull", True),
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
        if r.get("pass_force_tol") is False:
            continue
        if require_hull and r.get("pass_hull") is not True:
            continue
        cand.append(r)

    def key(r):
        bh = r.get("relerr_B_H_vs_pred")
        gh = r.get("relerr_G_H_vs_pred")
        eh = r.get("relerr_E_H_vs_pred")
        tt = r.get("relerr_voigt21_fro")
        ehull = r.get("mp_e_above_hull")
        # Smaller is better. Use large fallback.
        score = (
            float(bh) if bh is not None else 1e9,
            float(gh) if gh is not None else 1e9,
            float(eh) if eh is not None else 1e9,
            float(tt) if tt is not None else 1e9,
            float(ehull) if ehull is not None else (0.0 if not require_hull else 1e9),
        )
        return score

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
    ap.add_argument("--force_tol", type=float, default=0.05)
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
            force_tol=float(args.force_tol),
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
    print("DFT CAMPAIGN ANALYSIS COMPLETE")
    print("=" * 72)
    print(f"in_manifest={args.campaign_manifest}")
    print(f"out_csv={args.out_csv}")
    print(f"out_summary={args.out_summary_json}")
    if args.out_validated_csv:
        print(f"out_validated={args.out_validated_csv}")
    print(f"status_counts={summary.get('status_counts')}")


if __name__ == "__main__":
    main()

