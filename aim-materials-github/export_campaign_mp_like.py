import argparse
import csv
import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


def _to_float(x: Any) -> Optional[float]:
    try:
        if x is None or x == "":
            return None
        return float(x)
    except Exception:
        return None


def _load_manifest(manifest_csv: Path) -> Dict[str, Dict[str, str]]:
    rows: Dict[str, Dict[str, str]] = {}
    with manifest_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rel = (row.get("candidate_relpath") or "").strip()
            if rel:
                rows[rel] = row
    return rows


def _load_candidate_paths(path_file: Path) -> List[str]:
    out: List[str] = []
    with path_file.open("r", encoding="utf-8") as f:
        for line in f:
            rel = line.strip()
            if rel:
                out.append(rel)
    return out


def _load_moduli_source(moduli_json: Path) -> Dict[Tuple[str, str], Dict[str, Any]]:
    if not moduli_json.exists():
        return {}
    data = json.loads(moduli_json.read_text(encoding="utf-8"))
    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    if not isinstance(data, list):
        return out
    for row in data:
        if not isinstance(row, dict):
            continue
        set_name = str(row.get("set", "")).strip()
        mid = str(row.get("material_id", "")).strip()
        if set_name and mid:
            out[(set_name, mid)] = row
    return out


def _load_moduli_csv_by_set(root: Path) -> Dict[Tuple[str, str], Dict[str, Any]]:
    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for set_name in ("2el", "3el", "4el"):
        p = root / f"candidates_{set_name}_densityaware_strict_v3_retrain_strict_novel_unique.csv"
        if not p.exists():
            continue
        with p.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                mid = str(row.get("material_id", "")).strip()
                if mid:
                    out[(set_name, mid)] = row
    return out


def _load_tensor_csv_by_set(root: Path) -> Dict[Tuple[str, str], List[float]]:
    out: Dict[Tuple[str, str], List[float]] = {}
    pred_dir = root / "predictions"
    for set_name in ("2el", "3el", "4el"):
        p = pred_dir / f"{set_name}_densityaware_strict_v3_retrain.csv"
        if not p.exists():
            continue
        with p.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                mid = str(row.get("material_id", "")).strip()
                if not mid:
                    continue
                vals: List[float] = []
                ok = True
                for i in range(21):
                    key = f"c{i}"
                    if key not in row:
                        ok = False
                        break
                    try:
                        vals.append(float(row[key]))
                    except Exception:
                        ok = False
                        break
                if ok:
                    out[(set_name, mid)] = vals
    return out


def _voigt21_to_6x6(v: List[float]) -> List[List[float]]:
    (
        c11,
        c12,
        c13,
        c14,
        c15,
        c16,
        c22,
        c23,
        c24,
        c25,
        c26,
        c33,
        c34,
        c35,
        c36,
        c44,
        c45,
        c46,
        c55,
        c56,
        c66,
    ) = v

    c = [[0.0] * 6 for _ in range(6)]
    c[0][0] = c11
    c[1][1] = c22
    c[2][2] = c33
    c[3][3] = c44
    c[4][4] = c55
    c[5][5] = c66

    c[0][1] = c[1][0] = c12
    c[0][2] = c[2][0] = c13
    c[1][2] = c[2][1] = c23

    c[0][3] = c[3][0] = c14
    c[0][4] = c[4][0] = c15
    c[0][5] = c[5][0] = c16
    c[1][3] = c[3][1] = c24
    c[1][4] = c[4][1] = c25
    c[1][5] = c[5][1] = c26
    c[2][3] = c[3][2] = c34
    c[2][4] = c[4][2] = c35
    c[2][5] = c[5][2] = c36

    c[3][4] = c[4][3] = c45
    c[3][5] = c[5][3] = c46
    c[4][5] = c[5][4] = c56
    return c


def _prune_nulls(x: Any) -> Any:
    if isinstance(x, dict):
        out: Dict[str, Any] = {}
        for k, v in x.items():
            pv = _prune_nulls(v)
            if pv is not None:
                out[k] = pv
        return out
    if isinstance(x, list):
        out_list = []
        for v in x:
            pv = _prune_nulls(v)
            if pv is not None:
                out_list.append(pv)
        return out_list
    if x is None:
        return None
    return x


def _symmetry_info(structure: Structure) -> Dict[str, Any]:
    try:
        sga = SpacegroupAnalyzer(structure, symprec=0.1)
        crystal_system = sga.get_crystal_system()
        if isinstance(crystal_system, str):
            crystal_system = crystal_system.capitalize()
        try:
            import spglib  # type: ignore

            spg_ver = getattr(spglib, "__version__", None)
        except Exception:
            spg_ver = None
        return {
            "crystal_system": crystal_system,
            "symbol": sga.get_space_group_symbol(),
            "number": int(sga.get_space_group_number()),
            "point_group": sga.get_point_group_symbol(),
            "symprec": 0.1,
            "version": spg_ver,
        }
    except Exception:
        return {
            "crystal_system": None,
            "symbol": None,
            "number": None,
            "point_group": None,
            "symprec": 0.1,
            "version": None,
        }


def _composition_info(structure: Structure) -> Tuple[List[str], Dict[str, float], Dict[str, float], str, str, str]:
    comp = structure.composition
    reduced = comp.reduced_composition
    elements = sorted(str(e) for e in comp.elements)
    chemsys = "-".join(elements)
    formula_pretty = reduced.reduced_formula
    formula_anonymous = reduced.anonymized_formula
    comp_dict = {str(k): float(v) for k, v in comp.as_dict().items()}
    comp_reduced = {str(k): float(v) for k, v in reduced.as_dict().items()}
    return elements, comp_dict, comp_reduced, formula_pretty, formula_anonymous, chemsys


def _build_builder_meta(dataset_name: str) -> Dict[str, Any]:
    try:
        from importlib.metadata import version

        pymatgen_ver = version("pymatgen")
    except Exception:
        pymatgen_ver = None
    return {
        "emmet_version": "aim-materials-local",
        "pymatgen_version": pymatgen_ver,
        "pull_request": None,
        "database_version": dataset_name,
        "build_date": datetime.now(timezone.utc).isoformat(),
        "license": "BY-C",
    }


def _blank_summary_record() -> Dict[str, Any]:
    return {
        "builder_meta": None,
        "nsites": None,
        "elements": None,
        "nelements": None,
        "composition": None,
        "composition_reduced": None,
        "formula_pretty": None,
        "formula_anonymous": None,
        "chemsys": None,
        "volume": None,
        "density": None,
        "density_atomic": None,
        "symmetry": None,
        "property_name": "summary",
        "material_id": None,
        "deprecated": False,
        "deprecation_reasons": None,
        "last_updated": None,
        "origins": [],
        "warnings": [],
        "structure": None,
        "task_ids": [],
        "uncorrected_energy_per_atom": None,
        "energy_per_atom": None,
        "formation_energy_per_atom": None,
        "energy_above_hull": None,
        "is_stable": None,
        "equilibrium_reaction_energy_per_atom": None,
        "decomposes_to": None,
        "xas": None,
        "grain_boundaries": None,
        "band_gap": None,
        "cbm": None,
        "vbm": None,
        "efermi": None,
        "is_gap_direct": None,
        "is_metal": None,
        "es_source_calc_id": None,
        "bandstructure": None,
        "dos": None,
        "dos_energy_up": None,
        "dos_energy_down": None,
        "is_magnetic": None,
        "ordering": None,
        "total_magnetization": None,
        "total_magnetization_normalized_vol": None,
        "total_magnetization_normalized_formula_units": None,
        "num_magnetic_sites": None,
        "num_unique_magnetic_sites": None,
        "types_of_magnetic_species": None,
        "bulk_modulus": None,
        "shear_modulus": None,
        "universal_anisotropy": None,
        "homogeneous_poisson": None,
        "e_total": None,
        "e_ionic": None,
        "e_electronic": None,
        "n": None,
        "e_ij_max": None,
        "weighted_surface_energy_EV_PER_ANG2": None,
        "weighted_surface_energy": None,
        "weighted_work_function": None,
        "surface_anisotropy": None,
        "shape_factor": None,
        "has_reconstructed": None,
        "possible_species": None,
        "has_props": ["summary", "elasticity"],
        "theoretical": True,
        "database_IDs": None,
    }


def _blank_elasticity_record() -> Dict[str, Any]:
    return {
        "builder_meta": None,
        "nsites": None,
        "elements": None,
        "nelements": None,
        "composition": None,
        "composition_reduced": None,
        "formula_pretty": None,
        "formula_anonymous": None,
        "chemsys": None,
        "volume": None,
        "density": None,
        "density_atomic": None,
        "symmetry": None,
        "property_name": "elasticity",
        "material_id": None,
        "deprecated": False,
        "deprecation_reasons": None,
        "last_updated": None,
        "origins": [],
        "warnings": [],
        "structure": None,
        "order": 2,
        "elastic_tensor": None,
        "compliance_tensor": None,
        "bulk_modulus": None,
        "shear_modulus": None,
        "sound_velocity": None,
        "thermal_conductivity": None,
        "young_modulus": None,
        "universal_anisotropy": None,
        "homogeneous_poisson": None,
        "debye_temperature": None,
        "fitting_data": None,
        "fitting_method": "ml_predicted_proxy",
        "state": "predicted",
    }


def _fill_common(
    rec: Dict[str, Any],
    *,
    material_id: str,
    structure: Structure,
    builder_meta: Dict[str, Any],
) -> Dict[str, Any]:
    elements, comp, comp_red, f_pretty, f_anon, chemsys = _composition_info(structure)
    rec["builder_meta"] = deepcopy(builder_meta)
    rec["nsites"] = int(len(structure))
    rec["elements"] = elements
    rec["nelements"] = int(len(elements))
    rec["composition"] = comp
    rec["composition_reduced"] = comp_red
    rec["formula_pretty"] = f_pretty
    rec["formula_anonymous"] = f_anon
    rec["chemsys"] = chemsys
    rec["volume"] = float(structure.volume)
    rec["density"] = float(structure.density)
    rec["density_atomic"] = float(structure.volume / len(structure))
    rec["symmetry"] = _symmetry_info(structure)
    rec["material_id"] = material_id
    rec["last_updated"] = datetime.now(timezone.utc).isoformat()
    rec["structure"] = structure.as_dict()
    return rec


def build_exports(
    root_dir: Path,
    candidate_paths_file: Path,
    campaign_manifest_csv: Path,
    moduli_source_json: Path,
    out_summary_json: Path,
    out_elasticity_json: Path,
) -> None:
    rows_by_rel = _load_manifest(campaign_manifest_csv)
    moduli_by_key = _load_moduli_source(moduli_source_json)
    moduli_csv_by_key = _load_moduli_csv_by_set(Path("."))
    tensor_csv_by_key = _load_tensor_csv_by_set(Path("."))
    relpaths = _load_candidate_paths(candidate_paths_file)

    builder_meta = _build_builder_meta("qe_campaign_v1_local")

    summary_records: List[Dict[str, Any]] = []
    elasticity_records: List[Dict[str, Any]] = []

    missing_cif: List[str] = []
    missing_manifest: List[str] = []

    for rel in relpaths:
        row = rows_by_rel.get(rel)
        if row is None:
            missing_manifest.append(rel)
            set_name = rel.split("/")[0] if "/" in rel else "unknown"
            gen_id = Path(rel).name
            reduced_formula = gen_id
            quality_score = None
            b_h = g_h = e_h = nu_h = a_u = None
            voigt21 = None
        else:
            set_name = row.get("set", "unknown")
            gen_id = row.get("material_id", Path(rel).name)
            reduced_formula = row.get("reduced_formula", Path(rel).name)
            quality_score = _to_float(row.get("quality_score"))
            b_h = _to_float(row.get("B_H"))
            g_h = _to_float(row.get("G_H"))
            e_h = _to_float(row.get("E_H"))
            nu_h = _to_float(row.get("nu_H"))
            a_u = _to_float(row.get("A_U"))
            voigt21 = None

        src = moduli_by_key.get((set_name, gen_id))
        if src:
            if quality_score is None:
                quality_score = _to_float(src.get("quality_score"))
            if b_h is None:
                b_h = _to_float(src.get("B_H"))
            if g_h is None:
                g_h = _to_float(src.get("G_H"))
            if e_h is None:
                e_h = _to_float(src.get("E_H"))
            if nu_h is None:
                nu_h = _to_float(src.get("nu_H"))
            if a_u is None:
                a_u = _to_float(src.get("A_U"))
            v = src.get("targets_voigt21")
            if isinstance(v, list) and len(v) == 21:
                voigt21 = [float(t) for t in v]

        src_csv = moduli_csv_by_key.get((set_name, gen_id))
        if src_csv:
            if quality_score is None:
                quality_score = _to_float(src_csv.get("quality_score"))
            if b_h is None:
                b_h = _to_float(src_csv.get("B_H"))
            if g_h is None:
                g_h = _to_float(src_csv.get("G_H"))
            if e_h is None:
                e_h = _to_float(src_csv.get("E_H"))
            if nu_h is None:
                nu_h = _to_float(src_csv.get("nu_H"))
            if a_u is None:
                a_u = _to_float(src_csv.get("A_U"))

        if voigt21 is None:
            voigt21 = tensor_csv_by_key.get((set_name, gen_id))

        aim_mid = f"aim-{set_name}-{gen_id}"
        cif_path = root_dir / rel / "structure.cif"
        if not cif_path.exists():
            missing_cif.append(rel)
            continue

        structure = Structure.from_file(cif_path)

        srec = _blank_summary_record()
        srec = _fill_common(srec, material_id=aim_mid, structure=structure, builder_meta=builder_meta)
        srec["bulk_modulus"] = b_h
        srec["shear_modulus"] = g_h
        srec["universal_anisotropy"] = a_u
        srec["homogeneous_poisson"] = nu_h
        srec["database_IDs"] = {
            "aim_materials": [aim_mid],
            "source_candidate_relpath": rel,
            "source_manifest_id": gen_id,
            "source_set": set_name,
            "source_reduced_formula": reduced_formula,
        }
        if quality_score is not None:
            srec["warnings"].append(f"quality_score={quality_score:.6f}")

        erec = _blank_elasticity_record()
        erec = _fill_common(erec, material_id=aim_mid, structure=structure, builder_meta=builder_meta)
        erec["bulk_modulus"] = {"voigt": b_h, "reuss": b_h, "vrh": b_h}
        erec["shear_modulus"] = {"voigt": g_h, "reuss": g_h, "vrh": g_h}
        erec["young_modulus"] = e_h
        erec["universal_anisotropy"] = a_u
        erec["homogeneous_poisson"] = nu_h
        erec["fitting_data"] = {
            "source": "aim_materials_ml_predictions",
            "selection_quality_score": quality_score,
            "source_candidate_relpath": rel,
        }
        if voigt21 is not None:
            c_raw = _voigt21_to_6x6(voigt21)
            erec["elastic_tensor"] = {"raw": c_raw, "ieee_format": c_raw}
            try:
                s_raw = np.linalg.inv(np.array(c_raw, dtype=float)).tolist()
                erec["compliance_tensor"] = {"raw": s_raw, "ieee_format": s_raw}
            except Exception:
                pass

        summary_records.append(_prune_nulls(srec))
        elasticity_records.append(_prune_nulls(erec))

    out_summary_json.parent.mkdir(parents=True, exist_ok=True)
    out_elasticity_json.parent.mkdir(parents=True, exist_ok=True)

    out_summary_json.write_text(json.dumps(summary_records, indent=2), encoding="utf-8")
    out_elasticity_json.write_text(json.dumps(elasticity_records, indent=2), encoding="utf-8")

    print(f"wrote_summary={out_summary_json}")
    print(f"wrote_elasticity={out_elasticity_json}")
    print(f"records={len(summary_records)}")
    if missing_manifest:
        print(f"missing_manifest={len(missing_manifest)}")
    if missing_cif:
        print(f"missing_cif={len(missing_cif)}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Export campaign candidates to MP-summary-like and MP-elasticity-like JSON files."
    )
    ap.add_argument(
        "--root_dir",
        default="qe_campaign_v1_local",
        help="Campaign root directory containing candidate folders.",
    )
    ap.add_argument(
        "--candidate_paths",
        default="qe_campaign_v1_local/candidate_paths.txt",
        help="Text file of candidate relative paths.",
    )
    ap.add_argument(
        "--campaign_manifest",
        default="qe_campaign_v1_local/campaign_manifest.csv",
        help="Campaign manifest CSV.",
    )
    ap.add_argument(
        "--moduli_source",
        default="candidates_all_strict_novel_unique_top200.json",
        help="JSON source containing B_H/G_H/E_H/nu_H/A_U and targets_voigt21.",
    )
    ap.add_argument(
        "--out_summary",
        default="qe_campaign_v1_local/mp_summary_data_candidates_mixed54.json",
        help="Output summary-like JSON path.",
    )
    ap.add_argument(
        "--out_elasticity",
        default="qe_campaign_v1_local/mp_elasticity_data_candidates_mixed54.json",
        help="Output elasticity-like JSON path.",
    )
    args = ap.parse_args()

    build_exports(
        root_dir=Path(args.root_dir),
        candidate_paths_file=Path(args.candidate_paths),
        campaign_manifest_csv=Path(args.campaign_manifest),
        moduli_source_json=Path(args.moduli_source),
        out_summary_json=Path(args.out_summary),
        out_elasticity_json=Path(args.out_elasticity),
    )


if __name__ == "__main__":
    main()
