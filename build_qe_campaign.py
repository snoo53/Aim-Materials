"""
Build a Quantum ESPRESSO-ready campaign package from shortlist metadata.

For each selected candidate this script writes:
- structure.cif
- POSCAR (for cross-tool interoperability)
- qe_relax.in / qe_scf.in
- qe_elastic/strain_xxx/qe_scf.in (+ strain_manifest.csv)
- PSEUDO.spec (symbol -> pseudo filename hints; no binaries)
- metadata.json
- local runner scripts (PowerShell + bash)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
from collections import Counter, defaultdict
from typing import Dict, List

import numpy as np
from pymatgen.core import Lattice, Structure
from pymatgen.io.pwscf import PWInput
from pymatgen.io.vasp.inputs import Kpoints, Poscar


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


def as_float(x, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def sanitize(text: str) -> str:
    s = str(text)
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s)
    s = s.strip("._")
    return s or "unknown"


def score_row(row: dict) -> float:
    quality = as_float(row.get("quality_score"), 0.0)
    min_d = as_float(row.get("min_distance"), 0.0)
    us = as_float(row.get("scalar_std_mean"), 0.0)
    uv = as_float(row.get("voigt_std_mean"), 0.0)
    fmax = as_float(row.get("chgnet_force_max"), 0.0)
    score = quality + 0.15 * min_d - 40.0 * us - 30.0 * uv - 0.8 * fmax
    return float(score)


def filter_primary(rows: List[dict], args) -> List[dict]:
    out = []
    for r in rows:
        if not as_bool(r.get("strict_pass")):
            continue
        if not as_bool(r.get("formula_novel_vs_train")):
            continue
        if float(args.max_scalar_std) > 0.0 and as_float(r.get("scalar_std_mean"), 999.0) > float(args.max_scalar_std):
            continue
        if float(args.max_voigt_std) > 0.0 and as_float(r.get("voigt_std_mean"), 999.0) > float(args.max_voigt_std):
            continue
        if bool(args.require_chgnet_pass):
            if not as_bool(r.get("chgnet_pass")):
                continue
        elif float(args.max_force) > 0.0 and str(r.get("chgnet_force_max", "")).strip() != "":
            if as_float(r.get("chgnet_force_max"), 999.0) > float(args.max_force):
                continue
        out.append(r)
    return out


def filter_fallback(rows: List[dict], args) -> List[dict]:
    out = []
    for r in rows:
        if not as_bool(r.get("strict_pass")):
            continue
        if not as_bool(r.get("formula_novel_vs_train")):
            continue
        out.append(r)
    return out


def select_rows(rows: List[dict], top_n: int, max_per_formula: int) -> List[dict]:
    ranked = sorted(rows, key=score_row, reverse=True)
    out = []
    formula_counter = Counter()
    seen_ids = set()
    for r in ranked:
        mid = str(r.get("material_id", ""))
        if not mid or mid in seen_ids:
            continue
        rf = str(r.get("reduced_formula", ""))
        if int(max_per_formula) > 0 and formula_counter[rf] >= int(max_per_formula):
            continue
        seen_ids.add(mid)
        formula_counter[rf] += 1
        rr = dict(r)
        rr["selection_score"] = score_row(r)
        out.append(rr)
        if len(out) >= int(top_n):
            break
    return out


def kgrid_from_kppra(structure: Structure, kppra: float) -> List[int]:
    if len(structure) <= 0:
        return [1, 1, 1]
    kp = Kpoints.automatic_density(structure, float(kppra), force_gamma=True)
    if kp.kpts and len(kp.kpts[0]) == 3:
        return [max(1, int(x)) for x in kp.kpts[0]]
    return [1, 1, 1]


def pseudo_map_for_structure(structure: Structure, pseudo_suffix: str) -> Dict[str, str]:
    out = {}
    for el in sorted({str(sp) for sp in structure.species}):
        out[el] = f"{el}{pseudo_suffix}"
    return out


def write_pw_input(
    structure: Structure,
    out_path: str,
    calculation: str,
    prefix: str,
    pseudo_dir: str,
    pseudo_map: Dict[str, str],
    kgrid: List[int],
    ecutwfc: float,
    ecutrho: float,
    conv_thr: float,
    degauss_ry: float,
    press_kbar: float,
):
    control = {
        "calculation": calculation,
        "prefix": prefix,
        "pseudo_dir": pseudo_dir,
        "outdir": "./tmp",
        "tprnfor": True,
        "tstress": True,
        "wf_collect": False,
    }

    system = {
        "ecutwfc": float(ecutwfc),
        "ecutrho": float(ecutrho),
        "occupations": "smearing",
        "smearing": "mv",
        "degauss": float(degauss_ry),
    }
    electrons = {"conv_thr": float(conv_thr), "mixing_beta": 0.4}

    ions = None
    cell = None
    if calculation in {"vc-relax", "vc_md", "vc-md"}:
        ions = {"ion_dynamics": "bfgs"}
        cell = {"cell_dynamics": "bfgs", "press": float(press_kbar), "cell_dofree": "all"}

    pw = PWInput(
        structure=structure,
        pseudo=pseudo_map,
        control=control,
        system=system,
        electrons=electrons,
        ions=ions,
        cell=cell,
        kpoints_mode="automatic",
        kpoints_grid=tuple(int(max(1, x)) for x in kgrid),
        kpoints_shift=(0, 0, 0),
    )
    pw.write_file(out_path)


def voigt6_to_small_strain(e: np.ndarray) -> np.ndarray:
    # Engineering shear convention: e4=gamma_yz, e5=gamma_xz, e6=gamma_xy.
    return np.array(
        [
            [e[0], 0.5 * e[5], 0.5 * e[4]],
            [0.5 * e[5], e[1], 0.5 * e[3]],
            [0.5 * e[4], 0.5 * e[3], e[2]],
        ],
        dtype=float,
    )


def apply_small_strain(structure: Structure, voigt6: np.ndarray) -> Structure:
    eps = voigt6_to_small_strain(voigt6)
    deform = np.eye(3, dtype=float) + eps
    new_lat = np.dot(structure.lattice.matrix, deform)
    strained = Structure(
        lattice=Lattice(new_lat),
        species=structure.species,
        coords=structure.frac_coords,
        coords_are_cartesian=False,
        to_unit_cell=True,
        site_properties=structure.site_properties,
    )
    return strained


def write_strain_manifest(path: str, rows: List[dict]) -> None:
    fields = [
        "strain_id",
        "eps1",
        "eps2",
        "eps3",
        "eps4",
        "eps5",
        "eps6",
        "input",
        "output",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def build_strain_vectors(strain_amp: float) -> List[np.ndarray]:
    out = []
    for j in range(6):
        for sgn in (-1.0, 1.0):
            e = np.zeros(6, dtype=float)
            e[j] = sgn * float(strain_amp)
            out.append(e)
    return out


def write_runner_scripts(out_dir: str) -> None:
    sh_relax = """#!/usr/bin/env bash
set -euo pipefail
QE_CMD="${1:-pw.x}"
NPROC="${2:-1}"

while IFS= read -r rel; do
  [[ -z "${rel}" ]] && continue
  d="$PWD/$rel"
  (
    cd "$d/01_relax"
    mkdir -p tmp
    if [[ "$NPROC" -gt 1 ]]; then
      mpirun -np "$NPROC" "$QE_CMD" -in qe_relax.in > qe_relax.out
    else
      "$QE_CMD" -in qe_relax.in > qe_relax.out
    fi
  )
done < "$PWD/candidate_paths.txt"
"""

    sh_scf = """#!/usr/bin/env bash
set -euo pipefail
QE_CMD="${1:-pw.x}"
NPROC="${2:-1}"

while IFS= read -r rel; do
  [[ -z "${rel}" ]] && continue
  d="$PWD/$rel"
  (
    cd "$d/02_scf"
    mkdir -p tmp
    if [[ "$NPROC" -gt 1 ]]; then
      mpirun -np "$NPROC" "$QE_CMD" -in qe_scf.in > qe_scf.out
    else
      "$QE_CMD" -in qe_scf.in > qe_scf.out
    fi
  )
done < "$PWD/candidate_paths.txt"
"""

    sh_elastic = """#!/usr/bin/env bash
set -euo pipefail
QE_CMD="${1:-pw.x}"
NPROC="${2:-1}"

while IFS= read -r rel; do
  [[ -z "${rel}" ]] && continue
  d="$PWD/$rel/03_elastic"
  if [[ ! -f "$d/strain_manifest.csv" ]]; then
    continue
  fi
  tail -n +2 "$d/strain_manifest.csv" | while IFS=, read -r sid e1 e2 e3 e4 e5 e6 inp outp; do
    dd="$d/strain_${sid}"
    (
      cd "$dd"
      mkdir -p tmp
      if [[ "$NPROC" -gt 1 ]]; then
        mpirun -np "$NPROC" "$QE_CMD" -in qe_scf.in > qe_scf.out
      else
        "$QE_CMD" -in qe_scf.in > qe_scf.out
      fi
    )
  done
done < "$PWD/candidate_paths.txt"
"""

    sh_all = """#!/usr/bin/env bash
set -euo pipefail
QE_CMD="${1:-pw.x}"
NPROC="${2:-1}"

bash run_relax.sh "$QE_CMD" "$NPROC"
bash run_scf.sh "$QE_CMD" "$NPROC"
bash run_elastic.sh "$QE_CMD" "$NPROC"
"""

    ps_relax = r"""param([string]$QeExe="pw.x",[int]$NProc=1)
Get-Content -Path ".\candidate_paths.txt" | ForEach-Object {
  $rel = $_.Trim()
  if ($rel -eq "") { return }
  $d = Join-Path (Get-Location) $rel
  Push-Location (Join-Path $d "01_relax")
  New-Item -ItemType Directory -Force -Path "tmp" | Out-Null
  if ($NProc -gt 1) {
    & mpirun -np $NProc $QeExe -in qe_relax.in | Tee-Object -FilePath qe_relax.out
  } else {
    & $QeExe -in qe_relax.in | Tee-Object -FilePath qe_relax.out
  }
  Pop-Location
}
"""

    ps_scf = r"""param([string]$QeExe="pw.x",[int]$NProc=1)
Get-Content -Path ".\candidate_paths.txt" | ForEach-Object {
  $rel = $_.Trim()
  if ($rel -eq "") { return }
  $d = Join-Path (Get-Location) $rel
  Push-Location (Join-Path $d "02_scf")
  New-Item -ItemType Directory -Force -Path "tmp" | Out-Null
  if ($NProc -gt 1) {
    & mpirun -np $NProc $QeExe -in qe_scf.in | Tee-Object -FilePath qe_scf.out
  } else {
    & $QeExe -in qe_scf.in | Tee-Object -FilePath qe_scf.out
  }
  Pop-Location
}
"""

    ps_elastic = r"""param([string]$QeExe="pw.x",[int]$NProc=1)
Get-Content -Path ".\candidate_paths.txt" | ForEach-Object {
  $rel = $_.Trim()
  if ($rel -eq "") { return }
  $d = Join-Path (Get-Location) $rel
  $manifest = Join-Path $d "03_elastic\strain_manifest.csv"
  if (-not (Test-Path $manifest)) { return }
  Import-Csv $manifest | ForEach-Object {
    $sid = $_.strain_id
    $dd = Join-Path $d ("03_elastic\strain_" + $sid)
    Push-Location $dd
    New-Item -ItemType Directory -Force -Path "tmp" | Out-Null
    if ($NProc -gt 1) {
      & mpirun -np $NProc $QeExe -in qe_scf.in | Tee-Object -FilePath qe_scf.out
    } else {
      & $QeExe -in qe_scf.in | Tee-Object -FilePath qe_scf.out
    }
    Pop-Location
  }
}
"""

    ps_all = r"""param([string]$QeExe="pw.x",[int]$NProc=1)
.\run_relax.ps1 -QeExe $QeExe -NProc $NProc
.\run_scf.ps1 -QeExe $QeExe -NProc $NProc
.\run_elastic.ps1 -QeExe $QeExe -NProc $NProc
"""

    with open(os.path.join(out_dir, "run_relax.sh"), "w", encoding="utf-8") as f:
        f.write(sh_relax)
    with open(os.path.join(out_dir, "run_scf.sh"), "w", encoding="utf-8") as f:
        f.write(sh_scf)
    with open(os.path.join(out_dir, "run_elastic.sh"), "w", encoding="utf-8") as f:
        f.write(sh_elastic)
    with open(os.path.join(out_dir, "run_all.sh"), "w", encoding="utf-8") as f:
        f.write(sh_all)

    with open(os.path.join(out_dir, "run_relax.ps1"), "w", encoding="utf-8") as f:
        f.write(ps_relax)
    with open(os.path.join(out_dir, "run_scf.ps1"), "w", encoding="utf-8") as f:
        f.write(ps_scf)
    with open(os.path.join(out_dir, "run_elastic.ps1"), "w", encoding="utf-8") as f:
        f.write(ps_elastic)
    with open(os.path.join(out_dir, "run_all.ps1"), "w", encoding="utf-8") as f:
        f.write(ps_all)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest_csv", required=True, help="Input shortlist manifest CSV.")
    ap.add_argument("--out_dir", required=True, help="Output campaign folder.")
    ap.add_argument("--top_per_set", type=int, default=24)
    ap.add_argument("--max_per_formula", type=int, default=2)
    ap.add_argument("--max_scalar_std", type=float, default=0.05)
    ap.add_argument("--max_voigt_std", type=float, default=0.05)
    ap.add_argument("--max_force", type=float, default=0.25)
    ap.add_argument("--require_chgnet_pass", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--kppra_relax", type=float, default=1200.0)
    ap.add_argument("--kppra_scf", type=float, default=4000.0)
    ap.add_argument("--strain_amp", type=float, default=0.005)
    ap.add_argument("--ecutwfc", type=float, default=80.0)
    ap.add_argument("--ecutrho", type=float, default=640.0)
    ap.add_argument("--degauss_ry", type=float, default=0.02)
    ap.add_argument("--conv_thr_relax", type=float, default=1e-8)
    ap.add_argument("--conv_thr_scf", type=float, default=1e-10)
    ap.add_argument("--press_kbar", type=float, default=0.0)
    ap.add_argument("--pseudo_dir", default="./pseudo")
    ap.add_argument("--pseudo_suffix", default=".upf")
    return ap.parse_args()


def main():
    args = parse_args()
    rows = read_csv(args.manifest_csv)
    if not rows:
        raise RuntimeError(f"No rows found in {args.manifest_csv}")

    os.makedirs(args.out_dir, exist_ok=True)
    selected_all = []
    summary = {"input_rows": len(rows), "sets": {}}

    by_set = defaultdict(list)
    for r in rows:
        by_set[str(r.get("set", ""))].append(r)

    for set_name, set_rows in by_set.items():
        primary = filter_primary(set_rows, args)
        fallback = filter_fallback(set_rows, args)

        picked = select_rows(primary, top_n=int(args.top_per_set), max_per_formula=int(args.max_per_formula))
        if len(picked) < int(args.top_per_set):
            needed = int(args.top_per_set) - len(picked)
            existing = {str(r.get("material_id")) for r in picked}
            fallback_extra = [
                r
                for r in select_rows(fallback, top_n=10000, max_per_formula=int(args.max_per_formula))
                if str(r.get("material_id")) not in existing
            ]
            picked.extend(fallback_extra[:needed])

        summary["sets"][set_name] = {
            "n_input": len(set_rows),
            "n_primary_after_filters": len(primary),
            "n_fallback_pool": len(fallback),
            "n_selected": len(picked),
        }
        selected_all.extend(picked)

    campaign_rows = []
    candidate_paths = []
    strain_vectors = build_strain_vectors(float(args.strain_amp))

    for r in selected_all:
        set_name = str(r.get("set", "unknown"))
        mid = str(r.get("material_id", ""))
        formula = sanitize(r.get("reduced_formula", "unknown"))
        rank = int(len([x for x in campaign_rows if x.get("set") == set_name]) + 1)

        src_cif = os.path.normpath(str(r.get("cif_path", "")).strip())
        if not src_cif or not os.path.exists(src_cif):
            continue

        cand_rel = os.path.join(set_name, f"{rank:03d}_{sanitize(mid)}_{formula}")
        cand_dir = os.path.join(args.out_dir, cand_rel)
        os.makedirs(cand_dir, exist_ok=True)

        # Structure files.
        dst_cif = os.path.join(cand_dir, "structure.cif")
        shutil.copy2(src_cif, dst_cif)
        struct = Structure.from_file(src_cif)
        Poscar(struct).write_file(os.path.join(cand_dir, "POSCAR"))

        # QE settings.
        pseudo_map = pseudo_map_for_structure(struct, args.pseudo_suffix)
        k_relax = kgrid_from_kppra(struct, float(args.kppra_relax))
        k_scf = kgrid_from_kppra(struct, float(args.kppra_scf))
        prefix = sanitize(f"{set_name}_{mid}_{formula}")

        relax_dir = os.path.join(cand_dir, "01_relax")
        scf_dir = os.path.join(cand_dir, "02_scf")
        elas_dir = os.path.join(cand_dir, "03_elastic")
        os.makedirs(relax_dir, exist_ok=True)
        os.makedirs(scf_dir, exist_ok=True)
        os.makedirs(elas_dir, exist_ok=True)

        write_pw_input(
            structure=struct,
            out_path=os.path.join(relax_dir, "qe_relax.in"),
            calculation="vc-relax",
            prefix=prefix,
            pseudo_dir=args.pseudo_dir,
            pseudo_map=pseudo_map,
            kgrid=k_relax,
            ecutwfc=float(args.ecutwfc),
            ecutrho=float(args.ecutrho),
            conv_thr=float(args.conv_thr_relax),
            degauss_ry=float(args.degauss_ry),
            press_kbar=float(args.press_kbar),
        )
        write_pw_input(
            structure=struct,
            out_path=os.path.join(scf_dir, "qe_scf.in"),
            calculation="scf",
            prefix=prefix,
            pseudo_dir=args.pseudo_dir,
            pseudo_map=pseudo_map,
            kgrid=k_scf,
            ecutwfc=float(args.ecutwfc),
            ecutrho=float(args.ecutrho),
            conv_thr=float(args.conv_thr_scf),
            degauss_ry=float(args.degauss_ry),
            press_kbar=float(args.press_kbar),
        )

        strain_rows = []
        for i, eps in enumerate(strain_vectors, start=1):
            sid = f"{i:03d}"
            sd = os.path.join(elas_dir, f"strain_{sid}")
            os.makedirs(sd, exist_ok=True)
            strained = apply_small_strain(struct, eps)
            write_pw_input(
                structure=strained,
                out_path=os.path.join(sd, "qe_scf.in"),
                calculation="scf",
                prefix=f"{prefix}_s{sid}",
                pseudo_dir=args.pseudo_dir,
                pseudo_map=pseudo_map,
                kgrid=k_scf,
                ecutwfc=float(args.ecutwfc),
                ecutrho=float(args.ecutrho),
                conv_thr=float(args.conv_thr_scf),
                degauss_ry=float(args.degauss_ry),
                press_kbar=float(args.press_kbar),
            )
            strain_rows.append(
                {
                    "strain_id": sid,
                    "eps1": float(eps[0]),
                    "eps2": float(eps[1]),
                    "eps3": float(eps[2]),
                    "eps4": float(eps[3]),
                    "eps5": float(eps[4]),
                    "eps6": float(eps[5]),
                    "input": os.path.join("strain_" + sid, "qe_scf.in"),
                    "output": os.path.join("strain_" + sid, "qe_scf.out"),
                }
            )
        write_strain_manifest(os.path.join(elas_dir, "strain_manifest.csv"), strain_rows)

        # Pseudo hint file.
        with open(os.path.join(cand_dir, "PSEUDO.spec"), "w", encoding="utf-8") as f:
            for el in sorted(pseudo_map):
                f.write(f"{el},{pseudo_map[el]}\n")

        # Metadata.
        meta = dict(r)
        meta["campaign_set"] = set_name
        meta["campaign_rank_in_set"] = rank
        meta["campaign_selection_score"] = score_row(r)
        meta["campaign_dir"] = cand_dir
        meta["qe_prefix"] = prefix
        meta["qe_kgrid_relax"] = k_relax
        meta["qe_kgrid_scf"] = k_scf
        with open(os.path.join(cand_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        campaign_rows.append(
            {
                "set": set_name,
                "rank_in_set": rank,
                "material_id": mid,
                "reduced_formula": r.get("reduced_formula"),
                "selection_score": score_row(r),
                "strict_pass": r.get("strict_pass"),
                "formula_novel_vs_train": r.get("formula_novel_vs_train"),
                "quality_score": r.get("quality_score"),
                "scalar_std_mean": r.get("scalar_std_mean"),
                "voigt_std_mean": r.get("voigt_std_mean"),
                "chgnet_force_max": r.get("chgnet_force_max", ""),
                "campaign_dir": cand_dir,
                "candidate_relpath": cand_rel.replace("\\", "/"),
                "cif": dst_cif,
                "scf_input": os.path.join(scf_dir, "qe_scf.in"),
                "elastic_manifest": os.path.join(elas_dir, "strain_manifest.csv"),
            }
        )
        candidate_paths.append(cand_rel.replace("\\", "/"))

    campaign_csv = os.path.join(args.out_dir, "campaign_manifest.csv")
    campaign_json = os.path.join(args.out_dir, "campaign_manifest.json")
    summary_json = os.path.join(args.out_dir, "campaign_summary.json")
    write_csv(campaign_csv, campaign_rows)
    with open(campaign_json, "w", encoding="utf-8") as f:
        json.dump(campaign_rows, f, indent=2)

    with open(os.path.join(args.out_dir, "candidate_paths.txt"), "w", encoding="utf-8") as f:
        for p in candidate_paths:
            f.write(p + "\n")

    summary["n_selected_total"] = len(campaign_rows)
    summary["filters"] = {
        "max_scalar_std": float(args.max_scalar_std),
        "max_voigt_std": float(args.max_voigt_std),
        "max_force": float(args.max_force),
        "require_chgnet_pass": bool(args.require_chgnet_pass),
        "top_per_set": int(args.top_per_set),
        "max_per_formula": int(args.max_per_formula),
    }
    summary["qe"] = {
        "ecutwfc": float(args.ecutwfc),
        "ecutrho": float(args.ecutrho),
        "kppra_relax": float(args.kppra_relax),
        "kppra_scf": float(args.kppra_scf),
        "strain_amp": float(args.strain_amp),
        "conv_thr_relax": float(args.conv_thr_relax),
        "conv_thr_scf": float(args.conv_thr_scf),
        "pseudo_dir": args.pseudo_dir,
        "pseudo_suffix": args.pseudo_suffix,
        "n_strains_per_candidate": len(strain_vectors),
    }
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    write_runner_scripts(args.out_dir)
    with open(os.path.join(args.out_dir, "README_campaign.txt"), "w", encoding="utf-8") as f:
        f.write(
            "QE Campaign Package (Local/Open-source)\n"
            "=======================================\n"
            "1) Place UPF pseudo files under the pseudo directory referenced by qe_*.in.\n"
            "2) Relax: run ./run_relax.sh [pw.x] [nproc] or .\\run_relax.ps1.\n"
            "3) SCF: run ./run_scf.sh [pw.x] [nproc].\n"
            "4) Elastic (finite-strain stress): run ./run_elastic.sh [pw.x] [nproc].\n"
            "5) Analyze outputs with analyze_qe_campaign_results.py.\n"
            "\n"
            "Notes:\n"
            "- Elastic constants are fitted from stress response of +/- strain_amp perturbations.\n"
            "- Uses engineering shear convention in Voigt notation.\n"
            "- If your QE stress sign appears inverted, use --qe_stress_sign in analyzer.\n"
        )

    print("=" * 72)
    print("QE CAMPAIGN BUILD COMPLETE")
    print("=" * 72)
    print(f"in_manifest={args.manifest_csv}")
    print(f"out_dir={args.out_dir}")
    print(f"n_selected_total={len(campaign_rows)}")
    print(f"campaign_manifest={campaign_csv}")
    print(f"summary={summary_json}")


if __name__ == "__main__":
    main()
