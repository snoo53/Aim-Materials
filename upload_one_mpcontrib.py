#!/usr/bin/env python3
"""
Check MPContribs project access and upload one elastic-ready candidate.

Usage:
  python upload_one_mpcontrib.py --project aim_materials_v1

Required environment variable:
  - MPCONTRIBS_API_KEY (preferred) or MP_API_KEY
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def _choose_api_key() -> Optional[str]:
    key = os.getenv("MPCONTRIBS_API_KEY", "").strip()
    if key:
        return key
    key = os.getenv("MP_API_KEY", "").strip()
    if key:
        return key
    return None


def _clean_none(obj: Any) -> Any:
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            vv = _clean_none(v)
            if vv is not None:
                out[k] = vv
        return out
    if isinstance(obj, list):
        out = [_clean_none(v) for v in obj]
        return [v for v in out if v is not None]
    return obj


def _pick_row(rows: List[Dict[str, Any]], material_id: Optional[str]) -> Dict[str, Any]:
    if material_id:
        for r in rows:
            if r.get("material_id") == material_id:
                return r
        raise RuntimeError(f"material_id not found: {material_id}")

    for r in rows:
        q = r.get("qe_validation") or {}
        if q.get("status") == "elastic_ready":
            return r

    raise RuntimeError("No row with qe_validation.status == elastic_ready")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True, help="MPContribs project name")
    ap.add_argument(
        "--data_json",
        default="qe_campaign_v1_local/mp_combined_data_candidates_mixed51_with_qe_status.json",
        help="Path to combined MP-like JSON",
    )
    ap.add_argument("--material_id", default="", help="Optional explicit material_id")
    ap.add_argument(
        "--skip_structure",
        action="store_true",
        help="Do not upload structure component",
    )
    ap.add_argument(
        "--dry_run",
        action="store_true",
        help="Check access + print payload preview only",
    )
    args = ap.parse_args()

    key = _choose_api_key()
    if not key:
        print("ERROR: set MPCONTRIBS_API_KEY (or MP_API_KEY) first.")
        return 2

    data_path = Path(args.data_json)
    if not data_path.exists():
        print(f"ERROR: data file not found: {data_path}")
        return 2

    try:
        from mpcontribs.client import Client
    except Exception as exc:
        print(f"ERROR: mpcontribs-client not available: {exc}")
        print("Install with: pip install mpcontribs-client mp_api pymatgen")
        return 2

    rows = json.loads(data_path.read_text(encoding="utf-8"))
    row = _pick_row(rows, args.material_id or None)
    q = row.get("qe_validation") or {}

    # MPContribs labels/keys are strict; avoid underscores/special chars in data keys.
    data = {
        "formula": row.get("formula_pretty"),
        "chemsys": row.get("chemsys"),
        "nelements": row.get("nelements"),
        "density": row.get("density"),
        "qe": {
            "status": q.get("status"),
            "candidatepath": q.get("candidate_relpath"),
            "relaxconverged": q.get("relax_converged"),
            "scfconverged": q.get("scf_converged"),
            "elasticpointsok": q.get("elastic_points_ok"),
            "elasticfitrmsgpa": q.get("elastic_fit_rms_gpa"),
            "dftbh": q.get("dft_B_H"),
            "dftgh": q.get("dft_G_H"),
            "dfteh": q.get("dft_E_H"),
            "dftnu": q.get("dft_nu_H"),
            "dftau": q.get("dft_A_U"),
            "finalenergyevatom": q.get("final_energy_ev_atom"),
            "totalforcerybohr": q.get("total_force_ry_bohr"),
        },
    }
    data = _clean_none(data)

    # Keep units unset for robust first uploads; units can be added later after schema stabilizes.
    columns = {
        "formula": None,
        "chemsys": None,
        "nelements": None,
        "density": None,
        "qe.status": None,
        "qe.candidatepath": None,
        "qe.relaxconverged": None,
        "qe.scfconverged": None,
        "qe.elasticpointsok": None,
        "qe.elasticfitrmsgpa": None,
        "qe.dftbh": None,
        "qe.dftgh": None,
        "qe.dfteh": None,
        "qe.dftnu": None,
        "qe.dftau": None,
        "qe.finalenergyevatom": None,
        "qe.totalforcerybohr": None,
    }

    contribution: Dict[str, Any] = {
        "identifier": row["material_id"],
        "data": data,
    }

    if not args.skip_structure and row.get("structure") is not None:
        try:
            from pymatgen.core import Structure

            contribution["structures"] = [Structure.from_dict(row["structure"])]
        except Exception as exc:
            print(f"WARN: structure conversion failed, continuing without structures: {exc}")

    print(f"Project: {args.project}")
    print(f"Uploading material: {row['material_id']} ({row.get('formula_pretty')})")

    client = Client(apikey=key, project=args.project)

    # Access check. If this fails, project doesn't exist or key lacks permission.
    try:
        _ = client.available_query_params()
    except Exception as exc:
        print(f"ERROR: project access failed for '{args.project}': {exc}")
        print("Confirm project name and that your API key belongs to that MP account.")
        return 3

    if args.dry_run:
        print("DRY RUN payload preview:")
        print(json.dumps({"identifier": contribution["identifier"], "data": contribution["data"]}, indent=2))
        return 0

    # Ensure columns exist and upload one contribution.
    client.init_columns(columns)
    resp = client.submit_contributions([contribution])
    print("Upload response:")
    print(resp)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
