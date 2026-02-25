#!/usr/bin/env python3
"""
Bulk upload Aim Materials candidates to MPContribs.

This uploader is conservative:
- uses key-safe data labels (no underscores),
- can skip already-uploaded identifiers,
- verifies post-upload presence by querying identifiers.

Usage:
  python upload_candidates_mpcontrib.py --project aim_materials_v1 --mode all
  python upload_candidates_mpcontrib.py --project aim_materials_v1 --mode elastic_ready
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


def choose_api_key() -> Optional[str]:
    for env_name in ("MPCONTRIBS_API_KEY", "MP_API_KEY"):
        v = os.getenv(env_name, "").strip()
        if v:
            return v
    return None


def clean_none(obj: Any) -> Any:
    if isinstance(obj, dict):
        out: Dict[str, Any] = {}
        for k, v in obj.items():
            vv = clean_none(v)
            if vv is not None:
                out[k] = vv
        return out
    if isinstance(obj, list):
        vals = [clean_none(v) for v in obj]
        return [v for v in vals if v is not None]
    return obj


def get_nested(d: Dict[str, Any], path: Iterable[str], default: Any = None) -> Any:
    cur: Any = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def fetch_identifiers(client: Any, project: str) -> Set[str]:
    ids: Set[str] = set()
    # Keep simple and robust for <=500 contributions.
    try:
        resp = client.contributions.queryContributions(
            project=project,
            _fields=["identifier"],
            _limit=1000,
        ).result()
    except TypeError:
        resp = client.contributions.queryContributions(
            project=project,
            _fields=["identifier"],
        ).result()
    rows = (resp or {}).get("data", [])
    for r in rows:
        ident = r.get("identifier")
        if ident:
            ids.add(ident)
    return ids


def select_rows(rows: List[Dict[str, Any]], mode: str) -> List[Dict[str, Any]]:
    if mode == "all":
        return rows
    if mode == "elastic_ready":
        return [r for r in rows if (r.get("qe_validation") or {}).get("status") == "elastic_ready"]
    raise ValueError(f"Unsupported mode: {mode}")


def build_data_payload(row: Dict[str, Any]) -> Dict[str, Any]:
    q = row.get("qe_validation") or {}

    payload = {
        "formula": row.get("formula_pretty"),
        "chemsys": row.get("chemsys"),
        "nelements": row.get("nelements"),
        "nsites": row.get("nsites"),
        "density": row.get("density"),
        "densityatomic": row.get("density_atomic"),
        "source": "Aim Materials",
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
        "elastic": {
            "bulkvrh": get_nested(row, ["elasticity", "bulk_modulus", "vrh"]),
            "shearvrh": get_nested(row, ["elasticity", "shear_modulus", "vrh"]),
            "young": get_nested(row, ["elasticity", "young_modulus"]),
            "au": row.get("universal_anisotropy"),
        },
        "symmetry": {
            "crystalsystem": get_nested(row, ["symmetry", "crystal_system"]),
            "symbol": get_nested(row, ["symmetry", "symbol"]),
            "number": get_nested(row, ["symmetry", "number"]),
        },
    }
    return clean_none(payload)


def init_columns(client: Any) -> None:
    # Keep units unset for maximum compatibility across mixed data completeness.
    cols = {
        "formula": None,
        "chemsys": None,
        "nelements": None,
        "nsites": None,
        "density": None,
        "densityatomic": None,
        "source": None,
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
        "elastic.bulkvrh": None,
        "elastic.shearvrh": None,
        "elastic.young": None,
        "elastic.au": None,
        "symmetry.crystalsystem": None,
        "symmetry.symbol": None,
        "symmetry.number": None,
    }
    client.init_columns(cols)


def chunked(seq: List[Dict[str, Any]], size: int) -> Iterable[List[Dict[str, Any]]]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument(
        "--data_json",
        default="qe_campaign_v1_local/mp_combined_data_candidates_mixed51_with_qe_status.json",
    )
    ap.add_argument("--mode", choices=["all", "elastic_ready"], default="all")
    ap.add_argument("--batch_size", type=int, default=10)
    ap.add_argument("--skip_existing", action="store_true", default=True)
    ap.add_argument("--no_skip_existing", dest="skip_existing", action="store_false")
    ap.add_argument("--include_structure", action="store_true", default=True)
    ap.add_argument("--no_include_structure", dest="include_structure", action="store_false")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    key = choose_api_key()
    if not key:
        print("ERROR: set MPCONTRIBS_API_KEY (or MP_API_KEY) first")
        return 2

    p = Path(args.data_json)
    if not p.exists():
        print(f"ERROR: not found: {p}")
        return 2

    try:
        from mpcontribs.client import Client
    except Exception as exc:
        print(f"ERROR: mpcontribs-client unavailable: {exc}")
        return 2

    rows = json.loads(p.read_text(encoding="utf-8"))
    rows = select_rows(rows, args.mode)

    client = Client(apikey=key, project=args.project)

    # Access / existence check
    try:
        _ = client.projects.getProjectByName(pk=args.project).result()
    except Exception as exc:
        print(f"ERROR: cannot access project '{args.project}': {exc}")
        return 3

    before_ids = fetch_identifiers(client, args.project)
    print(f"Project: {args.project}")
    print(f"Mode: {args.mode}")
    print(f"Rows selected: {len(rows)}")
    print(f"Existing contributions before upload: {len(before_ids)}")

    contribs: List[Dict[str, Any]] = []
    skipped_existing = 0
    failed_structure = 0

    Structure = None
    if args.include_structure:
        try:
            from pymatgen.core import Structure as _Structure

            Structure = _Structure
        except Exception:
            print("WARN: pymatgen Structure import failed; continuing without structure payload")
            args.include_structure = False

    for row in rows:
        ident = row.get("material_id")
        if not ident:
            continue
        if args.skip_existing and ident in before_ids:
            skipped_existing += 1
            continue

        c: Dict[str, Any] = {"identifier": ident, "data": build_data_payload(row)}
        if args.include_structure and Structure is not None and row.get("structure") is not None:
            try:
                c["structures"] = [Structure.from_dict(row["structure"])]
            except Exception:
                failed_structure += 1
        contribs.append(c)

    print(f"Prepared contributions: {len(contribs)} (skipped existing: {skipped_existing})")
    if failed_structure:
        print(f"Structure conversion skipped for {failed_structure} rows")

    if args.dry_run:
        if contribs:
            example = {"identifier": contribs[0]["identifier"], "data_keys": sorted(contribs[0]["data"].keys())}
            print("Dry-run sample:", example)
        return 0

    if not contribs:
        print("Nothing to upload.")
        return 0

    init_columns(client)

    attempted = 0
    for batch in chunked(contribs, max(1, args.batch_size)):
        attempted += len(batch)
        client.submit_contributions(batch)
        print(f"Submitted batch: {len(batch)} (attempted total: {attempted})")

    after_ids = fetch_identifiers(client, args.project)
    uploaded_now = sorted((set(c["identifier"] for c in contribs) & after_ids) - before_ids)
    missing_now = sorted(set(c["identifier"] for c in contribs) - after_ids)

    print(f"Existing contributions after upload: {len(after_ids)}")
    print(f"Newly present identifiers: {len(uploaded_now)}")
    if missing_now:
        print(f"Still missing after upload: {len(missing_now)}")
        print("Missing IDs:", missing_now[:20])
    else:
        print("All attempted identifiers are present.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

