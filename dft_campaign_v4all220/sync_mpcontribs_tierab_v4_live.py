#!/usr/bin/env python3
"""
Sync live TierAB v4 relax status to MPContribs entries (aim-v4-*).

Status mapping:
  - pass list -> status=passed, relaxconverged=True
  - fail list -> status=failed, relaxconverged=False
  - current running + untouched -> status=pending, relaxconverged=pending

Both nested `data.qe.*` and flattened `data.cqe*` keys are updated.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple


def choose_api_key() -> str:
    for key in ("MPCONTRIBS_API_KEY", "MP_API_KEY"):
        val = os.getenv(key, "").strip()
        if val:
            return val
    raise RuntimeError("Set MPCONTRIBS_API_KEY or MP_API_KEY.")


def read_nonempty_lines(path: Path) -> List[str]:
    if not path.exists():
        return []
    return [x.strip() for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def parse_running(log_path: Path, passed: set[str], failed: set[str]) -> str:
    if not log_path.exists():
        return ""
    last_run = ""
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("[RUN ] "):
            last_run = line[len("[RUN ] ") :].strip()
    if last_run and last_run not in passed and last_run not in failed:
        return last_run
    return ""


def rel_to_identifier(rel: str) -> str:
    # rel example: "2el/009_gen_00127_CsF" -> "aim-v4-2el-gen_00127"
    parts = rel.split("/", 1)
    if len(parts) != 2:
        raise ValueError(f"invalid candidate relpath: {rel}")
    set_name, tail = parts[0].strip(), parts[1].strip()
    m = re.search(r"(gen_\d+)", tail)
    if not m:
        raise ValueError(f"cannot parse gen id from: {rel}")
    gen_id = m.group(1)
    return f"aim-v4-{set_name}-{gen_id}"


def desired_status_for(rel: str, passed: set[str], failed: set[str], running: str) -> Tuple[str, str]:
    if rel in passed:
        return "passed", "True"
    if rel in failed:
        return "failed", "False"
    # running + pending both map to pending in MPContribs
    return "pending", "pending"


def main() -> int:
    ap = argparse.ArgumentParser(description="Sync live TierAB v4 status to MPContribs.")
    ap.add_argument("--project", default="aim_materials_v1")
    ap.add_argument("--full_list", default="dft_shortlist_from_271/candidate_paths_tierAB_v4.txt")
    ap.add_argument("--pass_file", default="dft_campaign_v4all220/relax_passed_tierab_v4.txt")
    ap.add_argument("--fail_file", default="dft_campaign_v4all220/relax_failed_tierab_v4.txt")
    ap.add_argument("--stage_log", default="dft_campaign_v4all220/relax_stage_tierab_v4.log")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    full_list = read_nonempty_lines(Path(args.full_list))
    passed = set(read_nonempty_lines(Path(args.pass_file)))
    failed = set(line.split(",", 1)[0].strip() for line in read_nonempty_lines(Path(args.fail_file)))
    running = parse_running(Path(args.stage_log), passed, failed)

    desired_by_ident: Dict[str, Tuple[str, str]] = {}
    for rel in full_list:
        ident = rel_to_identifier(rel)
        desired_by_ident[ident] = desired_status_for(rel, passed, failed, running)

    print(
        f"tierab_v4 total={len(full_list)} passed={len(passed)} failed={len(failed)} "
        f"running={(1 if running else 0)} pending={len(full_list)-len(passed)-len(failed)-(1 if running else 0)}"
    )
    if running:
        print("running_candidate:", running, "=>", rel_to_identifier(running))

    from mpcontribs.client import Client

    api_key = choose_api_key()
    client = Client(apikey=api_key, project=args.project)

    idents = list(desired_by_ident.keys())
    remote_rows: List[Dict] = []
    # chunk queries for safety
    for i in range(0, len(idents), 100):
        batch = idents[i : i + 100]
        resp = client.contributions.queryContributions(
            project=args.project,
            identifier_in=batch,
            _fields=["id", "identifier", "data"],
            _limit=1500,
        ).result()
        remote_rows.extend(resp.get("data", []))

    remote_by_ident = {str(r.get("identifier")): r for r in remote_rows if r.get("identifier")}
    missing = [ident for ident in idents if ident not in remote_by_ident]
    if missing:
        print("WARNING: missing identifiers on MPContribs:", len(missing))
        for x in missing[:10]:
            print(" ", x)

    updates: List[Tuple[str, str, Dict]] = []
    unchanged = 0
    for ident, (want_status, want_relax) in desired_by_ident.items():
        rec = remote_by_ident.get(ident)
        if not rec:
            continue
        cid = rec.get("id")
        data = rec.get("data") or {}
        qe = data.get("qe")
        if not isinstance(qe, dict):
            qe = {}

        have_status = str(qe.get("status", ""))
        have_relax = str(qe.get("relaxconverged", ""))
        have_scf = str(qe.get("scfconverged", ""))
        have_elp = str(qe.get("elasticpointsok", ""))
        have_cqe_status = str(data.get("cqestatus", ""))
        have_cqe_relax = str(data.get("cqerelaxconverged", ""))
        have_cqe_scf = str(data.get("cqescfconverged", ""))
        have_cqe_elp = str(data.get("cqeelasticpointsok", ""))

        same = (
            have_status == want_status
            and have_relax == want_relax
            and have_scf == "pending"
            and have_elp == "0"
            and have_cqe_status == want_status
            and have_cqe_relax == want_relax
            and have_cqe_scf == "pending"
            and have_cqe_elp == "0"
        )
        if same:
            unchanged += 1
            continue

        data_new = dict(data)
        qe_new = dict(qe)
        qe_new["status"] = want_status
        qe_new["relaxconverged"] = want_relax
        qe_new["scfconverged"] = "pending"
        qe_new["elasticpointsok"] = "0"
        data_new["qe"] = qe_new
        data_new["cqestatus"] = want_status
        data_new["cqerelaxconverged"] = want_relax
        data_new["cqescfconverged"] = "pending"
        data_new["cqeelasticpointsok"] = "0"

        updates.append((ident, str(cid), {"data": data_new}))

    print(
        f"mp_targeted={len(desired_by_ident)} found={len(remote_by_ident)} "
        f"update={len(updates)} unchanged={unchanged} mode={'DRY-RUN' if args.dry_run else 'WRITE'}"
    )

    if not args.dry_run:
        for ident, cid, payload in updates:
            client.contributions.updateContributionById(pk=cid, contribution=payload).result()

    # Verify status counts on targeted v4 subset
    verify_rows = []
    for i in range(0, len(idents), 100):
        batch = idents[i : i + 100]
        resp = client.contributions.queryContributions(
            project=args.project,
            identifier_in=batch,
            _fields=["identifier", "data"],
            _limit=1500,
        ).result()
        verify_rows.extend(resp.get("data", []))

    counts: Dict[str, int] = {"passed": 0, "failed": 0, "pending": 0, "other": 0}
    for row in verify_rows:
        qe = (row.get("data") or {}).get("qe") or {}
        st = str(qe.get("status", "other"))
        if st in counts:
            counts[st] += 1
        else:
            counts["other"] += 1
    print("verify_v4_status_counts:", counts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

