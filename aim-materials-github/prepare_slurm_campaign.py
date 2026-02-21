"""
Prepare SLURM array-job scripts for a DFT campaign package.

Generates:
- candidate_paths.txt
- submit_relax_array.sbatch
- submit_static_array.sbatch
- submit_elastic_array.sbatch
- submit_all.sh
- README_SLURM.txt
- optional zip archive for transfer
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
from pathlib import Path
from typing import List


def read_csv(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def uniq(seq: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in seq:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def header_lines(
    job_name: str,
    n_tasks: int,
    walltime: str,
    n_array: int,
    partition: str,
    account: str,
) -> List[str]:
    lines = [
        "#!/bin/bash",
        f"#SBATCH -J {job_name}",
        "#SBATCH -N 1",
        f"#SBATCH --ntasks={int(n_tasks)}",
        f"#SBATCH -t {walltime}",
        f"#SBATCH --array=1-{int(n_array)}",
        f"#SBATCH -o logs/{job_name}_%A_%a.out",
        f"#SBATCH -e logs/{job_name}_%A_%a.err",
    ]
    if partition.strip():
        lines.append(f"#SBATCH -p {partition.strip()}")
    if account.strip():
        lines.append(f"#SBATCH -A {account.strip()}")
    return lines


def build_stage_script(
    stage: str,
    root_name: str,
    n_tasks: int,
    walltime: str,
    n_array: int,
    partition: str,
    account: str,
    module_cmds: str,
    vasp_cmd: str,
) -> str:
    job_name = f"{root_name}_{stage}"
    lines = header_lines(job_name, n_tasks, walltime, n_array, partition, account)
    lines += [
        "",
        "set -euo pipefail",
        'ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"',
        "mkdir -p \"$ROOT_DIR/logs\"",
        'CAND_REL="$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$ROOT_DIR/candidate_paths.txt")"',
        'if [[ -z "$CAND_REL" ]]; then',
        '  echo "No candidate path for SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"',
        "  exit 1",
        "fi",
        'CAND_DIR="$ROOT_DIR/$CAND_REL"',
        'if [[ ! -d "$CAND_DIR" ]]; then',
        '  echo "Missing candidate dir: $CAND_DIR"',
        "  exit 1",
        "fi",
        'cd "$CAND_DIR"',
    ]
    if module_cmds.strip():
        lines += ["", "# Module/environment setup", module_cmds.strip()]

    if stage == "relax":
        lines += [
            "",
            "cp INCAR.relax INCAR",
            "cp KPOINTS.relax KPOINTS",
            f"srun {vasp_cmd}",
        ]
    elif stage == "static":
        lines += [
            "",
            "if [[ -f CONTCAR ]]; then cp CONTCAR POSCAR; fi",
            "cp INCAR.static INCAR",
            "cp KPOINTS.static KPOINTS",
            f"srun {vasp_cmd}",
        ]
    elif stage == "elastic":
        lines += [
            "",
            "if [[ -f CONTCAR ]]; then cp CONTCAR POSCAR; fi",
            "cp INCAR.elastic INCAR",
            "cp KPOINTS.static KPOINTS",
            f"srun {vasp_cmd}",
        ]
    else:
        raise ValueError(f"Unknown stage: {stage}")
    lines.append("")
    return "\n".join(lines)


def build_submit_all(root_name: str) -> str:
    relax = f"{root_name}_relax"
    static = f"{root_name}_static"
    elastic = f"{root_name}_elastic"
    return f"""#!/bin/bash
set -euo pipefail

j_relax=$(sbatch submit_relax_array.sbatch | awk '{{print $4}}')
echo "Submitted relax:  $j_relax"

j_static=$(sbatch --dependency=afterok:${{j_relax}} submit_static_array.sbatch | awk '{{print $4}}')
echo "Submitted static: $j_static"

j_elastic=$(sbatch --dependency=afterok:${{j_static}} submit_elastic_array.sbatch | awk '{{print $4}}')
echo "Submitted elastic:$j_elastic"

echo "Chain submitted: $j_relax -> $j_static -> $j_elastic"
"""


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign_dir", required=True)
    ap.add_argument("--root_name", default="")
    ap.add_argument("--n_tasks", type=int, default=32)
    ap.add_argument("--time_relax", default="24:00:00")
    ap.add_argument("--time_static", default="08:00:00")
    ap.add_argument("--time_elastic", default="12:00:00")
    ap.add_argument("--partition", default="")
    ap.add_argument("--account", default="")
    ap.add_argument("--module_cmds", default="module purge\nmodule load vasp")
    ap.add_argument("--vasp_cmd", default="vasp_std")
    ap.add_argument("--make_zip", action=argparse.BooleanOptionalAction, default=True)
    return ap.parse_args()


def main():
    args = parse_args()
    camp = Path(args.campaign_dir).resolve()
    if not camp.is_dir():
        raise RuntimeError(f"Missing campaign dir: {camp}")

    manifest = camp / "campaign_manifest.csv"
    if not manifest.exists():
        raise RuntimeError(f"Missing campaign manifest: {manifest}")

    rows = read_csv(str(manifest))
    cand_dirs_abs = [str(Path(r["campaign_dir"]).resolve()) for r in rows if r.get("campaign_dir")]
    cand_dirs_abs = uniq(cand_dirs_abs)
    cand_dirs_abs = [p for p in cand_dirs_abs if os.path.isdir(p)]
    cand_dirs_abs.sort()
    if not cand_dirs_abs:
        raise RuntimeError("No candidate directories found from campaign_manifest.csv")

    # Relative paths from campaign root for portability.
    cand_rel = [os.path.relpath(p, start=str(camp)) for p in cand_dirs_abs]

    root_name = args.root_name.strip() if args.root_name.strip() else camp.name
    root_name = root_name.replace(" ", "_")
    n = len(cand_rel)

    # candidate_paths.txt
    cand_txt = camp / "candidate_paths.txt"
    with open(cand_txt, "w", encoding="utf-8") as f:
        for p in cand_rel:
            f.write(p.replace("\\", "/") + "\n")

    # stage scripts
    relax = build_stage_script(
        stage="relax",
        root_name=root_name,
        n_tasks=int(args.n_tasks),
        walltime=args.time_relax,
        n_array=n,
        partition=args.partition,
        account=args.account,
        module_cmds=args.module_cmds,
        vasp_cmd=args.vasp_cmd,
    )
    static = build_stage_script(
        stage="static",
        root_name=root_name,
        n_tasks=int(args.n_tasks),
        walltime=args.time_static,
        n_array=n,
        partition=args.partition,
        account=args.account,
        module_cmds=args.module_cmds,
        vasp_cmd=args.vasp_cmd,
    )
    elastic = build_stage_script(
        stage="elastic",
        root_name=root_name,
        n_tasks=int(args.n_tasks),
        walltime=args.time_elastic,
        n_array=n,
        partition=args.partition,
        account=args.account,
        module_cmds=args.module_cmds,
        vasp_cmd=args.vasp_cmd,
    )

    (camp / "submit_relax_array.sbatch").write_text(relax, encoding="utf-8")
    (camp / "submit_static_array.sbatch").write_text(static, encoding="utf-8")
    (camp / "submit_elastic_array.sbatch").write_text(elastic, encoding="utf-8")
    (camp / "submit_all.sh").write_text(build_submit_all(root_name), encoding="utf-8")

    readme = f"""SLURM Campaign Handoff
=====================
campaign_dir={camp}
n_candidates={n}

1) Copy campaign folder to cluster.
2) Edit module/account/partition in *.sbatch if needed.
3) Submit:
   bash submit_all.sh

Generated files:
- candidate_paths.txt
- submit_relax_array.sbatch
- submit_static_array.sbatch
- submit_elastic_array.sbatch
- submit_all.sh
"""
    (camp / "README_SLURM.txt").write_text(readme, encoding="utf-8")

    zip_path = None
    if bool(args.make_zip):
        zip_base = str(camp) + "_handoff"
        zip_path = shutil.make_archive(zip_base, "zip", root_dir=str(camp.parent), base_dir=camp.name)

    print("=" * 72)
    print("SLURM HANDOFF PREP COMPLETE")
    print("=" * 72)
    print(f"campaign_dir={camp}")
    print(f"n_candidates={n}")
    print(f"candidate_paths={cand_txt}")
    print(f"submit_chain={camp / 'submit_all.sh'}")
    if zip_path:
        print(f"zip_archive={zip_path}")


if __name__ == "__main__":
    main()

