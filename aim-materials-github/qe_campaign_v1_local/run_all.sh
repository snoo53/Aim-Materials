#!/usr/bin/env bash
set -euo pipefail
QE_CMD="${1:-pw.x}"
NPROC="${2:-1}"

bash run_relax.sh "$QE_CMD" "$NPROC"
bash run_scf.sh "$QE_CMD" "$NPROC"
bash run_elastic.sh "$QE_CMD" "$NPROC"
