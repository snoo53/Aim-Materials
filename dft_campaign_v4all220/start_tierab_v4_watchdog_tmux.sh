#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="${1:-dft_tierab_v4_watchdog}"
CAMPAIGN_DIR="/mnt/c/Users/sunwo/Desktop/aim-materials/dft_campaign_v4all220"

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "[INFO] tmux session already running: $SESSION_NAME"
  tmux ls | grep "^${SESSION_NAME}:" || true
  exit 0
fi

tmux_args=(-d -s "$SESSION_NAME" -c "$CAMPAIGN_DIR")
if [[ -n "${MPCONTRIBS_API_KEY:-}" ]]; then
  tmux_args+=(-e "MPCONTRIBS_API_KEY=$MPCONTRIBS_API_KEY")
fi
if [[ -n "${MP_API_KEY:-}" ]]; then
  tmux_args+=(-e "MP_API_KEY=$MP_API_KEY")
fi

tmux new-session "${tmux_args[@]}" "bash watch_tierab_v4_until_done.sh"
sleep 1
echo "[INFO] started tmux session: $SESSION_NAME"
tmux ls | grep "^${SESSION_NAME}:"
