#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/st/bin/python}"
RUNNER="$ROOT/scripts/heima_alignment/ab_loss1_shortcut_formal.py"
OUT="/data/zxl/runs/ab_loss1_shortcut_formal"
SESSION="ab_loss1_shortcut_formal"
GROUP="h0_heima_b_probe"
PREPARE_DATA=0
TRAIN=0
EVAL_ONLY=0
RESUME=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --group) GROUP="${2:?missing group}"; shift 2 ;;
    --prepare-data) PREPARE_DATA=1; shift ;;
    --train) TRAIN=1; shift ;;
    --eval-only) EVAL_ONLY=1; shift ;;
    --resume) RESUME=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

mkdir -p "$OUT/logs" "$ROOT/reports"

free_bytes=$(df --output=avail -B1 /data | tail -1 | tr -d " ")
if [[ "$free_bytes" -lt $((80 * 1024 * 1024 * 1024)) ]]; then
  echo "STOP: /data has less than 80GB free" >&2
  df -h /data >&2
  exit 1
fi

if [[ "$PREPARE_DATA" == "1" ]]; then
  "$PYTHON_BIN" "$RUNNER" --prepare-data --group "$GROUP"
fi

if [[ "$DRY_RUN" == "1" ]]; then
  "$PYTHON_BIN" "$RUNNER" --dry-run --group "$GROUP"
  exit 0
fi

if [[ "$EVAL_ONLY" == "1" ]]; then
  echo "Eval-only is reserved for checkpoint-specific follow-up evaluation; no training was started." >&2
  "$PYTHON_BIN" "$RUNNER" --prepare-data --group "$GROUP"
  exit 0
fi

if [[ "$TRAIN" == "0" ]]; then
  TRAIN=1
fi

if [[ "$TRAIN" == "1" ]]; then
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "STOP: tmux session $SESSION already exists" >&2
    tmux list-sessions | grep "$SESSION" >&2 || true
    exit 1
  fi
  cmd=("$PYTHON_BIN" "$RUNNER" --train --group "$GROUP")
  if [[ "$RESUME" == "1" ]]; then
    cmd+=(--resume)
  fi
  printf "%q " "${cmd[@]}" > "$OUT/${GROUP}_command.sh"
  printf "\n" >> "$OUT/${GROUP}_command.sh"
  chmod +x "$OUT/${GROUP}_command.sh"
  tmux new-session -d -s "$SESSION" "cd  && exec ${cmd[*]} >> /logs/.log 2>&1"
  sleep 3
  pgrep -af "ab_loss1_shortcut_formal.py.*--group $GROUP" > "$OUT/${GROUP}_pid.txt" || true
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader > "$OUT/gpu_at_start.txt" || true
  tmux list-sessions | grep "$SESSION" > "$OUT/tmux_session.txt" || true
  cat "$OUT/tmux_session.txt"
fi
