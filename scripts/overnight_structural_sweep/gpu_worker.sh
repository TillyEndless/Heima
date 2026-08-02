#!/usr/bin/env bash
set -uo pipefail
ROOT="${REPO_ROOT:-/data/zxl/Heima-qwen7b-overnight-20260803}"
LANE="$1"; GPU="$2"; PROTOCOL="$3"
QUEUE="$ROOT/runs/overnight_structural_sweep/queues/lane${LANE}.jsonl"
LOGDIR="$ROOT/logs/overnight_structural_sweep"; mkdir -p "$LOGDIR"
LOG="$LOGDIR/lane${LANE}.log"
echo "[$(date -Is)] worker start lane=$LANE gpu=$GPU protocol=$PROTOCOL" >> "$LOG"
while IFS= read -r job; do
  scale=$(python3 -c 'import json,sys; print(json.loads(sys.stdin.read())["scale"])' <<< "$job")
  echo "[$(date -Is)] pending $scale" >> "$LOG"
  while true; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" 2>/dev/null | head -1 | tr -d ' ')
    [ -z "$used" ] && used=999999
    if [ "$used" -lt 1000 ]; then break; fi
    echo "[$(date -Is)] gpu busy lane=$LANE gpu=$GPU used=${used}MiB; sleep" >> "$LOG"
    sleep 60
  done
  echo "[$(date -Is)] start $scale" >> "$LOG"
  CUDA_VISIBLE_DEVICES="$GPU" REPO_ROOT="$ROOT" python3 "$ROOT/scripts/overnight_structural_sweep/run_protocol.py" --lane "$LANE" --gpu "$GPU" --protocol "$PROTOCOL" --scale "$scale" >> "$LOG" 2>&1
  rc=$?
  echo "[$(date -Is)] finished $scale rc=$rc" >> "$LOG"
  sleep 3
done < "$QUEUE"
echo "[$(date -Is)] worker done lane=$LANE" >> "$LOG"
