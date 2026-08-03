#!/usr/bin/env bash
set -uo pipefail
ROOT="${REPO_ROOT:-/data/zxl/Heima-qwen7b-teacher-validation-20260803}"
GPU="$1"; QUEUE="$ROOT/runs/teacher_five_points/queues/gpu${GPU}.jsonl"; LOGDIR="$ROOT/logs/teacher_five_points"; mkdir -p "$LOGDIR"
LOG="$LOGDIR/gpu${GPU}.log"; echo "[$(date -Is)] worker start gpu=$GPU" >> "$LOG"
while IFS= read -r job; do
  protocol=$(python3 -c 'import json,sys; print(json.loads(sys.stdin.read())["protocol"])' <<< "$job")
  scale=$(python3 -c 'import json,sys; print(json.loads(sys.stdin.read())["scale"])' <<< "$job")
  echo "[$(date -Is)] pending protocol=$protocol scale=$scale" >> "$LOG"
  while true; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" 2>/dev/null | head -1 | tr -d ' ')
    [ -z "$used" ] && used=999999
    if [ "$used" -lt 1000 ]; then break; fi
    echo "[$(date -Is)] gpu busy gpu=$GPU used=${used}MiB; sleep" >> "$LOG"; sleep 60
  done
  echo "[$(date -Is)] start protocol=$protocol scale=$scale" >> "$LOG"
  CUDA_VISIBLE_DEVICES="$GPU" REPO_ROOT="$ROOT" python3 "$ROOT/scripts/teacher_five_points/teacher_job.py" --gpu "$GPU" --protocol "$protocol" --scale "$scale" >> "$LOG" 2>&1
  rc=$?; echo "[$(date -Is)] finished protocol=$protocol scale=$scale rc=$rc" >> "$LOG"; sleep 3
done < "$QUEUE"
echo "[$(date -Is)] worker done gpu=$GPU" >> "$LOG"
