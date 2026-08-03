#!/usr/bin/env bash
set -euo pipefail
ROOT="${REPO_ROOT:-/data/zxl/Heima-qwen7b-teacher-validation-20260803}"; cd "$ROOT"
for gpu in 0 1 2 3 4 5 6 7; do
  sess="teacher5_gpu${gpu}"; tmux has-session -t "$sess" 2>/dev/null && continue
  tmux new-session -d -s "$sess" "bash $ROOT/scripts/teacher_five_points/gpu_worker.sh $gpu"
done
PYTHONPATH=. python3 scripts/teacher_five_points/aggregate.py
