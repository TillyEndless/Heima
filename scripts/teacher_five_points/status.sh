#!/usr/bin/env bash
ROOT="${REPO_ROOT:-/data/zxl/Heima-qwen7b-teacher-validation-20260803}"
echo '== tmux =='; tmux ls 2>/dev/null | grep -E 'teacher5|teacher_five' || true
echo '== gpu =='; nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
echo '== logs =='; for f in "$ROOT"/logs/teacher_five_points/*.log; do [ -f "$f" ] && echo "--- $f" && tail -5 "$f"; done
