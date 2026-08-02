#!/usr/bin/env bash
ROOT="${REPO_ROOT:-/data/zxl/Heima-qwen7b-overnight-20260803}"
echo "== tmux =="; tmux ls 2>/dev/null | grep overnight || true
echo "== gpu =="; nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
echo "== logs =="; for f in "$ROOT"/logs/overnight_structural_sweep/*.log; do [ -f "$f" ] && echo "--- $f" && tail -5 "$f"; done
