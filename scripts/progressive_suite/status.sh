#!/usr/bin/env bash
set -euo pipefail
cd "${REPO_ROOT:-$(pwd)}"
echo "== tmux =="; tmux ls 2>/dev/null | grep progressive || true
echo "== gpu =="; nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
echo "== worker status =="; for f in status/progressive_suite/*.json; do [ -f "$f" ] && echo ---$f && cat "$f"; done
echo "== logs =="; tail -40 logs/progressive_suite/worker_gpu0.log 2>/dev/null || true; tail -40 logs/progressive_suite/worker_gpu1.log 2>/dev/null || true
