#!/usr/bin/env bash
set -euo pipefail
echo '== tmux =='; tmux ls 2>/dev/null | grep stage2 || true
echo '== gpu =='; nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
echo '== processes =='; pgrep -af 'stage2_runner.py train|gpu_worker.sh|stage2_runner.py gpt2-audit|stage2_monitor' || true
echo '== status =='; for f in status/stage2_overnight/*.json; do [ -f "$f" ] && echo ---$f && cat "$f"; done
echo '== metrics =='; [ -f reports/stage2_overnight/all_metrics.csv ] && cat reports/stage2_overnight/all_metrics.csv || true
