#!/usr/bin/env bash
set -euo pipefail
tmux kill-session -t stage2_gpu0 2>/dev/null || true
tmux kill-session -t stage2_gpu1 2>/dev/null || true
echo stopped stage2 gpu workers
