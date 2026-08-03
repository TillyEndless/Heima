#!/usr/bin/env bash
set -euo pipefail
for s in $(tmux ls 2>/dev/null | awk -F: '/^teacher5_/ {print $1}'); do tmux kill-session -t "$s"; done
echo "stopped teacher5 worker sessions owned by this sweep"
