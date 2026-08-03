#!/usr/bin/env bash
tmux kill-session -t progressive_gpu0 2>/dev/null || true
tmux kill-session -t progressive_gpu1 2>/dev/null || true
