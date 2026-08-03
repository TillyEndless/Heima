#!/usr/bin/env bash
set -euo pipefail
echo "teacher-five-points worktree prepared: $(pwd)"
git branch --show-current
git log -1 --oneline
