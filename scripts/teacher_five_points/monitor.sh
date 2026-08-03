#!/usr/bin/env bash
set -uo pipefail
ROOT="${REPO_ROOT:-/data/zxl/Heima-qwen7b-teacher-validation-20260803}"
while true; do python3 "$ROOT/scripts/teacher_five_points/aggregate.py" >/dev/null 2>&1 || true; sleep 600; done
