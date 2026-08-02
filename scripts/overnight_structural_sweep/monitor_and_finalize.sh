#!/usr/bin/env bash
set -uo pipefail
ROOT="${REPO_ROOT:-/data/zxl/Heima-qwen7b-overnight-20260803}"
while true; do python3 "$ROOT/scripts/overnight_structural_sweep/aggregate_results.py" >/dev/null 2>&1 || true; sleep 600; done
