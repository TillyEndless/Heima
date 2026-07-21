#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
CONFIG="${ROOT}/configs/heima_aligned/ab_loss1_qwen_vl3b.yaml"
RUN_ID="smoke_$(date -u +%Y%m%d_%H%M%S)"
ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --run-id) RUN_ID="$2"; shift 2 ;;
    *) ARGS+=("$1"); shift ;;
  esac
done
"${PYTHON:-python3}" "${SCRIPT_DIR}/pipeline.py" --config "$CONFIG" --mode heima_scaled_baseline --run-id "$RUN_ID" --smoke --dry-run "${ARGS[@]}"
