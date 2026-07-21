#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
PYTHON_BIN=${PYTHON:-python3}
DATASET_ROOT=${DATASET_ROOT:-/data/zxl/official_heima/datasets/LLaVA-CoT-100k}
RUN_ROOT=${RUN_ROOT:-/data/zxl/runs/heima_ab_loss1_mini_acceptance_v1}
CONFIG=${CONFIG:-${ROOT}/configs/heima_aligned/ab_loss1_qwen_vl3b.yaml}
MODE=${MODE:-heima_scaled_baseline}
RESUME=0
STAGE=""
EVAL_ONLY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --resume) RESUME=1; shift ;;
    --stage) STAGE="$2"; shift 2 ;;
    --eval-only) EVAL_ONLY=1; shift ;;
    --dataset-root) DATASET_ROOT="$2"; shift 2 ;;
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --config) CONFIG="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done
mkdir -p "${RUN_ROOT}" reports docs/heima_alignment
"${PYTHON_BIN}" "${SCRIPT_DIR}/audit_mini_acceptance_dataset.py" \
  --dataset-root "${DATASET_ROOT}" \
  --out "${ROOT}/docs/heima_alignment/mini_acceptance_dataset_audit.json"
# The current implementation intentionally delegates only after the real-data audit passes.
# Full training integration should be enabled in a follow-up commit after the full dataset is present.
ARGS=(--config "$CONFIG" --mode "$MODE" --run-id heima_ab_loss1_mini_acceptance_v1 --output-root /data/zxl/runs)
if [[ "$RESUME" == 1 ]]; then ARGS+=(--resume); fi
if [[ -n "$STAGE" ]]; then ARGS+=(--from-stage "$STAGE" --to-stage "$STAGE"); fi
if [[ "$EVAL_ONLY" == 1 ]]; then ARGS+=(--skip-train); fi
"${PYTHON_BIN}" "${ROOT}/scripts/heima_aligned/pipeline.py" "${ARGS[@]}"
