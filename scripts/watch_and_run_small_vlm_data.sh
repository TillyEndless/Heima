#!/usr/bin/env bash
set -euo pipefail

export PATH=/data/zxl/conda_envs/nlp-final/bin:$PATH
export HF_HOME=/data/zxl/hf
export HF_HUB_CACHE=/data/zxl/hf/hub
export HF_DATASETS_CACHE=/data/zxl/hf/datasets
export XDG_CACHE_HOME=/data/zxl/.cache
export TMPDIR=/data/zxl/tmp
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL_A=/data/zxl/small_models/Qwen2.5-VL-3B-Instruct
MODEL_B=/data/zxl/small_models/Qwen2.5-0.5B-Instruct
PROJECT=/data/zxl/Heima
BASE_OUT=/data/zxl/runs/heima_small_vlm_l1
LOG_ROOT=/data/zxl/logs/small_vlm_l1
mkdir -p "$BASE_OUT" "$LOG_ROOT" /data/zxl/tmp

while [ ! -f "$MODEL_A/config.json" ] || ! ls "$MODEL_A"/model-*.safetensors >/dev/null 2>&1; do
  date -u
  du -sh "$MODEL_A" 2>/dev/null || true
  sleep 60
done

python - <<'PY'
from pathlib import Path
root = Path("/data/zxl/small_models/Qwen2.5-VL-3B-Instruct")
parts = sorted(root.glob("model-*.safetensors"))
if not parts:
    raise SystemExit("Qwen2.5-VL weights are not materialized")
print("Qwen2.5-VL parts:", [p.name for p in parts])
print("bytes:", sum(p.stat().st_size for p in parts))
PY

cd "$PROJECT"

SMOKE_OUT="$BASE_OUT/smoke"
python scripts/run_data_small_vlm_l1.py \
  --model-a-path "$MODEL_A" \
  --model-b-path "$MODEL_B" \
  --out "$SMOKE_OUT" \
  --seed 42 \
  --s0-steps 2 \
  --s1-steps 2 \
  --joint-steps 2 \
  --batch-size 1 \
  --eval-samples 2 \
  --max-image-side 336 \
  --optimizer adamw \
  --log-every 1 \
  2>&1 | tee "$LOG_ROOT/smoke_$(date -u +%Y%m%d_%H%M%S).log"

RUN_OUT="$BASE_OUT/main_seed42"
python scripts/run_data_small_vlm_l1.py \
  --model-a-path "$MODEL_A" \
  --model-b-path "$MODEL_B" \
  --out "$RUN_OUT" \
  --seed 42 \
  --s0-steps 50 \
  --s1-steps 50 \
  --joint-steps 50 \
  --batch-size 1 \
  --eval-samples 12 \
  --max-image-side 336 \
  --optimizer adafactor \
  --log-every 10 \
  2>&1 | tee "$LOG_ROOT/main_seed42_$(date -u +%Y%m%d_%H%M%S).log"
