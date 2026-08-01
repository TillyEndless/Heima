#!/usr/bin/env bash
set -Eeuo pipefail
export REPO_ROOT="${REPO_ROOT:-/data2/zhouxiaoling/latent_cot/Heima-qwen7b-stage-validity-pilot}"
export HF_HOME="${HF_HOME:-/weights2/zhouxiaoling/hf_cache}"
export CUDA_VISIBLE_DEVICES=1
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false
PY="${PY:-/data2/zhouxiaoling/envs/latentcot/bin/python}"
cd "$REPO_ROOT"
mkdir -p reports/direct_generation_closure status/direct_generation_closure logs
run_phase() {
  local phase="$1"; local status_name="$2"
  if [[ -f "status/direct_generation_closure/${status_name}.json" ]] && grep -q '"status": "complete"' "status/direct_generation_closure/${status_name}.json"; then
    echo "[skip] $phase already complete"; return 0
  fi
  echo "[run] $phase $(date -Is)"
  "$PY" scripts/experiments/direct_generation_closure.py "$phase"
  sleep 2
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | tee -a logs/direct_generation_closure.log
}
{
  run_phase existing existing_direct
  run_phase cache cache_divergence
  run_phase continue_direct direct_continuation
  run_phase o0 o0_reeval
  run_phase final final
} 2>&1 | tee -a logs/direct_generation_closure.log
