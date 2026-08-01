#!/usr/bin/env bash
set -Eeuo pipefail
export REPO_ROOT="${REPO_ROOT:-/data2/zhouxiaoling/latent_cot/Heima-qwen7b-stage-validity-pilot}"
export ASSET_ROOT="${ASSET_ROOT:-/data2/zhouxiaoling/latent_cot/Heima-overnight-qwen7b}"
export HF_HOME="${HF_HOME:-/weights2/zhouxiaoling/hf_cache}"
export CUDA_VISIBLE_DEVICES=1
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false
PY="${PY:-/data2/zhouxiaoling/envs/latentcot/bin/python}"
cd "$REPO_ROOT"
mkdir -p logs status/stage_validity reports/stage_validity
run_phase() {
  local phase="$1"
  local status_name="$2"
  if [[ -f "status/stage_validity/${status_name}.json" ]] && grep -q '"status": "complete"' "status/stage_validity/${status_name}.json"; then
    echo "[skip] ${phase} already complete"
    return 0
  fi
  echo "[run] ${phase} $(date -Is)"
  "$PY" scripts/experiments/qwen7b_stage_validity_pilot.py "$phase"
  sleep 2
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tee -a logs/qwen7b_stage_validity_pilot.log
}
{
  run_phase inventory prior_inventory
  run_phase gate_a gate_a_reload
  run_phase gate_b gate_b_loss
  run_phase gate_c gate_c_length
  run_phase gate_d gate_d_overfit
} 2>&1 | tee -a logs/qwen7b_stage_validity_pilot.log
"$PY" scripts/experiments/qwen7b_stage_validity_pilot.py final
