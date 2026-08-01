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
mkdir -p reports/stage1_repair_v2 status/stage1_repair_v2 logs
run_phase() {
  local phase="$1"; local status_name="$2"
  if [[ -f "status/stage1_repair_v2/${status_name}.json" ]] && grep -q '"status": "complete"' "status/stage1_repair_v2/${status_name}.json"; then
    echo "[skip] $phase already complete"; return 0
  fi
  echo "[run] $phase $(date -Is)"
  "$PY" scripts/experiments/qwen7b_stage1_repair_v2.py "$phase"
  sleep 2
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | tee -a logs/qwen7b_stage1_repair_v2.log
}
{
  run_phase transition transition_audit
  run_phase ladder prefix_ladder
  run_phase direct direct_answer
  run_phase o1o2 o2_answer_heavy
} 2>&1 | tee -a logs/qwen7b_stage1_repair_v2.log
"$PY" scripts/experiments/qwen7b_stage1_repair_v2.py final
