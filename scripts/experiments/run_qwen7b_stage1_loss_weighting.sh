#!/usr/bin/env bash
set -euo pipefail
export REPO_ROOT="${REPO_ROOT:-/data2/zhouxiaoling/latent_cot/Heima-qwen7b-stage-validity-pilot}"
export HF_HOME="${HF_HOME:-/weights2/zhouxiaoling/hf_cache}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export CUDA_VISIBLE_DEVICES=1
cd "$REPO_ROOT"
PHASE="${1:-all}"
exec /data2/zhouxiaoling/envs/latentcot/bin/python scripts/experiments/qwen7b_stage1_loss_weighting.py "$PHASE"
