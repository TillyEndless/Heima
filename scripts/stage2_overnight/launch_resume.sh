#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-$(pwd)}"
export REPO_ROOT
export HF_HOME="${HF_HOME:-/weights2/zhouxiaoling/hf_cache}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false
export STAGE2_MANIFEST="${STAGE2_MANIFEST:-/data2/zhouxiaoling/latent_cot/Heima-overnight-qwen7b/data/processed_large/qwen7b_stage1/smoke1k.json}"
export STAGE2_SAVE_BEST="${STAGE2_SAVE_BEST:-0}"
export STAGE2_SAVE_FINAL="${STAGE2_SAVE_FINAL:-1}"
export STAGE2_EVAL_EVERY="${STAGE2_EVAL_EVERY:-200}"
export STAGE2_EVAL_N="${STAGE2_EVAL_N:-64}"
export STAGE2_MAX_K="${STAGE2_MAX_K:-64}"
export GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-120}"
export GPU_IDLE_MEM_MB="${GPU_IDLE_MEM_MB:-20000}"
export PY="${PY:-/data2/zhouxiaoling/envs/latentcot/bin/python}"
mkdir -p logs/stage2_overnight status/stage2_overnight reports/stage2_overnight checkpoints/stage2_overnight
tmux kill-session -t stage2_resume_gpu1 2>/dev/null || true
tmux kill-session -t stage2_resume_gpu0 2>/dev/null || true
tmux new-session -d -s stage2_resume_gpu1 "cd "$REPO_ROOT" && env REPO_ROOT="$REPO_ROOT" HF_HOME="$HF_HOME" HF_ENDPOINT="$HF_ENDPOINT" HF_HUB_DISABLE_XET=1 TOKENIZERS_PARALLELISM=false STAGE2_MANIFEST="$STAGE2_MANIFEST" STAGE2_SAVE_BEST=$STAGE2_SAVE_BEST STAGE2_SAVE_FINAL=$STAGE2_SAVE_FINAL STAGE2_EVAL_EVERY=$STAGE2_EVAL_EVERY STAGE2_EVAL_N=$STAGE2_EVAL_N STAGE2_MAX_K=$STAGE2_MAX_K GPU_POLL_SECONDS=$GPU_POLL_SECONDS GPU_IDLE_MEM_MB=$GPU_IDLE_MEM_MB PY="$PY" bash scripts/stage2_overnight/gpu_worker.sh 1 configs/stage2_overnight/resume_gpu1_queue.csv"
tmux new-session -d -s stage2_resume_gpu0 "cd "$REPO_ROOT" && env REPO_ROOT="$REPO_ROOT" HF_HOME="$HF_HOME" HF_ENDPOINT="$HF_ENDPOINT" HF_HUB_DISABLE_XET=1 TOKENIZERS_PARALLELISM=false STAGE2_MANIFEST="$STAGE2_MANIFEST" STAGE2_SAVE_BEST=$STAGE2_SAVE_BEST STAGE2_SAVE_FINAL=$STAGE2_SAVE_FINAL STAGE2_EVAL_EVERY=$STAGE2_EVAL_EVERY STAGE2_EVAL_N=$STAGE2_EVAL_N STAGE2_MAX_K=$STAGE2_MAX_K GPU_POLL_SECONDS=$GPU_POLL_SECONDS GPU_IDLE_MEM_MB=$GPU_IDLE_MEM_MB PY="$PY" bash scripts/stage2_overnight/gpu_worker.sh 0 configs/stage2_overnight/resume_gpu0_queue.csv"
cat > reports/stage2_overnight/resume_launch_runtime.json <<EOF
{"time":"$(date -Is)","sessions":["stage2_resume_gpu1","stage2_resume_gpu0"],"manifest":"$STAGE2_MANIFEST","save_best":"$STAGE2_SAVE_BEST","save_final":"$STAGE2_SAVE_FINAL","eval_every":"$STAGE2_EVAL_EVERY","eval_n":"$STAGE2_EVAL_N","max_k":"$STAGE2_MAX_K"}
EOF
tmux ls | grep stage2_resume
