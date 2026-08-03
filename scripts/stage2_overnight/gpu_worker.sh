#!/usr/bin/env bash
set -euo pipefail
GPU="$1"; QUEUE="$2"; REPO_ROOT="${REPO_ROOT:-$(pwd)}"; PY="${PY:-/data2/zhouxiaoling/envs/latentcot/bin/python}"
export HF_HOME="${HF_HOME:-/weights2/zhouxiaoling/hf_cache}" HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}" HF_HUB_DISABLE_XET=1 TOKENIZERS_PARALLELISM=false
mkdir -p logs/stage2_overnight status/stage2_overnight reports/stage2_overnight
LOG="logs/stage2_overnight/worker_gpu${GPU}.log"; FAIL="reports/stage2_overnight/failure_registry.jsonl"
echo "[$(date -Is)] worker start gpu=$GPU queue=$QUEUE" | tee -a "$LOG"
while IFS=, read -r job mode adapter steps lambda; do
  [[ -z "${job:-}" || "$job" = job ]] && continue
  while true; do used=$(nvidia-smi --id="$GPU" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' '); [[ "$used" -lt "${GPU_IDLE_MEM_MB:-20000}" ]] && break; echo "[$(date -Is)] gpu=$GPU busy used=${used}MiB" | tee -a "$LOG"; sleep 60; done
  export CUDA_VISIBLE_DEVICES="$GPU" REPO_ROOT="$REPO_ROOT"
  echo "[$(date -Is)] start $job mode=$mode adapter=$adapter steps=$steps lambda=$lambda" | tee -a "$LOG"
  set +e; "$PY" scripts/stage2_overnight/stage2_runner.py train --job "$job" --mode "$mode" --adapter-kind "$adapter" --steps "$steps" --lambda-decode "$lambda" >> "$LOG" 2>&1; rc=$?; set -e
  if [[ "$rc" != 0 ]]; then echo "{\"time\":\"$(date -Is)\",\"gpu\":$GPU,\"job\":\"$job\",\"rc\":$rc}" >> "$FAIL"; fi
  "$PY" scripts/stage2_overnight/stage2_runner.py aggregate >> "$LOG" 2>&1 || true
  echo "[$(date -Is)] finished $job rc=$rc" | tee -a "$LOG"
done < "$QUEUE"
echo "[$(date -Is)] worker done gpu=$GPU" | tee -a "$LOG"
