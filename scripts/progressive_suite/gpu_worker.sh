#!/usr/bin/env bash
set -euo pipefail
GPU="$1"; QUEUE="$2"
export REPO_ROOT="${REPO_ROOT:-$(pwd)}" HF_HOME="${HF_HOME:-/weights2/zhouxiaoling/hf_cache}" HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}" HF_HUB_DISABLE_XET=1 TOKENIZERS_PARALLELISM=false
PY="${PY:-/data2/zhouxiaoling/envs/latentcot/bin/python}"
mkdir -p logs/progressive_suite status/progressive_suite reports/progressive_suite
LOG="logs/progressive_suite/worker_gpu${GPU}.log"; STATUS="status/progressive_suite/worker_gpu${GPU}.json"; FAIL="reports/progressive_suite/failure_registry.jsonl"
echo "[$(date -Is)] worker start gpu=$GPU queue=$QUEUE" | tee -a "$LOG"
while IFS=, read -r protocol scale steps; do
  [[ -z "${protocol:-}" || "$protocol" = protocol ]] && continue
  while true; do used=$(nvidia-smi --id="$GPU" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d " "); [[ "$used" -lt "${GPU_IDLE_MEM_MB:-20000}" ]] && break; echo "[$(date -Is)] gpu=$GPU busy used=${used}MiB" | tee -a "$LOG"; sleep 60; done
  export CUDA_VISIBLE_DEVICES="$GPU"
  printf "{\"time\":\"%s\",\"gpu\":%s,\"protocol\":\"%s\",\"scale\":\"%s\",\"state\":\"running\"}\n" "$(date -Is)" "$GPU" "$protocol" "$scale" > "$STATUS"
  echo "[$(date -Is)] start $protocol $scale steps=$steps" | tee -a "$LOG"
  set +e; "$PY" scripts/progressive_suite/train_protocol.py train --protocol "$protocol" --scale "$scale" --steps "$steps" >> "$LOG" 2>&1; rc=$?; set -e
  if [[ "$rc" != 0 ]]; then echo "{\"time\":\"$(date -Is)\",\"gpu\":$GPU,\"protocol\":\"$protocol\",\"scale\":\"$scale\",\"rc\":$rc}" >> "$FAIL"; fi
  "$PY" scripts/progressive_suite/train_protocol.py aggregate >> "$LOG" 2>&1 || true
  printf "{\"time\":\"%s\",\"gpu\":%s,\"protocol\":\"%s\",\"scale\":\"%s\",\"state\":\"finished\",\"rc\":%s}\n" "$(date -Is)" "$GPU" "$protocol" "$scale" "$rc" > "$STATUS"
  echo "[$(date -Is)] finished $protocol $scale rc=$rc" | tee -a "$LOG"
done < "$QUEUE"
echo "[$(date -Is)] worker done gpu=$GPU" | tee -a "$LOG"
