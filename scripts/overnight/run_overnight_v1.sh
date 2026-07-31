#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-/data2/zhouxiaoling/latent_cot/Heima-overnight-qwen7b}"
HF_HOME="${HF_HOME:-/weights2/zhouxiaoling/hf_cache}"
PYTHON="${PYTHON:-/data2/zhouxiaoling/envs/latentcot/bin/python}"
LOG_DIR="${REPO_ROOT}/logs"
LOG_FILE="${LOG_DIR}/overnight_v1.log"
LOCK_FILE="${REPO_ROOT}/overnight_gpu1.lock"

mkdir -p "${LOG_DIR}" "${HF_HOME}"

trap 'echo "[overnight] failed at line ${LINENO}" >&2' ERR

export HF_HOME
export CUDA_VISIBLE_DEVICES=1
export TOKENIZERS_PARALLELISM=false

cd "${REPO_ROOT}"

{
  echo "[overnight] start $(date -Is)"
  echo "[overnight] repo=${REPO_ROOT}"
  echo "[overnight] hf_home=${HF_HOME}"
  echo "[overnight] python=${PYTHON}"
  echo "[overnight] cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"

  flock -n "${LOCK_FILE}" "${PYTHON}" scripts/latent_cot_overnight_qwen7b.py --phase all

  if [[ "${RUN_STAGE1:-0}" == "1" ]]; then
    echo "[overnight] RUN_STAGE1=1, launching Qwen-7B Stage-1 only smoke"
    flock -n "${LOCK_FILE}.stage1" "${PYTHON}" scripts/latent_cot_overnight_qwen7b.py --phase stage1 --stage1
  else
    echo "[overnight] RUN_STAGE1 is not 1, skipping Qwen Stage-1 smoke"
  fi

  echo "[overnight] done $(date -Is)"
} 2>&1 | tee -a "${LOG_FILE}"
