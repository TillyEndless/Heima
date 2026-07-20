#!/usr/bin/env bash
set -Eeuo pipefail

export PATH=/data/zxl/conda_envs/nlp-final/bin:$PATH

export http_proxy=http://127.0.0.1:17890
export https_proxy=http://127.0.0.1:17890
export HTTP_PROXY=http://127.0.0.1:17890
export HTTPS_PROXY=http://127.0.0.1:17890
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost

unset HF_HUB_OFFLINE
unset TRANSFORMERS_OFFLINE

export HF_HOME=/data/zxl/hf
export HF_HUB_CACHE=/data/zxl/hf/hub
export HF_DATASETS_CACHE=/data/zxl/hf/datasets
export TRANSFORMERS_CACHE=/data/zxl/hf/transformers
export HF_XET_CACHE=/data/zxl/hf/xet
export XDG_CACHE_HOME=/data/zxl/.cache
export TORCH_HOME=/data/zxl/.cache/torch
export TMPDIR=/data/zxl/tmp
export TEMP=/data/zxl/tmp
export TMP=/data/zxl/tmp

export HF_HUB_ETAG_TIMEOUT=60
export HF_HUB_DOWNLOAD_TIMEOUT=600
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false

mkdir -p \
  "$HF_HOME" \
  "$HF_HUB_CACHE" \
  "$HF_DATASETS_CACHE" \
  "$HF_XET_CACHE" \
  "$XDG_CACHE_HOME" \
  "$TORCH_HOME" \
  "$TMPDIR"

echo "Proxy:"
env | grep -i proxy | sort

echo
echo "Hugging Face connectivity:"
curl -I --proxy http://127.0.0.1:17890 --connect-timeout 15 --max-time 60 https://huggingface.co

echo
echo "Hugging Face auth:"
hf auth whoami || echo "WARNING: hf auth whoami failed; continuing because token-backed downloads will verify access."

exec bash /data/zxl/download_official_heima.sh
