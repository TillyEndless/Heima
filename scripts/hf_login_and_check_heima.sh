#!/usr/bin/env bash
set -Eeuo pipefail

export PATH=/data/zxl/conda_envs/nlp-final/bin:$PATH

export HF_HOME=/data/zxl/hf
export HF_HUB_CACHE=/data/zxl/hf/hub
export HF_DATASETS_CACHE=/data/zxl/hf/datasets
export HF_XET_CACHE=/data/zxl/hf/xet
export XDG_CACHE_HOME=/data/zxl/.cache
export TMPDIR=/data/zxl/tmp

export http_proxy=http://127.0.0.1:17890
export https_proxy=http://127.0.0.1:17890
export HTTP_PROXY=http://127.0.0.1:17890
export HTTPS_PROXY=http://127.0.0.1:17890
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost

unset HF_HUB_OFFLINE
unset TRANSFORMERS_OFFLINE

mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$HF_XET_CACHE" "$XDG_CACHE_HOME" "$TMPDIR"

echo "Checking proxy..."
curl -I --proxy http://127.0.0.1:17890 --connect-timeout 15 --max-time 60 https://huggingface.co >/dev/null
echo "Proxy OK."

read -rsp "Paste Hugging Face read token: " HF_TOKEN
echo

hf auth login --token "$HF_TOKEN"
unset HF_TOKEN

hf auth whoami

ls -lh /data/zxl/hf/token
chmod 600 /data/zxl/hf/token

python - <<'PY'
from huggingface_hub import HfApi

api = HfApi()

resources = [
    ("model", "Xkev/Llama-3.2V-11B-cot"),
    ("dataset", "Xkev/LLaVA-CoT-100k"),
    ("model", "shawnricecake/Heima"),
    ("model", "meta-llama/Llama-3.1-8B-Instruct"),
]

failed = False

for kind, repo_id in resources:
    try:
        info = api.dataset_info(repo_id) if kind == "dataset" else api.model_info(repo_id)
        print(f"OK   {kind:7s} {repo_id}")
        print(f"     revision: {info.sha}")
    except Exception as exc:
        failed = True
        print(f"FAIL {kind:7s} {repo_id}")
        print(f"     {type(exc).__name__}: {exc}")

if failed:
    raise SystemExit("Required repository access check failed.")
PY

rm -rf /data/zxl/hf_access_test

hf download meta-llama/Llama-3.1-8B-Instruct \
  config.json \
  --local-dir /data/zxl/hf_access_test

ls -lh /data/zxl/hf_access_test/config.json

echo
echo "HF login and official resource access checks passed."
echo "Next:"
echo "  tmux new -s heima-download"
echo "  bash /data/zxl/run_heima_download_with_proxy.sh"
