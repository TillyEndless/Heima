#!/usr/bin/env bash
set -Eeuo pipefail

export PATH=/data/zxl/conda_envs/nlp-final/bin:$PATH

ROOT=/data/zxl/official_heima
MODEL_ROOT="$ROOT/models"
CKPT_ROOT="$ROOT/checkpoints"
LOG_ROOT="$ROOT/logs"
REPORT_DIR=/data/zxl/Heima/reports/official_heima_reproduction

export HF_HOME=/data/zxl/hf
export HF_HUB_CACHE=/data/zxl/hf/hub
export HF_DATASETS_CACHE=/data/zxl/hf/datasets
export HF_XET_CACHE=/data/zxl/hf/xet
export XDG_CACHE_HOME=/data/zxl/.cache
export TMPDIR=/data/zxl/tmp
export TEMP=/data/zxl/tmp
export TMP=/data/zxl/tmp

export http_proxy=http://127.0.0.1:17890
export https_proxy=http://127.0.0.1:17890
export HTTP_PROXY=http://127.0.0.1:17890
export HTTPS_PROXY=http://127.0.0.1:17890
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost

unset HF_HUB_OFFLINE
unset TRANSFORMERS_OFFLINE
export HF_HUB_DISABLE_XET=1
export HF_HUB_ETAG_TIMEOUT=60
export HF_HUB_DOWNLOAD_TIMEOUT=1800
export TOKENIZERS_PARALLELISM=false

mkdir -p "$MODEL_ROOT" "$CKPT_ROOT" "$LOG_ROOT" "$REPORT_DIR" "$TMPDIR"
exec > >(tee -a "$LOG_ROOT/models_priority_$(date +%Y%m%d_%H%M%S).log") 2>&1

echo "============================================================"
echo "Official Heima model/checkpoint priority download"
echo "Started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "============================================================"
df -h /data
env | grep -i proxy | sort
hf auth whoami || echo "WARNING: hf auth whoami failed; repo download will verify token access."

python - <<'PY' > /data/zxl/official_heima/model_revisions.env.tmp
from huggingface_hub import HfApi
api = HfApi()
repos = [
    ("VLM_REV", "model", "Xkev/Llama-3.2V-11B-cot"),
    ("LLM_REV", "model", "meta-llama/Llama-3.1-8B-Instruct"),
    ("HEIMA_REV", "model", "shawnricecake/Heima"),
]
for key, kind, repo in repos:
    info = api.model_info(repo)
    print(f"{key}={info.sha}")
PY
mv /data/zxl/official_heima/model_revisions.env.tmp /data/zxl/official_heima/model_revisions.env
source /data/zxl/official_heima/model_revisions.env
export VLM_REV LLM_REV HEIMA_REV
cat /data/zxl/official_heima/model_revisions.env

python - <<'PY'
import json, os
from pathlib import Path
payload = {
    "Xkev/Llama-3.2V-11B-cot": os.environ["VLM_REV"],
    "meta-llama/Llama-3.1-8B-Instruct": os.environ["LLM_REV"],
    "shawnricecake/Heima": os.environ["HEIMA_REV"],
}
path = Path("/data/zxl/official_heima/model_revisions.json")
path.write_text(json.dumps(payload, indent=2) + "\n")
print(path)
PY

echo
echo "[1/3] Downloading official Heima checkpoint repository..."
hf download shawnricecake/Heima \
  --revision "$HEIMA_REV" \
  --local-dir "$CKPT_ROOT/shawnricecake-Heima"
du -sh "$CKPT_ROOT/shawnricecake-Heima"
df -h /data

echo
echo "[2/3] Downloading official decoder base model: meta-llama/Llama-3.1-8B-Instruct..."
hf download meta-llama/Llama-3.1-8B-Instruct \
  --revision "$LLM_REV" \
  --exclude "original/*" \
  --local-dir "$MODEL_ROOT/meta-llama-Llama-3.1-8B-Instruct"
du -sh "$MODEL_ROOT/meta-llama-Llama-3.1-8B-Instruct"
df -h /data

echo
echo "[3/3] Downloading official encoder base model: Xkev/Llama-3.2V-11B-cot..."
hf download Xkev/Llama-3.2V-11B-cot \
  --revision "$VLM_REV" \
  --exclude "original/*" \
  --local-dir "$MODEL_ROOT/Xkev-Llama-3.2V-11B-cot"
du -sh "$MODEL_ROOT/Xkev-Llama-3.2V-11B-cot"
df -h /data

python - <<'PY'
from __future__ import annotations
import hashlib, json, os
from pathlib import Path

roots = {
    "heima_checkpoint": Path("/data/zxl/official_heima/checkpoints/shawnricecake-Heima"),
    "llm_base": Path("/data/zxl/official_heima/models/meta-llama-Llama-3.1-8B-Instruct"),
    "vlm_base": Path("/data/zxl/official_heima/models/Xkev-Llama-3.2V-11B-cot"),
}
summary = {"completed_at_utc": __import__("datetime").datetime.datetime.utcnow().isoformat() + "Z", "roots": {}}
for name, root in roots.items():
    files = []
    total = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        size = path.stat().st_size
        total += size
        if path.suffix in {".json", ".model", ".txt", ".md"} or path.name.endswith((".safetensors", ".bin", ".pt", ".pth")):
            files.append({"path": str(path.relative_to(root)), "size": size})
    summary["roots"][name] = {"path": str(root), "total_bytes": total, "files": files}
Path("/data/zxl/Heima/reports/official_heima_reproduction/model_download_status.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2)[:4000])
PY

echo "Done: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
