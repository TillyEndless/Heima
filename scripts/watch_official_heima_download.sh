#!/usr/bin/env bash
set -uo pipefail

ROOT=/data/zxl
PROJECT=/data/zxl/Heima
DOWNLOAD_SCRIPT=/data/zxl/run_heima_download_with_proxy.sh
REPORT_DIR="$PROJECT/reports/official_heima_reproduction"
STATUS_JSON="$REPORT_DIR/download_watch_status.json"
WATCH_LOG_DIR=/data/zxl/official_heima/logs
PROXY_URL=http://127.0.0.1:17890

mkdir -p "$REPORT_DIR" "$WATCH_LOG_DIR"

write_status() {
  local state="$1"
  local detail="$2"
  python - "$STATUS_JSON" "$state" "$detail" <<'PY'
import json
import sys
from datetime import datetime, timezone

path, state, detail = sys.argv[1:4]
payload = {
    "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    "state": state,
    "detail": detail,
    "proxy_url": "http://127.0.0.1:17890",
    "download_script": "/data/zxl/run_heima_download_with_proxy.sh",
    "auto_training_enabled": False,
    "next_allowed_step_after_download": "resource verification and official checkpoint-loading smoke only",
}
with open(path, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2, ensure_ascii=False)
    f.write("\n")
PY
}

export PATH=/data/zxl/conda_envs/nlp-final/bin:$PATH
export HF_HOME=/data/zxl/hf
export HF_HUB_CACHE=/data/zxl/hf/hub
export HF_DATASETS_CACHE=/data/zxl/hf/datasets
export HF_XET_CACHE=/data/zxl/hf/xet
export XDG_CACHE_HOME=/data/zxl/.cache
export TMPDIR=/data/zxl/tmp
export http_proxy="$PROXY_URL"
export https_proxy="$PROXY_URL"
export HTTP_PROXY="$PROXY_URL"
export HTTPS_PROXY="$PROXY_URL"
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost
unset HF_HUB_OFFLINE
unset TRANSFORMERS_OFFLINE

attempt=0
write_status "waiting_for_proxy" "download watcher started"

while true; do
  attempt=$((attempt + 1))
  ts=$(date -u +%Y%m%d_%H%M%S)
  log="$WATCH_LOG_DIR/watch_download_${ts}.log"

  if ! curl -I --proxy "$PROXY_URL" --connect-timeout 10 --max-time 30 https://huggingface.co >/dev/null 2>&1; then
    write_status "waiting_for_proxy" "attempt ${attempt}: proxy check failed; waiting before retry"
    sleep 120
    continue
  fi

  if [[ ! -x "$DOWNLOAD_SCRIPT" && ! -f "$DOWNLOAD_SCRIPT" ]]; then
    write_status "failed" "missing download script: $DOWNLOAD_SCRIPT"
    exit 2
  fi

  write_status "running_download" "attempt ${attempt}: running $DOWNLOAD_SCRIPT; log=$log"
  bash "$DOWNLOAD_SCRIPT" >"$log" 2>&1
  rc=$?

  if [[ $rc -eq 0 ]]; then
    write_status "download_complete" "download script completed successfully; verify resources before any training"
    exit 0
  fi

  write_status "download_failed_retrying" "attempt ${attempt}: exit code ${rc}; log=$log; retrying after cooldown"
  sleep 300
done
