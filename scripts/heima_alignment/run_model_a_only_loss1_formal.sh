#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="$ROOT/configs/heima_aligned/model_a_only_loss1_formal.yaml"
SPLIT="$ROOT/data_split.json"
OUT="/data/zxl/runs/model_a_only_loss1_formal"
STAGE="train"
DRY_RUN=0
RESUME=0
EVAL_ONLY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --resume) RESUME=1; shift ;;
    --eval-only) EVAL_ONLY=1; shift ;;
    --stage) STAGE="${2:?missing stage}"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

python3 - <<PY
import json, shutil, subprocess, sys
from pathlib import Path
root=Path('$ROOT')
split=Path('$SPLIT')
config=Path('$CONFIG')
report_path=root/'reports'/'model_a_only_loss1_data_audit.json'
ckpt_report=root/'reports'/'model_a_only_loss1_checkpoint_audit.json'
if not split.exists():
    raise SystemExit('missing data_split.json; run data audit first')
if not report_path.exists():
    raise SystemExit('missing reports/model_a_only_loss1_data_audit.json')
if not ckpt_report.exists():
    raise SystemExit('missing reports/model_a_only_loss1_checkpoint_audit.json')
s=json.load(open(split))
r=json.load(open(report_path))
c=json.load(open(ckpt_report))
free=shutil.disk_usage('/data').free
summary={
  'config': str(config),
  'split_status': s.get('status'),
  'available_usable': s.get('available_usable'),
  'required_train_min': s.get('required_train_min'),
  'checkpoint': c.get('stage0_checkpoint'),
  'checkpoint_status': c.get('status'),
  'output_dir': '$OUT',
  'loss2_enabled': False,
  'cumulative_latent': False,
  'heima_b_interpreter_training': False,
  'tiny_acceptance': False,
  'disk_free_gb': round(free/1024**3,2),
  'dry_run': bool($DRY_RUN),
  'resume': bool($RESUME),
  'eval_only': bool($EVAL_ONLY),
  'stage': '$STAGE',
}
print(json.dumps(summary, indent=2, ensure_ascii=False))
if free < 100 * 1024**3:
    raise SystemExit('STOP: /data free space below 100GB')
if s.get('status') != 'ready' or int(s.get('available_usable') or 0) < 10000:
    raise SystemExit('STOP: insufficient image-backed complete examples for formal pilot; training not started')
if bool($DRY_RUN):
    raise SystemExit(0)
PY

if [[ "$DRY_RUN" == "1" ]]; then
  exit 0
fi

# Reaching here means preflight passed. This command intentionally uses only
# Model-A-only Loss1; Loss2 and Model-B training are not involved.
exec python3 "$ROOT/scripts/heima_stage2_model_a_only_self_decode.py" \
  --model-a-path /data/zxl/small_models/Qwen2.5-VL-3B-Instruct \
  --dataset-path "$ROOT/formal_split" \
  --image-root /data/zxl/official_heima/datasets/LLaVA-CoT-100k/image_files \
  --output-dir "$OUT" \
  --sections summary,caption,reasoning \
  --lambda-self 0.1 \
  --stage0-checkpoint /data/zxl/runs/heima_stage2_interp_supervision_small_v1/seed42/20260722_135515/checkpoints/s0_encoder.pt \
  --max-train-samples 10000 \
  --max-eval-samples 512 \
  --max-steps 5000 \
  --eval-every 500 \
  --save-every 1000 \
  --mode a_only_self_decode
