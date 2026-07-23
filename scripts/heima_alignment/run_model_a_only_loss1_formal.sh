#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="$ROOT/configs/heima_aligned/model_a_only_loss1_formal.yaml"
OUT="/data/zxl/runs/model_a_only_loss1_formal"
SPLIT="$OUT/data_split.json"
DATASET="$OUT/formal_split"
IMAGE_ROOT="$OUT/image_files"
SESSION="model_a_only_loss1_formal"
STAGE="all"
PREPARE_DATA=0
TRAIN=0
EVAL_ONLY=0
RESUME=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prepare-data) PREPARE_DATA=1; shift ;;
    --train) TRAIN=1; shift ;;
    --eval-only) EVAL_ONLY=1; shift ;;
    --resume) RESUME=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --stage) STAGE="${2:?missing stage}"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ "$STAGE" == "prepare-data" ]]; then PREPARE_DATA=1; fi
if [[ "$STAGE" == "train" ]]; then TRAIN=1; fi
if [[ "$STAGE" == "all" && "$PREPARE_DATA" == "0" && "$TRAIN" == "0" && "$EVAL_ONLY" == "0" && "$DRY_RUN" == "0" ]]; then
  PREPARE_DATA=1
  TRAIN=1
fi

mkdir -p "$OUT" "$ROOT/reports"

if [[ "$PREPARE_DATA" == "1" ]]; then
  python3 "$ROOT/scripts/heima_alignment/build_formal_image_subset.py" \
    --input /data/zxl/official_heima/datasets/LLaVA-CoT-100k/train.jsonl \
    --out-dir "$OUT" \
    --train-size 5000 \
    --eval-size 512 \
    --seed 42
fi

python3 - <<PY
import json, shutil
from pathlib import Path
root=Path('$ROOT')
out=Path('$OUT')
split=Path('$SPLIT')
config=Path('$CONFIG')
ckpt=Path('/data/zxl/runs/heima_stage2_interp_supervision_small_v1/seed42/20260722_135515/checkpoints/s0_encoder.pt')
free=shutil.disk_usage('/data').free
if not config.exists():
    raise SystemExit('STOP: missing config')
if not ckpt.exists():
    raise SystemExit(f'STOP: missing stage0 checkpoint: {ckpt}')
if not split.exists():
    raise SystemExit('STOP: missing run data_split.json; run --prepare-data first')
s=json.load(open(split))
summary={
  'config': str(config),
  'split_status': s.get('status'),
  'selected_train_count': s.get('selected_train_count'),
  'selected_eval_count': s.get('selected_eval_count'),
  'available_usable': s.get('available_usable'),
  'split_hash': s.get('split_hash'),
  'dataset_path': s.get('dataset_path'),
  'image_root': s.get('image_root'),
  'stage0_checkpoint': str(ckpt),
  'output_dir': str(out),
  'has_model_b': False,
  'loss_main': True,
  'loss1_enabled': True,
  'lambda_loss1': 0.1,
  'loss2_enabled': False,
  'cumulative_latent': False,
  'heima_b_interpreter_training': False,
  'full_llava_cot_100k': False,
  'heima_benchmark_evaluation': False,
  'tiny_acceptance': False,
  'self_decode_with_image': False,
  'detach_latent': False,
  'disk_free_gb': round(free/1024**3,2),
  'dry_run': bool($DRY_RUN),
  'resume': bool($RESUME),
  'eval_only': bool($EVAL_ONLY),
  'train_requested': bool($TRAIN),
}
print(json.dumps(summary, indent=2, ensure_ascii=False))
(root/'reports'/'model_a_only_loss1_formal_preflight.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True)+'\n', encoding='utf-8')
if free < 80 * 1024**3:
    raise SystemExit('STOP: /data free space below 80GB')
if s.get('status') != 'ready' or int(s.get('selected_train_count') or 0) < 5000 or int(s.get('selected_eval_count') or 0) < 512:
    raise SystemExit('STOP: insufficient image-backed complete examples for formal pilot; training not started')
PY

if [[ "$DRY_RUN" == "1" || "$TRAIN" == "0" ]]; then
  exit 0
fi

CMD=(python3 "$ROOT/scripts/heima_stage2_model_a_only_self_decode.py"
  --model-a-path /data/zxl/small_models/Qwen2.5-VL-3B-Instruct
  --dataset-path "$DATASET"
  --image-root "$IMAGE_ROOT"
  --output-dir "$OUT"
  --sections summary,caption,reasoning
  --lambda-self 0.1
  --stage0-checkpoint /data/zxl/runs/heima_stage2_interp_supervision_small_v1/seed42/20260722_135515/checkpoints/s0_encoder.pt
  --max-train-samples 5000
  --max-eval-samples 512
  --max-steps 5000
  --eval-every 1000
  --save-every 1000
  --mode a_only_self_decode)

printf '%q ' "${CMD[@]}" > "$OUT/train_command.sh"
printf '\n' >> "$OUT/train_command.sh"
chmod +x "$OUT/train_command.sh"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "STOP: tmux session $SESSION already exists" >&2
  exit 1
fi

tmux new-session -d -s "$SESSION" "cd '$ROOT' && exec ${CMD[*]} >> '$OUT/train.log' 2>&1"
sleep 2
pgrep -af "heima_stage2_model_a_only_self_decode.py.*model_a_only_loss1_formal" > "$OUT/train_pid.txt" || true
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader > "$OUT/gpu_at_start.txt" || true
tmux list-sessions | grep "$SESSION" > "$OUT/tmux_session.txt" || true
cat "$OUT/tmux_session.txt"
