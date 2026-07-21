#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
PYTHON_BIN=${PYTHON:-python3}
SUBSET_ROOT=${SUBSET_ROOT:-/data/zxl/official_heima/micro_subsets/chartqa_sqa_v1}
RUN_ROOT=${RUN_ROOT:-/data/zxl/runs/heima_ab_loss1_tiny_acceptance_v1}
CONFIG=${CONFIG:-${ROOT}/configs/heima_aligned/ab_loss1_tiny_qwen_vl3b.yaml}
RESUME=0
STAGE=""
EVAL_ONLY=0
DRY_RUN=0
ALLOW_SMALLER_AVAILABLE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --resume) RESUME=1; shift ;;
    --stage) STAGE="$2"; shift 2 ;;
    --eval-only) EVAL_ONLY=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --allow-smaller-available) ALLOW_SMALLER_AVAILABLE=1; shift ;;
    --subset-root) SUBSET_ROOT="$2"; shift 2 ;;
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --config) CONFIG="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done
if [[ "$SUBSET_ROOT" == *"datasets/LLaVA-CoT-100k"* ]]; then
  echo "Refusing to use full LLaVA-CoT-100k train.jsonl for tiny acceptance: $SUBSET_ROOT" >&2
  exit 1
fi
mkdir -p "$RUN_ROOT" reports docs/heima_alignment
SPLIT="$RUN_ROOT/data_split.json"
SPLIT_ARGS=(
  --subset-root "$SUBSET_ROOT"
  --out "$SPLIT"
  --seed 42
  --train-size 192
  --eval-size 48
)
if [[ "$ALLOW_SMALLER_AVAILABLE" == 1 ]]; then
  SPLIT_ARGS+=(--allow-smaller-available)
fi
"$PYTHON_BIN" "$SCRIPT_DIR/create_tiny_acceptance_split.py" "${SPLIT_ARGS[@]}"
ALLOW_SMALLER_AVAILABLE="$ALLOW_SMALLER_AVAILABLE" "$PYTHON_BIN" - <<PY
import json, os, pathlib
subset = pathlib.Path('$SUBSET_ROOT')
split = json.loads(pathlib.Path('$SPLIT').read_text())
allow_smaller = os.environ.get('ALLOW_SMALLER_AVAILABLE') == '1'
if split.get('exact_requested_size') is not True and not allow_smaller:
    raise SystemExit(f"exact 192/48 split not available; got train={split['train_size']} eval={split['eval_size']}")
if not allow_smaller:
    assert split['train_size'] == 192, split['train_size']
    assert split['eval_size'] == 48, split['eval_size']

for path_key in ('source_subset', 'train_file', 'eval_file'):
    value = str(split.get(path_key, ''))
    assert '/datasets/LLaVA-CoT-100k/' not in value, f'unexpected full dataset path in {path_key}: {value}'
for key in ('train','eval'):
    for item in split[key]:
        p=pathlib.Path(item['resolved_image_path'])
        assert p.is_file(), p
        assert subset in p.parents or p == subset, (subset, p)
print({'split': '$SPLIT', 'train': split['train_size'], 'eval': split['eval_size'], 'exact_requested_size': split.get('exact_requested_size'), 'subset': str(subset)})
PY
PIPELINE_ARGS=(--config "$CONFIG" --output-root "$RUN_ROOT")
if [[ "$RESUME" == 1 ]]; then PIPELINE_ARGS+=(--resume); fi
if [[ -n "$STAGE" ]]; then PIPELINE_ARGS+=(--from-stage "$STAGE" --to-stage "$STAGE"); fi
if [[ "$EVAL_ONLY" == 1 ]]; then PIPELINE_ARGS+=(--skip-train); fi
if [[ "$DRY_RUN" == 1 ]]; then PIPELINE_ARGS+=(--dry-run); fi

# Baseline: frozen/detached Loss1 branch.
BASE_ARGS=("${PIPELINE_ARGS[@]}" --mode tiny_reasoning_baseline_detach --run-id baseline_loss1_detach)
# Ours: Loss1 gradient can flow into A.
OURS_ARGS=("${PIPELINE_ARGS[@]}" --mode tiny_reasoning_ours_no_detach --run-id ours_loss1_no_detach)

"$PYTHON_BIN" "$ROOT/scripts/heima_aligned/pipeline.py" "${BASE_ARGS[@]}"
"$PYTHON_BIN" "$ROOT/scripts/heima_aligned/pipeline.py" "${OURS_ARGS[@]}"

cat > "$ROOT/reports/heima_ab_loss1_tiny_acceptance_report.md" <<EOF
# Heima A+B Loss1 Tiny Real-Image Acceptance

Status: dry-run/protocol-ready unless this script is launched without --dry-run under a trainer implementation.

Data:
- subset: $SUBSET_ROOT
- split: $SPLIT
- requested train/eval: 192/48
- actual train/eval: see data_split.json; exact_requested_size indicates whether the strict request was met
- full LLaVA-CoT-100k train.jsonl: not used

Schedule:
- Stage 0: explicit CoT SFT
- Stage 1: reasoning-only latent replacement
- Stage 2: freeze A, train B_reasoning
- Stage 3 baseline: Loss1 with detached A latent
- Stage 3 ours: Loss1 with non-detached A latent

Required evaluation outputs for a real run:
- answer accuracy
- reasoning reconstruction NLL
- deterministic generation examples
- correct latent vs shuffle latent
- zero latent
- question-only baseline

Interpretation:
- This is mechanism validation, not benchmark reproduction.
- Passing this run validates real image loading and Loss1 wiring on existing data.
- It does not reproduce full Heima paper metrics or full LLaVA-CoT-100k training.
EOF
echo "tiny acceptance prepared: $RUN_ROOT"
