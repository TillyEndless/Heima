#!/usr/bin/env bash
set -euo pipefail

export PATH=/data/zxl/conda_envs/nlp-final/bin:$PATH
export HF_HOME=/data/zxl/hf
export HF_HUB_CACHE=/data/zxl/hf/hub
export HF_DATASETS_CACHE=/data/zxl/hf/datasets
export HF_XET_CACHE=/data/zxl/hf/xet
export XDG_CACHE_HOME=/data/zxl/.cache
export TMPDIR=/data/zxl/tmp
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1

PROJECT=/data/zxl/Heima
EXP_ROOT=/data/zxl/runs/overnight_mllm_loss1_suite_$(date +%Y%m%d_%H%M%S)
mkdir -p "$EXP_ROOT"/{configs,logs,runs,reports,tmp}

cd "$PROJECT"

python - <<PY
import json, pathlib, subprocess, time
root = pathlib.Path("$EXP_ROOT")
def sh(cmd):
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"ERROR: {exc}"
manifest = {
    "suite": "overnight_mllm_loss1_suite",
    "started_at_local": time.strftime("%Y-%m-%d %H:%M:%S"),
    "project": "$PROJECT",
    "experiment_root": str(root),
    "goal": "Single-seed task coverage for small-MLLM Heima-style Main-only, joint-detach, ours-no-detach, latent-NTP ablation, and staged latent semantic gate.",
    "resource_note": "Official 11B/8B download is not resumed by this launcher. cad218 is not used for duplicate main experiments.",
    "host": sh(["hostname"]),
    "disk_data": sh(["df", "-h", "/data"]),
    "gpu_snapshot": sh(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free,utilization.gpu", "--format=csv,noheader"]),
    "models": {
        "model_a": "/data/zxl/small_models/Qwen2.5-VL-3B-Instruct",
        "model_b": "/data/zxl/small_models/Qwen2.5-0.5B-Instruct"
    },
    "data": {
        "subset": "/data/zxl/official_heima/micro_subsets/chartqa_sqa_available_images_v1",
        "image_root": "/data/zxl/official_heima/micro_subsets/chartqa_sqa_v1/image_files"
    },
    "strict_core": {
        "thinking_state_mode": "predictor",
        "projector": "HeimaOfficialAbstractProjection",
        "replacement": "official_embedding_replacement",
        "loss": "heima_ce_loss / CEWithChunkedOutputLoss when available",
        "detach_only_function": "prepare_latent_for_decoder"
    },
    "jobs": {
        "gpu0_text_labels_m0_m1_m2_plus_fresh_b": {
            "purpose": "Primary comparison: M0 Main-only, M1 joint-detach, M2 ours-no-detach; then fresh B_eval for frozen M0/M1/M2 A.",
            "s0_steps": 200,
            "s1_steps": 800,
            "joint_steps": 300,
            "fresh_b_eval_steps": 300
        },
        "gpu1_latent_marker_ntp_ablation": {
            "purpose": "Teacher-requested label-design ablation: add NTP supervision on typed latent marker slot during Loss1.",
            "s0_steps": 200,
            "s1_steps": 500,
            "joint_steps": 200,
            "train_latent_marker_ntp": True
        },
        "gpu2_progressive_staged_s1_long": {
            "purpose": "Long frozen-A Heima staged interpreter gate from existing progressive recover checkpoint.",
            "s1_steps": 1000
        }
    }
}
(root / "configs" / "suite_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
PY

cat > "$EXP_ROOT/configs/job_commands.txt" <<EOF
[GPU0 primary]
CUDA_VISIBLE_DEVICES=0 python scripts/run_data_small_vlm_official_sections.py --out "$EXP_ROOT/runs/text_labels_m0_m1_m2" --seed 42 --s0-steps 200 --s1-steps 800 --joint-steps 300 --batch-size 1 --eval-samples 16 --optimizer adafactor --max-image-side 336 --log-every 50

[GPU1 latent marker NTP]
CUDA_VISIBLE_DEVICES=1 python scripts/run_data_small_vlm_official_sections.py --out "$EXP_ROOT/runs/latent_marker_ntp_m0_m1_m2" --seed 42 --s0-steps 200 --s1-steps 500 --joint-steps 200 --batch-size 1 --eval-samples 16 --optimizer adafactor --max-image-side 336 --log-every 50 --train-latent-marker-ntp

[GPU2 long staged S1]
CUDA_VISIBLE_DEVICES=2 python scripts/run_data_small_vlm_progressive_interpreters.py --out "$EXP_ROOT/runs/progressive_staged_s1_long" --seed 42 --s1-steps 1000 --batch-size 1 --eval-samples 16 --optimizer adafactor --max-image-side 336 --log-every 50
EOF

cat > "$EXP_ROOT/tmp/run_gpu0_primary.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
export PATH=/data/zxl/conda_envs/nlp-final/bin:$PATH
export HF_HOME=/data/zxl/hf
export HF_HUB_CACHE=/data/zxl/hf/hub
export HF_DATASETS_CACHE=/data/zxl/hf/datasets
export HF_XET_CACHE=/data/zxl/hf/xet
export XDG_CACHE_HOME=/data/zxl/.cache
export TMPDIR=/data/zxl/tmp
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
cd /data/zxl/Heima
EXP_ROOT="__EXP_ROOT__"
CUDA_VISIBLE_DEVICES=0 python scripts/run_data_small_vlm_official_sections.py \
  --out "$EXP_ROOT/runs/text_labels_m0_m1_m2" \
  --seed 42 \
  --s0-steps 200 \
  --s1-steps 800 \
  --joint-steps 300 \
  --batch-size 1 \
  --eval-samples 16 \
  --optimizer adafactor \
  --max-image-side 336 \
  --log-every 50
RUN_DIR=$(find "$EXP_ROOT/runs/text_labels_m0_m1_m2/seed42" -mindepth 1 -maxdepth 1 -type d | sort | tail -1)
echo "PRIMARY_RUN_DIR=$RUN_DIR" | tee "$EXP_ROOT/reports/primary_run_dir.env"
for group in s0_encoder joint_detach ours_l1_no_detach; do
  case "$group" in
    s0_encoder) ckpt="$RUN_DIR/checkpoints/s0_encoder.pt"; source_group="M0_main_only" ;;
    joint_detach) ckpt="$RUN_DIR/checkpoints/joint_detach.pt"; source_group="M1_joint_detach" ;;
    ours_l1_no_detach) ckpt="$RUN_DIR/checkpoints/ours_l1_no_detach.pt"; source_group="M2_ours_no_detach" ;;
  esac
  CUDA_VISIBLE_DEVICES=0 python scripts/run_data_small_vlm_fresh_b_eval.py \
    --encoder-checkpoint "$ckpt" \
    --source-group "$source_group" \
    --out "$EXP_ROOT/runs/fresh_b_eval_text_labels" \
    --seed 42 \
    --s1-steps 300 \
    --batch-size 1 \
    --eval-samples 16 \
    --optimizer adafactor \
    --max-image-side 336 \
    --log-every 50
done
EOF

cat > "$EXP_ROOT/tmp/run_gpu1_latent_ntp.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
export PATH=/data/zxl/conda_envs/nlp-final/bin:$PATH
export HF_HOME=/data/zxl/hf
export HF_HUB_CACHE=/data/zxl/hf/hub
export HF_DATASETS_CACHE=/data/zxl/hf/datasets
export HF_XET_CACHE=/data/zxl/hf/xet
export XDG_CACHE_HOME=/data/zxl/.cache
export TMPDIR=/data/zxl/tmp
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
cd /data/zxl/Heima
EXP_ROOT="__EXP_ROOT__"
CUDA_VISIBLE_DEVICES=1 python scripts/run_data_small_vlm_official_sections.py \
  --out "$EXP_ROOT/runs/latent_marker_ntp_m0_m1_m2" \
  --seed 42 \
  --s0-steps 200 \
  --s1-steps 500 \
  --joint-steps 200 \
  --batch-size 1 \
  --eval-samples 16 \
  --optimizer adafactor \
  --max-image-side 336 \
  --log-every 50 \
  --train-latent-marker-ntp
EOF

cat > "$EXP_ROOT/tmp/run_gpu2_staged_long.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
export PATH=/data/zxl/conda_envs/nlp-final/bin:$PATH
export HF_HOME=/data/zxl/hf
export HF_HUB_CACHE=/data/zxl/hf/hub
export HF_DATASETS_CACHE=/data/zxl/hf/datasets
export HF_XET_CACHE=/data/zxl/hf/xet
export XDG_CACHE_HOME=/data/zxl/.cache
export TMPDIR=/data/zxl/tmp
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
cd /data/zxl/Heima
EXP_ROOT="__EXP_ROOT__"
CUDA_VISIBLE_DEVICES=2 python scripts/run_data_small_vlm_progressive_interpreters.py \
  --out "$EXP_ROOT/runs/progressive_staged_s1_long" \
  --seed 42 \
  --s1-steps 1000 \
  --batch-size 1 \
  --eval-samples 16 \
  --optimizer adafactor \
  --max-image-side 336 \
  --log-every 50
EOF

for f in "$EXP_ROOT"/tmp/run_gpu*.sh; do
  sed -i "s#__EXP_ROOT__#$EXP_ROOT#g" "$f"
  chmod +x "$f"
done

tmux new-session -d -s overnight_mllm_gpu0 "bash '$EXP_ROOT/tmp/run_gpu0_primary.sh' 2>&1 | tee '$EXP_ROOT/logs/gpu0_primary.log'"
tmux new-session -d -s overnight_mllm_gpu1 "bash '$EXP_ROOT/tmp/run_gpu1_latent_ntp.sh' 2>&1 | tee '$EXP_ROOT/logs/gpu1_latent_ntp.log'"
tmux new-session -d -s overnight_mllm_gpu2 "bash '$EXP_ROOT/tmp/run_gpu2_staged_long.sh' 2>&1 | tee '$EXP_ROOT/logs/gpu2_staged_long.log'"

echo "$EXP_ROOT" | tee /data/zxl/runs/latest_overnight_mllm_loss1_suite.txt
echo "Started tmux sessions: overnight_mllm_gpu0 overnight_mllm_gpu1 overnight_mllm_gpu2"
tmux ls | grep overnight_mllm || true
