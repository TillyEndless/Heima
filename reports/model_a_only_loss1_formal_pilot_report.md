# Model-A-Only Loss1 Formal Pilot Report

Date: 2026-07-24
Worktree: `/data/zxl/Heima-model-a-only-loss1-formal`
Branch: `feat/model-a-only-self-decode`
Base commit before this prep: `2c3947f5229fe48769e6b3e30068fca47d68ebb9`

## Status

Formal pilot training was **not started**.

Reason: data preflight failed. The full LLaVA-CoT-100k text file has enough records, but only 257 complete examples currently have accessible image files on disk. The required threshold was `train >= 10000` image-backed examples.

No tmux session was started:

`model_a_only_loss1_formal` does not exist.

## Checkpoint Source

Checkpoint audit file:

`reports/model_a_only_loss1_checkpoint_audit.json`

Selected Stage0 checkpoint:

`/data/zxl/runs/heima_stage2_interp_supervision_small_v1/seed42/20260722_135515/checkpoints/s0_encoder.pt`

Properties:

- Model: `/data/zxl/small_models/Qwen2.5-VL-3B-Instruct`
- Source: strict Heima Stage0 explicit section/answer warmup
- Source dataset: `/data/zxl/official_heima/micro_subsets/chartqa_sqa_available_images_v1`
- Source train samples: 167
- Limitation: no larger Qwen2.5-VL-3B explicit-CoT Stage0 checkpoint was found during this audit

## Dataset Statistics

Data audit file:

`reports/model_a_only_loss1_data_audit.json`

Candidate full dataset:

`/data/zxl/official_heima/datasets/LLaVA-CoT-100k/train.jsonl`

Candidate image roots checked:

- `/data/zxl/official_heima/datasets/LLaVA-CoT-100k/image_files`
- `/data/zxl/official_heima/datasets/LLaVA-CoT-100k`
- `/data/zxl/official_heima/micro_subsets/chartqa_sqa_v1/image_files`
- `/data/zxl/official_heima/micro_subsets/chartqa_sqa_available_images_v1/image_files`

Audit counts:

| metric | count |
|---|---:|
| total records | 98,582 |
| records with question | 98,582 |
| complete summary/caption/reasoning/answer | 98,568 |
| image accessible | 257 |
| usable complete + image-backed | 257 |

Rates:

- section completeness: 99.986%
- image access rate: 0.261%
- usable rate: 0.261%

Decision:

`do_not_start_training`: usable image-backed complete examples are below the 10,000 minimum.

A `data_split.json` was generated, but its status is `insufficient_accessible_images`; it contains no formal train/eval split and only records an available preview.

## Training Configuration

Config added:

`configs/heima_aligned/model_a_only_loss1_formal.yaml`

Intended formal pilot:

- Mode: `model_a_only_self_decode`
- Model: Qwen2.5-VL-3B-Instruct
- Stage0: use existing CoT checkpoint; do not retrain Stage0
- Loss2: disabled
- cumulative latent: disabled
- Heima B interpreter training: disabled
- tiny acceptance: disabled
- target train samples: 10,000
- target eval samples: 512
- max steps: 5,000
- eval every: 500
- save model weights at 1000, 2500, 5000, final
- do not save optimizer checkpoint

Loss:

`L_total = L_main + 0.1 * mean(Loss1_summary, Loss1_caption, Loss1_reasoning)`

Self-decode:

- `with_image: false`
- `detach_latent: false`
- sections: `summary, caption, reasoning`
- latent source: `last_hidden_state_of_latent_cot_i`

## One-Key Entrypoint

Script added:

`scripts/heima_alignment/run_model_a_only_loss1_formal.sh`

Supports:

- `--dry-run`
- `--resume`
- `--eval-only`
- `--stage <name>`

Dry-run command run:

`bash scripts/heima_alignment/run_model_a_only_loss1_formal.sh --dry-run`

Dry-run result:

- checkpoint found: yes
- config found: yes
- output dir: `/data/zxl/runs/model_a_only_loss1_formal`
- Loss2 enabled: false
- cumulative latent: false
- Heima B interpreter training: false
- tiny acceptance: false
- disk free: 158.97GB
- split status: `insufficient_accessible_images`
- available usable samples: 257
- required train minimum: 10,000
- exit: stopped before training

## Resource Check

`df -h /data`:

- `/data` size: 3.5T
- used: 3.4T
- available: 154G
- use: 96%

Requirement was at least 100GB free, so disk passed.

`nvidia-smi`:

- 8 x NVIDIA GeForce RTX 4090 visible
- all reported 0 MiB used and 0% utilization at audit time

GPU availability passed.

## Loss Curve

N/A. Training was not started because the data requirement failed.

## Gradient Audit

N/A for formal pilot. Training/eval was not started. The existing Model-A-only branch unit tests from the prior commit cover the intended gradient contract on toy data, but no formal data-backed gradient audit was run in this task.

Expected formal condition once data is available:

- `grad_z_summary > 0`
- `grad_z_caption > 0`
- `grad_z_reasoning > 0`
- `grad_A_from_loss1 > 0`

## Main Accuracy

N/A. No formal eval was run because training was not started.

## Latent Intervention

N/A. No formal eval was run because training was not started.

Expected eval once data is available:

- correct latent
- shuffle latent
- zero latent
- Q-only / no latent
- report correct-shuffle margin per section

## Does This Prove Loss1 Shapes Latent Reasoning?

No. This round only completed formal pilot preflight and found a blocking data availability issue. Since no formal training or eval ran, it provides no evidence for or against the hypothesis.

## Loss2 / Excluded Runs

Loss2 was not run. Cumulative latent experiments were not run. Heima B interpreter training was not run. Full LLaVA-CoT-100k training was not run. Tiny acceptance was not run.

## Next Required Fix

Make at least 10,000 complete LLaVA-CoT image files accessible under a stable image root, or provide an existing official real-image subset with at least 10,000 complete examples. After that, rerun:

`bash scripts/heima_alignment/run_model_a_only_loss1_formal.sh --dry-run`

Only if preflight reports `split_status: ready` should the formal tmux training be started.
