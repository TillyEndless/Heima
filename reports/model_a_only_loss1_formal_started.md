# Model-A-only Loss1 Formal Pilot Started

Generated: 2026-07-23 17:23:39 UTC

## 1. Data Recovery Result

- Source jsonl: `/data/zxl/official_heima/datasets/LLaVA-CoT-100k/train.jsonl`
- Run split: `/data/zxl/runs/model_a_only_loss1_formal/data_split.json`
- Formal dataset path: `/data/zxl/runs/model_a_only_loss1_formal/formal_split`
- Image root: `/data/zxl/runs/model_a_only_loss1_formal/image_files`
- Train samples: `5000`
- Eval samples: `512`
- Available usable image-backed samples: `5512`
- Restored from local partial zip: `20347`
- Full image.zip download: not performed

## 2. Split Hash

`f1a0c51bc4960c58add4c5421525ad764d1cee09ccfda7ff79fb153d20d08325`

## 3. Training Config

- Model A: `Qwen2.5-VL-3B-Instruct`
- Model B: absent, `has_model_b=false`
- Objective: `L_total = L_main + 0.1 * mean(Loss1_summary, Loss1_caption, Loss1_reasoning)`
- Loss2: disabled
- Cumulative latent: disabled
- Self decode with image: false
- Detach latent: false
- Max optimizer steps: `5000`
- Eval every: `1000`
- Save steps: `1000, 2500, 5000, final`
- Optimizer checkpoint saving: disabled

## 4. Checkpoint

- Stage0 checkpoint: `/data/zxl/runs/heima_stage2_interp_supervision_small_v1/seed42/20260722_135515/checkpoints/s0_encoder.pt`
- Key transform: `qwen25_vl_model_bridge`
- Loaded tensors: `825/825`
- Missing tensors after load: `0`
- Run directory: `/data/zxl/runs/model_a_only_loss1_formal/seed42/20260723_172211`

## 5. Tmux

- Session: `model_a_only_loss1_formal`
- Tmux status: `model_a_only_loss1_formal: 1 windows (created Thu Jul 23 17:22:08 2026)`
- PID line: `2110691 /root/miniconda3/envs/st/bin/python /data/zxl/Heima-model-a-only-loss1-formal/scripts/heima_stage2_model_a_only_self_decode.py --model-a-path /data/zxl/small_models/Qwen2.5-VL-3B-Instruct --dataset-path /data/zxl/runs/model_a_only_loss1_formal/formal_split --image-root /data/zxl/runs/model_a_only_loss1_formal/image_files --output-dir /data/zxl/runs/model_a_only_loss1_formal --sections summary,caption,reasoning --lambda-self 0.1 --stage0-checkpoint /data/zxl/runs/heima_stage2_interp_supervision_small_v1/seed42/20260722_135515/checkpoints/s0_encoder.pt --max-train-samples 5000 --max-eval-samples 512 --max-steps 5000 --eval-every 1000 --save-every 0 --save-steps 1000,2500,5000 --eval-probe-samples 32 --mode a_only_self_decode`
- Command: `/root/miniconda3/envs/st/bin/python /data/zxl/Heima-model-a-only-loss1-formal/scripts/heima_stage2_model_a_only_self_decode.py --model-a-path /data/zxl/small_models/Qwen2.5-VL-3B-Instruct --dataset-path /data/zxl/runs/model_a_only_loss1_formal/formal_split --image-root /data/zxl/runs/model_a_only_loss1_formal/image_files --output-dir /data/zxl/runs/model_a_only_loss1_formal --sections summary\,caption\,reasoning --lambda-self 0.1 --stage0-checkpoint /data/zxl/runs/heima_stage2_interp_supervision_small_v1/seed42/20260722_135515/checkpoints/s0_encoder.pt --max-train-samples 5000 --max-eval-samples 512 --max-steps 5000 --eval-every 1000 --save-every 0 --save-steps 1000\,2500\,5000 --eval-probe-samples 32 --mode a_only_self_decode`
- Log: `/data/zxl/runs/model_a_only_loss1_formal/train.log`

## 6. GPU

```text
0, NVIDIA GeForce RTX 4090, 31647 MiB, 49140 MiB, 100 %
1, NVIDIA GeForce RTX 4090, 18139 MiB, 49140 MiB, 87 %
2, NVIDIA GeForce RTX 4090, 3 MiB, 49140 MiB, 0 %
3, NVIDIA GeForce RTX 4090, 3 MiB, 49140 MiB, 0 %
4, NVIDIA GeForce RTX 4090, 18139 MiB, 46068 MiB, 96 %
5, NVIDIA GeForce RTX 4090, 18183 MiB, 49140 MiB, 83 %
6, NVIDIA GeForce RTX 4090, 3 MiB, 49140 MiB, 0 %
7, NVIDIA GeForce RTX 4090, 3 MiB, 49140 MiB, 0 %
```

## 7. Early Signal

Latest step log:

```json
{"actual_forward_count_per_batch": 4, "expected_forward_count_per_batch": 4, "extra_trainable_params_except_A": 0, "finite": true, "grad_A_from_self_decode_norm": 30.741316150448267, "grad_A_total_norm": 2868.5421605894926, "grad_z_norm": {"caption": 0.0018055004371289058, "reasoning": 0.00018787451163948974, "summary": 0.0012672496301100906}, "has_model_b": false, "main_loss": 0.6825500726699829, "mode": "a_only_self_decode", "optimizer_contains_model_b": false, "per_section_loss": {"caption": 1.7109375, "reasoning": 1.5234375, "summary": 1.6640625}, "self_loss": 1.6328125, "step": 40, "total_loss": 0.8456360101699829, "use_projector": false, "use_role_embedding": false}
```

The step 1 gradient audit shows nonzero `grad_z_summary`, `grad_z_caption`, `grad_z_reasoning`, and `grad_A_from_self_decode_norm`, so Loss1 is connected back to Model A in this launch.

## 8. Expected Runtime

Rough overnight pilot expectation: several hours for 5000 optimizer steps on the current single-process Qwen2.5-VL-3B setup, with eval probes every 1000 steps and checkpoints at 1000/2500/5000/final.

## 9. Explicit Non-Starts

- Loss2 was not started.
- Cumulative latent was not started.
- B interpreter training was not started.
- Full LLaVA-CoT-100k training was not started.
- Heima benchmark evaluation was not started.
- Tiny acceptance was not started.

## 10. Old Experiment Safety

No old run directories were deleted or overwritten. Earlier invalid launch attempts inside this same new formal run were stopped after environment/checkpoint/config-gate issues and left on disk for traceability.

## 11. Git

- Branch: `feat/model-a-only-self-decode`
- Commit at launch: `4a64eae9fd58362be7250fab9ba03aa3aeb49987`
