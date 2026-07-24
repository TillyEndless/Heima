# A+B Loss1 Shortcut Formal Started

Generated: 2026-07-24 04:19:02

## Repository

- Worktree: `/data/zxl/Heima-ab-loss1-shortcut-formal`
- Branch: `feat/ab-loss1-shortcut-formal`
- Commit: `6345325`
- Git status at report time: `M reports/ab_loss1_shortcut_checkpoint_audit.json`
- Base branch: `feat/heima-stage2-interp-supervision`

## Data Recovery

- Source split: `/data/zxl/runs/model_a_only_loss1_formal/data_split.json`
- Dataset path: `/data/zxl/runs/model_a_only_loss1_formal/formal_split`
- Image root: `/data/zxl/runs/model_a_only_loss1_formal/image_files`
- Train/eval: `5000` / `512`
- Image access rate: `1.0`
- Missing images: `0`
- Split hash: `3ce0459e558f00e69dbcf1ee42a648275521fca76cd1e4dc0471487d5b4a1b60:6f5bd78c121a1dbcabbc983ddf9d724b5e8754a18a1f05685c261fd4a6911a1c`
- Source split hash: `f1a0c51bc4960c58add4c5421525ad764d1cee09ccfda7ff79fb153d20d08325`

## Configuration

- Model A: `/data/zxl/small_models/Qwen2.5-VL-3B-Instruct`
- Model B: `/data/zxl/small_models/Qwen2.5-0.5B-Instruct`
- Stage0 A checkpoint: `/data/zxl/runs/heima_stage2_interp_supervision_small_v1/seed42/20260722_135515/checkpoints/s0_encoder.pt`
- Sections: `summary`, `caption`, `reasoning`
- Latent mode: `local`; cumulative latent disabled
- Loss2: disabled
- Tiny acceptance: not run
- Full LLaVA-CoT-100k: not run
- Heima benchmark evaluation: not run

## Dry Run Gate

- H0: `pass`; `grad_A_from_loss1=0.0`, `grad_B_from_loss1=35.75615473054087`
- H1: `pass`; `grad_A_from_loss1=15.734638861234066`, `grad_B_from_loss1=35.75615473054087`
- H2: `deferred_until_h0_final_b_exists`; waits for `/data/zxl/runs/ab_loss1_shortcut_formal/h0_heima_b_probe/checkpoints/b_final.pt`

## Started Training

- Started group: `h0_heima_b_probe`
- Loss: `Loss1 only`
- Freeze A: true
- Train B: true
- Optimizer: B decoders + projectors only
- Planned steps: 5000
- Save checkpoints: `step1000`, `step2500`, `step5000`, `final`
- Save optimizer/scheduler/scaler: false
- Run directory: `/data/zxl/runs/ab_loss1_shortcut_formal/h0_heima_b_probe`
- Log: `/data/zxl/runs/ab_loss1_shortcut_formal/logs/h0_heima_b_probe.log`
- Tmux: `ab_loss1_shortcut_formal: 1 windows (created Fri Jul 24 04:17:03 2026)`
- PID: `2145886 /root/miniconda3/envs/st/bin/python /data/zxl/Heima-ab-loss1-shortcut-formal/scripts/heima_alignment/ab_loss1_shortcut_formal.py --train --group h0_heima_b_probe`
- Command: `/root/miniconda3/envs/st/bin/python /data/zxl/Heima-ab-loss1-shortcut-formal/scripts/heima_alignment/ab_loss1_shortcut_formal.py --train --group h0_heima_b_probe`

## GPU At Start

```text
0, NVIDIA GeForce RTX 4090, 0 MiB, 49140 MiB, 0 %
1, NVIDIA GeForce RTX 4090, 0 MiB, 49140 MiB, 0 %
2, NVIDIA GeForce RTX 4090, 0 MiB, 49140 MiB, 0 %
3, NVIDIA GeForce RTX 4090, 0 MiB, 49140 MiB, 0 %
4, NVIDIA GeForce RTX 4090, 0 MiB, 46068 MiB, 0 %
5, NVIDIA GeForce RTX 4090, 0 MiB, 49140 MiB, 0 %
6, NVIDIA GeForce RTX 4090, 0 MiB, 49140 MiB, 0 %
7, NVIDIA GeForce RTX 4090, 0 MiB, 49140 MiB, 0 %
```

## Latest Log Snippet

```text
/root/miniconda3/envs/st/lib/python3.11/site-packages/torch/cuda/__init__.py:61: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you.
  import pynvml  # type: ignore[import]
Using a slow image processor as `use_fast` is unset and a slow processor was saved with this model. `use_fast=True` will be the default behavior in v4.52, even if the model was saved with a slow processor. This will result in minor differences in outputs. You'll still be able to use a slow processor with `use_fast=False`.
Skipping import of cpp extensions due to incompatible torch version 2.6.0+cu124 for torchao version 0.15.0             Please see https://github.com/pytorch/ao/issues/2919 for more info

Loading checkpoint shards:   0%|          | 0/2 [00:00<?, ?it/s]
Loading checkpoint shards: 100%|██████████| 2/2 [00:00<00:00, 35.42it/s]
Sliding Window Attention is enabled but not implemented for `sdpa`; unexpected results may be encountered.
/root/miniconda3/envs/st/lib/python3.11/site-packages/torch/cuda/__init__.py:61: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you.
  import pynvml  # type: ignore[import]
{"step": 1, "main_loss": 1.3156023025512695, "loss1_mean": 1.9980440139770508, "loss1_by_section": {"summary": 2.073214054107666, "caption": 2.3452112674713135, "reasoning": 1.5757063627243042}, "total": 1.9980440139770508, "grad_A_total": 0.0, "grad_B_total": 35.75615473054087}
{"step": 50, "main_loss": 1.0502114295959473, "loss1_mean": 1.152591586112976, "loss1_by_section": {"summary": 0.8385462164878845, "caption": 1.5988309383392334, "reasoning": 1.0203973054885864}, "total": 1.152591586112976, "grad_A_total": 0.0, "grad_B_total": 20.7454773212953}
{"step": 100, "main_loss": 0.36557504534721375, "loss1_mean": 1.5514777898788452, "loss1_by_section": {"summary": 1.7879809141159058, "caption": 1.6829144954681396, "reasoning": 1.1835378408432007}, "total": 1.5514777898788452, "grad_A_total": 0.0, "grad_B_total": 19.66818708594594}
```

## Not Started Yet

- `h1_joint_ab_loss1` not started.
- `h2_frozen_b_loss1_to_a` not started; it is blocked until H0 final B exists.
- No Model-A-only training was started.
- No old run directory, checkpoint, or prior result was deleted or overwritten.
