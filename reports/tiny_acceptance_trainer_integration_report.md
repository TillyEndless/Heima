# Tiny Acceptance Trainer Integration Report

This integrates the tiny real-image acceptance data with the existing strict/scaled A+B Loss1 trainer. It does not introduce a new training framework.

## Existing Trainer Audited

- Reused script: `scripts/run_data_small_vlm_official_sections.py`
- Reused functions: `encoder_forward`, `decoder_forward`, `prepare_stage_latents`, `attribution`, `train_s0`, `train_s1`, `train_joint`, `evaluate`, `save_generation_eval`
- Reused detach switch: `prepare_stage_latents(... detach_encoder_latent=...)` -> `prepare_latent_for_decoder`
- Reused optimizer/checkpoint path from existing trainer; wrapper only selects data/split/run layout.
- Existing trainer is multi-section (`summary`, `caption`, `reasoning`); tiny acceptance reporting focuses on reasoning diagnostics without changing the trainer forward/loss path.

## Tiny Data
- split: `/data/zxl/runs/heima_ab_loss1_tiny_acceptance_v1/data_split.json`
- train/eval: 167/45

## Backward Smoke

- detach grad_A_from_loss1: `0.0`
- no-detach grad_A_from_loss1: `22.82187856655761`
- Expected: detach is zero; no-detach is finite and greater than zero.

## Formal Entrypoints

- Baseline: `python scripts/heima_alignment/run_tiny_acceptance_train.py --mode detach` writes under `runs/heima_ab_loss1_tiny_acceptance_v1/baseline/`.
- Ours: `python scripts/heima_alignment/run_tiny_acceptance_train.py --mode no_detach` writes under `runs/heima_ab_loss1_tiny_acceptance_v1/ours/`.
- Smoke only: add `--stage smoke`; no optimizer step or checkpoint is saved.

## Semantic Evaluator Contract

The integration keeps the existing trainer evaluation and adds the tiny acceptance contract for downstream summarization: reasoning reconstruction NLL, content token NLL, numeric/entity/answer token accuracy, Q-only/correct/shuffle/zero interventions, generation exact match, and answer accuracy.
