# Model-A-Only Self-Decode Design and Smoke Report

Date: 2026-07-23
Branch: `feat/model-a-only-self-decode`
Base commit before this feature: `a71a7d7978e84b9b2151d5c4a2f6791951e0ee0a`
Final feature commit: recorded by git after committing this report.

## Added Files

- `src/heima_stage2/model_a_only_self_decode.py`
- `scripts/heima_stage2_model_a_only_self_decode.py`
- `configs/heima_aligned/model_a_only_self_decode_small.yaml`
- `tests/heima_stage2/test_model_a_only_self_decode.py`
- `reports/model_a_only_self_decode_design_and_smoke.md`

`src/heima_stage2/__init__.py` was updated to export the new A-only helpers.

## Method

This implements the Model-A-only online self-decode mechanism as a separate path. It does not delete or alter the existing strict Heima Stage2 `heima_baseline` / `ours_interp_supervision` code.

For N sections, default `summary, caption, reasoning`, each batch performs N+1 calls to the same Model A object:

1. First pass: `A(image, question, latent_cot_1 ... latent_cot_N, answer)`
   - computes `L_main`
   - extracts each continuous latent `z_i` from the last hidden state at the section latent marker position

2. Second pass per section: `A(explain_prompt_i, question/context, z_i, section_prefix_i, text_cot_i)`
   - computes `L_cot_i`
   - uses `inputs_embeds` so `z_i` is a continuous latent slot, not an embedding lookup for a special token

Default v0 does not pass the image to self-decode. The config records `self_decode_with_image: false`; the script rejects `--self-decode-with-image` for now because that is reserved for a later ablation.

## Loss

For self-decode training:

`L_self = mean_i(L_cot_i)`

`L_total = L_main + lambda_self * L_self`

The default config sets `lambda_self: 0.05`.

For baseline mode:

`L_total = L_main`

Self-decode forwards are run only for eval/logging under `no_grad` with detached `z_i`, so `grad_A_from_self_decode_norm = 0`.

## Gradient Flow

`a_only_self_decode` mode keeps `z_i` attached. The self-decode loss can update Model A through both paths:

- through the continuous latent slot back into first-pass A
- through the second-pass decoder path of the same A

The helper audits `grad_z_<section>_norm`, `grad_A_from_self_decode_norm`, and `grad_A_total_norm`. The audit uses `torch.autograd.grad` for measurement and keeps the training step itself to one `L_total.backward()`.

## No Model B / No Extra Trainable Modules

Confirmed by implementation and tests:

- `has_model_b: false`
- `optimizer_contains_model_b: false`
- no Model B loader is called in the new entrypoint
- no B checkpoint is saved by the new path
- `use_projector: false`
- `use_role_embedding: false`
- `extra_trainable_params_except_A: 0`

No Linear/MLP/stage projector is introduced. No trainable section role embedding is introduced. Section distinction is carried by the text `explain_prompt_i` and section prefix.

## Latent Intervention Eval

The helper implements per-section self-decode NLL under:

- `correct`: original `z_i`
- `shuffle`: batch-shuffled `z_i`
- `zero`: zero latent
- `q_only`: no latent slot

It reports:

| section | correct | shuffle | zero | q_only | shuffle_margin | zero_margin | qz_gain |
|---|---:|---:|---:|---:|---:|---:|---:|
| each configured section | CE/NLL | CE/NLL | CE/NLL | CE/NLL | shuffle - correct | zero - correct | q_only - correct |

## Tests

Command run:

`PYTHONPATH=. /data/zxl/conda_envs/nlp-final/bin/pytest -q tests/heima_stage2/test_model_a_only_self_decode.py tests/heima_stage2/test_interp_supervision.py`

Result:

`10 passed in 2.30s`

Coverage from the new tests:

- `inputs_embeds` concat shape is correct
- label mask is correct: prompt/question, latent slot, and section prefix are `-100`; target CoT tokens are valid labels
- no Model B audit flags are false/zero as expected
- no projector / role embedding audit flags are false/zero as expected
- N=3 yields exactly 4 Model A forwards per batch
- self-decode mode has `grad_z_i > 0` and `grad_A_from_self_decode_norm > 0`
- baseline mode has `grad_A_from_self_decode_norm = 0`
- one-batch smoke loss is finite
- latent intervention eval emits `correct`, `shuffle`, `zero`, and `q_only` metrics

## Dry Run / Smoke

Entrypoint dry-run command run:

`PYTHONPATH=. /data/zxl/conda_envs/nlp-final/bin/python scripts/heima_stage2_model_a_only_self_decode.py --dry-run --output-dir /data/zxl/runs/model_a_only_self_decode_dryrun_v0 --max-train-samples 1 --max-eval-samples 1`

Latest dry-run manifest directory:

`/data/zxl/runs/model_a_only_self_decode_dryrun_v0/seed42/20260723_122622`

Dry-run confirmed:

- `model_B: null`
- `has_model_b: false`
- `optimizer_contains_model_b: false`
- `use_projector: false`
- `use_role_embedding: false`
- `sections: [summary, caption, reasoning]`
- `self_decode_with_image: false`

A real Qwen/VLM `smoke_backward_only` was not launched in this pass to avoid expensive N+1 large-model forwards. The one-batch finite-loss and gradient smoke is covered by the toy-A unit test. No full training, long training, or overnight job was launched.

## Not Started

No formal training was started. No existing run/checkpoint/result directory was deleted or modified. The old strict Heima Stage2 code path remains intact.
