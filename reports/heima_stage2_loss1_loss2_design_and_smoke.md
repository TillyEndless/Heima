# Strict Heima Stage2 Loss1 + Loss2 Design and Smoke Report

Date: 2026-07-23
Branch: `feat/heima-stage2-loss1-loss2`
Base branch: `feat/heima-stage2-interp-supervision`
Base commit: `a71a7d7978e84b9b2151d5c4a2f6791951e0ee0a`
Feature commit: recorded by git after this report is committed.

## Reuse vs New Implementation

Existing Loss2 branches were audited and found not directly usable as formal strict Heima Stage2 main+Loss1+Loss2 branches. This branch is a new implementation based on the strict Stage2 code before `feat/model-a-only-self-decode`.

It is fully isolated from `feat/model-a-only-self-decode`; the branch was created from `feat/heima-stage2-interp-supervision`, not from the Model-A-only branch.

## Added / Changed Files

- Added `src/heima_stage2/loss2_alignment.py`
- Added `tests/heima_stage2/test_loss2_alignment.py`
- Added `configs/heima_stage2/stage2_loss1_loss2.yaml`
- Added `reports/loss1_loss2_branch_audit.md`
- Added `reports/heima_stage2_loss1_loss2_design_and_smoke.md`
- Updated `src/heima_stage2/__init__.py`
- Updated `scripts/run_small_vlm_stage2_interp_supervision.py`

## Strict Heima Alignment

The implementation preserves the existing strict Heima-aligned flow:

Stage0:

- Model A trains on explicit `summary, caption, reasoning, answer` section/answer language.
- A checkpoint is saved as `s0_encoder.pt`.

Stage1:

- Model A is frozen.
- Model B interpreters and projectors are trained with Loss1 section reconstruction.
- A canonical Stage1 checkpoint is saved as `s1_staged_sections.pt`.

Stage2:

- Each mode reloads the same Stage1 state.
- Model A is trainable.
- Model B interpreters and projectors are frozen.
- The optimizer contains Model A parameters only.
- Existing `heima_baseline` and `ours_interp_supervision` modes are retained.
- New `ours_loss1_loss2` mode is added.

## Loss1 Formula

For each section `i` in `summary, caption, reasoning`:

`L_loss1_i = CE(B(explain_prompt_i, question/context, z_i, text_cot_i), text_cot_i)`

where `z_i` is extracted from Model A's last hidden state at the latent CoT marker position and projected into B hidden size through the frozen Stage1 projector.

Baseline detaches `z_i`; ours modes keep `z_i` attached.

## Loss2 Formula

Loss2 is computed entirely in the same frozen Model B hidden space.

Latent path:

`B(explain_prompt_i, question/context, z_i, text_cot_i)`

This produces target-token hidden states `H_latent_i` from B.

Text path:

`B(question/context, text_cot_1 ... text_cot_i)`

This produces target-token hidden states `H_text_i` from the same B model under `torch.no_grad()`. The text feature is detached and used as the teacher target.

Pooling:

- `loss2_pool = mean` by default
- `loss2_pool = last` also supported

Distance:

- default `normalized_mse(normalize(pool(H_latent_i)), normalize(pool(H_text_i)))`
- `mse` and `cosine` are also supported

Total loss in new mode:

`L_total = L_main + lambda_loss1 * L_loss1 + lambda_loss2 * L_loss2`

Defaults:

- `lambda_loss1 = 0.1`
- `lambda_loss2 = 0.05`

## Gradient Attribution Contract

Expected by mode:

| mode | grad_A_from_loss1 | grad_A_from_loss2 | grad_B_from_loss1 | grad_B_from_loss2 | optimizer_contains_B |
|---|---:|---:|---:|---:|---:|
| `heima_baseline` | 0 | 0 | 0 | 0 | false |
| `ours_loss1` / `ours_interp_supervision` | >0 | 0 | 0 | 0 | false |
| `ours_loss1_loss2` | >0 | >0 | 0 | 0 | false |

The runner records:

- `L_main`
- `L_loss1`
- `L_loss2`
- `L_total`
- per-section Loss1
- per-section Loss2
- `grad_A_from_loss1`
- `grad_A_from_loss2`
- `grad_B_from_loss1`
- `grad_B_from_loss2`
- `teacher_B_frozen`
- `optimizer_contains_B`
- `H_latent_i` shape
- `H_text_i` shape
- `loss2_pool`
- `lambda_loss1`
- `lambda_loss2`

## Tests

Commands run:

`python3 -m py_compile src/heima_stage2/loss2_alignment.py scripts/run_small_vlm_stage2_interp_supervision.py src/heima_stage2/__init__.py`

`PYTHONPATH=. /data/zxl/conda_envs/nlp-final/bin/pytest -q tests/heima_stage2/test_loss2_alignment.py tests/heima_stage2/test_interp_supervision.py`

Result:

`11 passed in 2.24s`

The new tests cover:

- Loss2 hidden shapes match
- Loss2 finite
- `H_text_i` detached
- B parameters frozen
- optimizer excludes B
- baseline `grad_A_from_loss2 = 0`
- `ours_loss1_loss2` `grad_A_from_loss2 > 0`
- Loss2 does not directly compare A hidden and B hidden
- `pool=last` and `pool=mean` both run
- one-batch smoke total loss finite

## Dry Run

Command run:

`PYTHONPATH=. /data/zxl/conda_envs/nlp-final/bin/python scripts/run_small_vlm_stage2_interp_supervision.py --dry-run --out /data/zxl/runs/heima_stage2_loss1_loss2_dryrun_v0 --s0-steps 1 --s1-steps 1 --stage2-steps 1 --eval-samples 1 --generation-samples 1`

Latest dry-run manifest:

`/data/zxl/runs/heima_stage2_loss1_loss2_dryrun_v0/seed42/20260723_124845/experiment_manifest.json`

Dry-run exited before Stage0/Stage1/Stage2 training and confirmed the manifest includes the new `stage2_ours_loss1_loss2` contract.

## Formal Training Status

No formal training, long training, or overnight job was started. No checkpoints or old runs were deleted. The Model-A-only branch was not modified.
