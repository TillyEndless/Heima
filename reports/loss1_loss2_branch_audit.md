# Loss1 + Loss2 Branch Audit

Date: 2026-07-23
Audit worktree: `/data/zxl/Heima-stage2-interp-supervision`
Current implementation branch after audit: `feat/heima-stage2-loss1-loss2`

## Candidates Checked

| candidate | branch | commit | contains Loss2 | strict Heima Stage0/1/2 aligned | B frozen in Stage2 | optimizer excludes B | direct formal branch? |
|---|---|---:|---:|---:|---:|---:|---:|
| `/data/zxl/Heima-align-ab-loss1-loss2` | `feat/heima-aligned-ab-loss1-loss2` | `030d1cd` | yes | no | only some protocol modes | no for several modes | no |
| `/data/zxl/Heima-loss2-clean` | `feat/loss2-frozen-b-teacher` | `d78622b` | yes | no | teacher only | no for B_dec | no |
| `/data/zxl/Heima-stage2-interp-supervision` | `feat/heima-stage2-interp-supervision` | `a71a7d7` | no | yes for strict Stage2 Loss1 pilot | yes | yes | base only |

## Candidate Details

### `feat/heima-aligned-ab-loss1-loss2`

Evidence read:

- `src/g1/loss2_teacher.py`
- `scripts/run_g1_loss2_smoke.py`
- `src/heima_aligned/protocol.py`
- `tests/heima_aligned/test_loss2_protocol.py`
- `configs/heima_aligned/ab_loss1_loss2_qwen_vl3b.yaml`

Loss2 exists, but it is not the required strict Stage2 implementation.

Implemented formula:

`h_L = B_dec(question, latent z, <SEM>, teacher-forced whole CoT)[feature_pos]`

`h_T = stopgrad(B_teacher(question, explicit whole CoT, <SEM>)[feature_pos])`

`Loss2 = distance(h_L, h_T)` with cosine/MSE/normalized-MSE options.

Problems for this task:

- It is G1 / whole-CoT oriented (`record["cot"]`), not strict Heima sections `summary, caption, reasoning` as the actual training path.
- It uses two B-side models, `B_dec` and `B_teacher`, rather than the same frozen Stage1 B producing latent-path and text-path features.
- `scripts/run_g1_loss2_smoke.py` adds `model_b_dec.parameters()` to the optimizer whenever Loss1 or Loss2 is enabled.
- `mode_plan("main_loss1_loss2")` and several loss2 modes have `train_b: True`.
- Existing tests check protocol flags and teacher freezing, but do not establish same Stage0/Stage1 state, same canonical B checkpoint, frozen B in Stage2, or optimizer exclusion for the actual B decoder.

Conclusion: useful Loss2 prototype and semantic audit code, not a direct formal strict Heima Stage2 main+Loss1+Loss2 branch.

### `feat/loss2-frozen-b-teacher`

Evidence read:

- `src/g1/loss2_teacher.py`
- `tests/g1/test_loss2_teacher.py`
- `docs/loss2_frozen_teacher.md`

Loss2 exists, but the docs explicitly describe a frozen teacher plus a trainable `B_dec`:

`L2 -> h_L -> B_dec -> injected latent -> z -> producer Model A`

This is not the requested strict Stage2 setup where the Stage1 interpreter B is frozen in Stage2 and excluded from the optimizer. It is a clean foundational Loss2 smoke branch, but not a strict Heima Stage0/1/2 aligned branch.

### `feat/heima-stage2-interp-supervision`

Evidence read:

- `scripts/run_small_vlm_stage2_interp_supervision.py`
- `src/heima_stage2/interp_supervision.py`
- `tests/heima_stage2/test_interp_supervision.py`
- `configs/heima_stage2/stage2_compare.yaml`

This is the correct strict base:

- Stage0 trains A on explicit section/answer language.
- Stage1 freezes A and trains B interpreters plus projectors for `summary, caption, reasoning`.
- Stage2 reloads the same Stage1 state for both arms.
- Stage2 freezes B and projectors.
- Stage2 optimizer contains A only.
- baseline uses `L_total = L_NTP` with detached interpreter eval.
- ours uses `L_total = L_NTP + lambda_interp * L_interp` with z attached.

But it had no Loss2 before this task.

## Decision

No existing branch fully satisfied all requirements for strict `main + Loss1 + Loss2`:

- strict Stage0/1/2 alignment
- same A init
- same canonical Stage1 B checkpoint
- B frozen in Stage2
- optimizer excludes B
- Loss1 can update A in ours
- Loss2 can update A through the latent path
- Loss2 aligns latent-path and text-path hidden features in the same frozen Model B hidden space

Therefore, a new branch was created from `feat/heima-stage2-interp-supervision`:

`feat/heima-stage2-loss1-loss2`
