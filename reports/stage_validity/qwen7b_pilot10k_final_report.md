# Qwen-7B Stage Validity And Pilot10k Final Report

## Outcome
Gate A-C passed. Gate D failed, so the 10k Main-only / Joint / Two-stage pilot was intentionally not launched. This follows the stop condition in the task: if 32-sample true overfit fails, do not start 10k Stage2.

## Gate Table

| gate | status | pass | evidence |
|---|---|---:|---|
| Gate A clean reload | complete | True | forward=True; generate=True; adapter_params=1107326976; think_id=151665 |
| Gate B loss/mask | complete | True | question_ignore=True; think_label=True; model_loss=10.9579 |
| Gate C length/cap | complete | True | latent_cap_rate=0.979; answer_truncation_rate=0.841; cot_truncation_rate=0.981 |
| Gate D true overfit | failed | False | steps=500; think_acc=0.9921875; answer_token_acc=1.0; free_answer_acc=0.375 |

## Gate A Clean Reload
- Base + CURRENT_STAGE1_ADAPTER loaded in a clean subprocess.
- Adapter loaded parameter count: `1107326976`
- `<THINK>` single token: `True`, id `151665`
- forward_ok: `True`; generate_ok: `True`
- checksum: `50d9ea79527af14a18c4439cafa1360d8a105d25a0c0b3b507803cb26d16c982`
- GPU memory was released after subprocess exit according to `nvidia-smi`.

## Gate B Loss Reduction
- `L_main` is token-weighted mean over all non-ignored shifted THINK and answer labels.
- model_loss: `10.957901`; recomputed token_weighted_main_loss: `10.957900`
- think_mean_loss: `18.965313` over `256` tokens
- answer_mean_loss: `1.149777` over `209` tokens
- Therefore previous logs like `loss_total=0.1420`, `loss_think=0.0982`, `loss_answer=0.3498` do not obey `loss_total = loss_think + loss_answer`; total is a weighted token mean.

## Gate C Length/Cap Audit
- latent cap rate: `0.979`
- answer truncation rate: `0.841`
- CoT truncation rate under current decode cap: `0.981`
- gold CoT P50/P95 tokens: `870.0` / `6107.149999999996`
- raw_K P50/P95: `435.0` / `3054.0499999999984`; used_K is capped at `128.0` median
- This means the current setup is capped r=0.5, not strict raw 0.5N. Oracle-K/free-K are not validated.

## Gate D True 32-sample Overfit
- Training ran `L_main` only for 500 steps; no Stage2/self-decode was started.
- Final teacher-forced think_token_accuracy: `0.9921875`
- Final teacher-forced answer_token_accuracy: `1.0`
- Free-generation valid_answer_rate on 8-case probe: `1.0`
- Free-generation answer_accuracy on 8-case probe: `0.375`
- Because required train/free-generation answer accuracy was not reached, overfit_gate=`fail`.

## M0/M1/M2 Pilot Status
- B0 base: not run in this gate-controlled pass.
- M0 Main-only 10k: not run, blocked by Gate D.
- M1 Joint-from-start 10k: not run, blocked by Gate D.
- M2 Two-stage 10k: not run, blocked by Gate D.

## Semantic And Intervention Status
- decoded-vs-gold semantic cosine: not run because M0/M1/M2 checkpoints were not produced.
- correct/shuffle/zero/remove/q-only intervention: not run because Stage2 pilot was blocked.

## Failure Interpretation
The model can overfit token-level teacher-forced `L_main` under the capped protocol, but the free-generation protocol is not yet reliable. The most likely blockers are answer-target formatting and severe truncation/capping: answer truncation is high and many examples have broad textual answers, so numeric parser accuracy is not an adequate pass criterion for this dataset mix.

## Recommendation
Do not scale this exact pilot to 100k/500k yet. First fix Gate D by using a cleaner answer format/subset, an explicit answer delimiter, and an overfit eval that covers all 32 samples with robust answer extraction. After Gate D passes, run the 10k M0/M1/M2 pilot.

