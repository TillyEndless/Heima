# Model-A-only Loss1 Early Stop Report

Generated: 2026-07-24 03:47:46 UTC

## 1. Step1000

Reached: yes. Training had already completed to step5000 before this early-stopping audit began, but `model_a_step1000.pt` exists and was evaluated offline.

- checkpoint: `/data/zxl/runs/model_a_only_loss1_formal/seed42/20260723_172211/checkpoints/model_a_step1000.pt`
- eval dir: `/data/zxl/runs/model_a_only_loss1_formal/eval_step1000/`
- main NLL: `0.377612`
- answer acc: `0.869141`
- mean Loss1 CE: `1.213313`
- mean shuffle margin: `0.00006612`
- bootstrap CI: `[-0.00027974, 0.00040690]`
- decision: `DECODER_SHORTCUT_RISK`

## 2. Continue Step2500?

No new continuation and no formal step2500 eval. Rule Case 2 applies: Loss1 improves, but correct-shuffle margin is approximately 0 and bootstrap CI crosses 0.

The step2500 checkpoint already exists from the completed run and is preserved, not deleted.

## 3. Continue Step5000?

No. Step5000 checkpoint/final already exist from the completed run and are preserved, but the early-stop decision would have stopped at step1000.

## 4. Latent Intervention

| section | correct | shuffle | zero | q_only | shuffle margin | zero margin | q_gain |
|---|---:|---:|---:|---:|---:|---:|---:|
| summary | 1.070328 | 1.070465 | 1.072739 | 1.070633 | 0.00013733 | 0.00241089 | 0.00030518 |
| caption | 1.257095 | 1.257095 | 1.263199 | 1.258057 | 0.00000000 | 0.00610352 | 0.00096130 |
| reasoning | 1.312515 | 1.312576 | 1.314697 | 1.313950 | 0.00006104 | 0.00218201 | 0.00143433 |

## 5. Does This Prove Loss1 Shapes Latent?

No. The gradient audit proves Loss1 can backpropagate into z and Model A, but step1000 does not prove that z carries robust sample-specific semantic information. The correct-shuffle margin is near zero and its bootstrap confidence interval crosses zero.

## 6. Overfit / Shortcut Signs

There is a shortcut risk: reconstruction CE/token accuracy look reasonable, but intervention barely distinguishes correct latent from shuffled latent. This is consistent with question/text-pattern reconstruction rather than latent-grounded reconstruction.

## 7. Files

- metrics: `/data/zxl/runs/model_a_only_loss1_formal/eval_step1000/metrics.json`
- gradient audit: `/data/zxl/runs/model_a_only_loss1_formal/eval_step1000/gradient_step1000.json`
- generations: `normal_generation.jsonl`, `shuffle_generation.jsonl`, `zero_generation.jsonl`
- analysis: `/data/zxl/Heima-model-a-only-loss1-formal/reports/model_a_only_loss1_step1000_analysis.md`
- curve: `/data/zxl/Heima-model-a-only-loss1-formal/reports/model_a_only_loss1_training_curve.md`
