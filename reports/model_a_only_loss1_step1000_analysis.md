# Model-A-only Loss1 Step1000 Analysis

Generated: 2026-07-24 03:47:46 UTC

## Decision

`DECODER_SHORTCUT_RISK`

Step1000 shows clear Loss1 optimization and nonzero gradient flow into Model A, but the latent intervention signal is not statistically separated from shuffle.

## Main Task

- validation main NLL: `0.377612`
- answer generation accuracy: `0.869141`

## Loss1 Reconstruction

| section | CE loss | token accuracy | correct NLL | shuffle NLL | shuffle margin | zero margin | q_gain |
|---|---:|---:|---:|---:|---:|---:|---:|
| summary | 1.070328 | 0.696890 | 1.070328 | 1.070465 | 0.00013733 | 0.00241089 | 0.00030518 |
| caption | 1.257095 | 0.675241 | 1.257095 | 1.257095 | 0.00000000 | 0.00610352 | 0.00096130 |
| reasoning | 1.312515 | 0.656083 | 1.312515 | 1.312576 | 0.00006104 | 0.00218201 | 0.00143433 |

## Latent Intervention Bootstrap

- mean correct-shuffle margin: `0.00006612`
- 95% bootstrap CI: `[-0.00027974, 0.00040690]`
- bootstrap samples over margins: `768`

The CI crosses 0, so step1000 does not yet support `PASS_LATENT_SEMANTIC_SIGNAL`.

## Gradient Audit

- grad_z_summary: `0.0001400141`
- grad_z_caption: `0.0002671927`
- grad_z_reasoning: `0.0004625732`
- grad_A_from_loss1: `15.262592`
- pass: `True`

Loss1 is connected to latent z and Model A, but the evaluation indicates the model can reconstruct largely from question/text priors rather than sample-specific latent content.

## Action

Do not spend additional GPU on new continuation. Because the original run had already reached step5000 before this early-stop policy was applied, the step2500/step5000 checkpoints are kept on disk, but no additional training was launched and no step2500/step5000 evaluation is run under this decision.
