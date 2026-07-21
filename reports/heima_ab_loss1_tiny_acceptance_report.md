# Heima A+B Loss1 Tiny Real-Image Acceptance

Status: semantic diagnostics implemented; no training was launched.

This is mechanism validation, not benchmark reproduction. It uses the existing local ChartQA/SQA micro subset to test whether the scaled A+B Loss1 path can load real images, replace the reasoning section with a typed latent token, train/read a reasoning interpreter, and distinguish correct latent from interventions.

## Data

- subset: `/data/zxl/official_heima/micro_subsets/chartqa_sqa_v1`
- requested split: `train=192`, `eval=48`, `seed=42`
- default split path: `/data/zxl/runs/heima_ab_loss1_tiny_acceptance_v1/data_split.json`
- exact split status: blocked by missing local images (`available_train=167`, `available_eval=45`)
- local-only split option: `--allow-smaller-available` writes `train=167`, `eval=45`
- full 98,582-row `train.jsonl`: not used

## Schedule

- Stage 0: explicit CoT SFT
- Stage 1: reasoning-only latent replacement
- Stage 2: freeze A and train `B_reasoning`
- Stage 3 baseline: Loss1 with detached A latent
- Stage 3 ours: Loss1 with non-detached A latent

## Reasoning Token-Level Evaluation

The evaluator contract now requires metrics that are less diluted by template text:

- full reasoning NLL
- content token NLL
- numeric token accuracy
- entity token accuracy
- answer token accuracy

## Latent Intervention Evaluation

For each example, the evaluator must keep question, decoder prompt, teacher target, labels, and attention mask fixed, changing only the injected latent:

- Q-only
- Q + correct latent
- Q + shuffle latent
- Q + zero latent

Required metrics:

- full NLL
- content NLL
- generation exact match

The primary latent-use signal is `correct.full_nll < shuffle.full_nll` and `correct.content_nll < shuffle.content_nll`, with generation exact match checked as a qualitative/semantic guard.

## Warm-B Interface

The interface now distinguishes:

- cold-B joint: joint A+B Loss1 starts from fresh pretrained B
- warm-B joint: joint A+B Loss1 starts after `freeze_A_train_B`

The intended comparison holds Model A checkpoint, split, batch order, optimizer hyperparameters, `lambda_loss1`, reasoning target, and intervention evaluator fixed.

## Interpretation

Passing this acceptance only supports the claim that the mechanism is wired and latent-sensitive on a tiny real-image subset. It does not reproduce Heima paper benchmark metrics or full LLaVA-CoT-100k training.
