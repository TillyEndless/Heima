# Heima A+B Loss1 Tiny Real-Image Acceptance

Status: dry-run/protocol-ready unless this script is launched without --dry-run under a trainer implementation.

Data:
- subset: /data/zxl/official_heima/micro_subsets/chartqa_sqa_v1
- split: /data/zxl/runs/heima_ab_loss1_tiny_acceptance_v1/data_split.json
- requested train/eval: 192/48
- actual train/eval: see data_split.json; exact_requested_size indicates whether the strict request was met
- full LLaVA-CoT-100k train.jsonl: not used

Schedule:
- Stage 0: explicit CoT SFT
- Stage 1: reasoning-only latent replacement
- Stage 2: freeze A, train B_reasoning
- Stage 3 baseline: Loss1 with detached A latent
- Stage 3 ours: Loss1 with non-detached A latent

Required evaluation outputs for a real run:
- answer accuracy
- reasoning reconstruction NLL
- deterministic generation examples
- correct latent vs shuffle latent
- zero latent
- question-only baseline

Interpretation:
- This is mechanism validation, not benchmark reproduction.
- Passing this run validates real image loading and Loss1 wiring on existing data.
- It does not reproduce full Heima paper metrics or full LLaVA-CoT-100k training.
