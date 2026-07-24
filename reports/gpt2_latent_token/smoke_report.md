# GPT2 Latent Token Ablation Smoke Report

Smoke run only: 32 train samples, 8 eval samples, 1 optimizer step per group. This is not formal training and should not be used as evidence for or against the latent-token hypothesis.

Artifacts:

- Run directory: `/data/zxl/runs/gpt2_latent_token_ablation_smoke`
- Split: `/data/zxl/runs/gpt2_latent_token_ablation_smoke/data_split.json`
- Comparison: `reports/gpt2_latent_token/comparison.json`
- Trainable parameter audit: `reports/gpt2_latent_token/trainable_parameter_reports.json`

Implemented groups:

- G0: vanilla CoT SFT, no latent path.
- G1: main answer loss plus latent token CE for `<THINK>`.
- G2: main answer loss plus self-decode CE from continuous `<THINK>` hidden state.
- G3: main answer loss plus self-decode CE plus latent token CE.

Safety constraints satisfied:

- No Model B.
- No projector.
- No role embedding.
- No cumulative latent.
- No Loss2.
- No formal training launched.
