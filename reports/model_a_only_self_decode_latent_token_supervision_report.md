# Model-A-only Latent Token Supervision Smoke Report

Branch: `feat/model-a-only-self-decode-latent-token-supervision`

Base commit: `2c3947f`

Smoke run: `/data/zxl/runs/model_a_only_self_decode_latent_token_supervision_smoke/seed42/20260724_144318`

## Contract

- No Model B is created.
- No projector is created.
- No role embedding is created.
- Self-decode still uses direct `inputs_embeds` insertion of `last_hidden_state_of_latent_token`.
- Loss is `main_loss + lambda_self * self_decode_loss + latent_token_loss_weight * latent_token_loss`.

## Smoke Result

The real single-batch smoke produced finite losses. In self-decode mode:

- `grad_A_from_latent_token_norm` > 0
- `grad_A_from_self_decode_norm` > 0
- `grad_z_summary/caption/reasoning` > 0

In baseline mode:

- self-decode gradient to z remains 0
- latent token supervision still gives gradient to A

See:

- `reports/model_a_only_self_decode_latent_token_supervision_smoke.json`
- `reports/trainable_parameter_report.json`

## Eval Additions

The training entrypoint now emits greedy answer substring accuracy under `answer_accuracy` in `result.json`, alongside latent intervention metrics including `correct`, `shuffle`, `zero`, `q_only`, and `latent_only`.
