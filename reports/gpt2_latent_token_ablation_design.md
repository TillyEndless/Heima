# GPT2 Latent Token Supervision Ablation

This branch adds a pure-text GPT2 ablation for testing whether language supervision can shape a continuous latent token without Model B or any extra projection module.

## Scope

- Model: `GPT2ForCausalLM`, default GPT2-small, optionally GPT2-medium.
- Shared parameters: the same GPT2 produces the `<THINK>` hidden state and decodes from it.
- Explicitly absent: Model B, projector, role embedding, cumulative latent, Loss2, visual inputs.
- Latent source: final hidden state at the `<THINK>` token from the first forward.
- Latent injection: second forward uses `inputs_embeds`; the latent slot embedding is replaced directly with `z`.

## Experiment Groups

| group | main loss | self decode | latent token CE | purpose |
|---|---:|---:|---:|---|
| G0 | yes | no | no | vanilla CoT SFT baseline |
| G1 | yes | no | yes | latent token supervision only |
| G2 | yes | yes | no | self decode only |
| G3 | yes | yes | yes | full method |

Default loss weights are `lambda_self=0.1` and `lambda_latent=0.05`.

## Smoke Only

The implemented entrypoint defaults to a 32-train / 8-eval smoke run. It verifies forward, backward, trainable parameter audit, checkpoint save/load, intervention metrics, and generation files. Formal training is intentionally blocked until `--smoke` results are inspected.
