# HEIMA-CORE-PARITY-MIGRATION

Status: migration/parity only. No formal training, Loss2, K expansion, or model change was run.

## Direct Official Imports

- CEWithChunkedOutputLoss: actual backend `torchtune.modules.loss.ce_chunked_output_loss.CEWithChunkedOutputLoss`, fallback_used=False.

## Copied / Mirrored Official Logic

- Shifted thinking-token mask mirrors Heima training lines 1503-1532.
- Embedding replacement mirrors Torchtune transformer lines 669-701.
- Official-compatible projector class mirrors transformer lines 404-416.

## HF Compatibility Adapters

- GPT-2 uses `inputs_embeds` replacement because HF GPT-2 has no `thinking_token` forward API.
- GPT-2 model adapters are represented by `HeimaEncoderInterface` and `HeimaDecoderInterface` wrappers.

## Key Findings

- Shifted hidden is not the same as the contextual hidden at `<THINKING>` under HF GPT-2. It selects the previous position that predicts `<THINKING>`. Use backend-specific adapter and do not mechanically enable it unless matching official predictor-hidden semantics.
- Current `LayerNorm -> Linear` projector does not match official `Linear -> ReLU -> Linear -> Dropout`.
- Loss value and gradient parity against HF shifted CE is covered by parity tests.
- detach=true/false is centralized in `prepare_latent_for_decoder(z, detach_encoder_latent)`.

## Next Training Permission

Do not enter next-stage training yet. Review this report and decide whether the experiment should use direct `<THINKING>` hidden or official shifted predictor hidden for the text-only GPT-2 backend.
