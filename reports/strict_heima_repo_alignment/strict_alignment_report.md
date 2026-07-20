# STRICT-HEIMA-REPO-ALIGNMENT

Status: code/test/report only. No formal training and no Loss2 run.

## Main Path

- `thinking_state_mode` replaces `heima_shifted_hidden`.
- strict_repo configs use `thinking_state_mode: predictor`.
- If the thinking token is at position `p`, Model A latent is `last_hidden_state[:, p-1, :]`.
- The official typed token is `<THINKING_OF_REASONING>`.
- Main sequence is `Question + <THINKING_OF_REASONING> + Answer`; labels include the thinking token and answer, with question/padding ignored.

## Decoder Path

- B prompt contains `Question + Instruction + <THINKING_OF_REASONING>`.
- Projected latent replaces the B-side `<THINKING_OF_REASONING>` embedding after token embedding lookup and before the decoder blocks.
- strict_repo uses `HeimaOfficialAbstractProjection` only; old `LayerNorm -> Linear` is disallowed for strict configs.

## Loss And Backend

- CE backend: `torchtune.modules.loss.ce_chunked_output_loss.CEWithChunkedOutputLoss`.
- fallback_used: `False`.
- detach is centralized in `prepare_latent_for_decoder(z, detach_encoder_latent)`.

## Config Parity

- `strict_repo_joint_detach.yaml` and `strict_repo_joint_no_detach.yaml` differ only in `detach_encoder_latent`.

## Training Permission

Do not start formal strict training until this report is reviewed. The code path is aligned for the next stage, but this task intentionally did not train.
