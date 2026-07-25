# Latent Extraction Comparison

## Position Semantics

Official Heima shifted extraction selects the hidden state that predicts the thinking token, not the contextual hidden state after consuming the thinking token.

- direct thinking token index example: `[10]` token `['<THINKING>']`
- official shifted predictor index example: `[9]` token `['.']`
- off-by-one exists: `True`
- official source note: heima/main_python/2-training...py:1503-1532 masks tokens[:, 1:] == thinking_id, selecting the hidden state that predicts the thinking token under next-token CE.

Current strict code path uses `thinking_state_mode: predictor` / `extract_thinking_state(... mode="predictor")`, which aligns with the official predictor-hidden semantics. A direct `hidden_states[position]` implementation would be misaligned; the strict path instead uses `hidden_states[position-1]`.

## Tensor Metadata

- example direct hidden shape: `[1, 768]`
- example shifted hidden shape: `[1, 768]`
- layer index: last hidden state (`hidden_states[-1]` / model final layer output)
- official run dtype: `bfloat16`
- current run dtype: `bfloat16`

## Embedding Replacement

- decoder input starts from `input_ids` embeddings, then replaces the thinking-token slot with the continuous projected latent via `inputs_embeds`.
- replacement position: after tok_embeddings(tokens), before transformer layers
- gradient to latent nonzero in parity test: `True`
- before replacement embedding norm is printed in the special-token table below. For remove/zero intervention, after replacement latent norm is exactly `0.0` while preserving the latent slot position. Correct-latent projected norms were not saved in the historical checkpoint artifact, so this audit does not invent them; it verifies replacement semantics and reports the recoverable norms.

## Projector

- official projector class/order: `['Linear', 'ReLU(inplace=True)', 'Linear', 'Dropout(p=0.0)']`
- official projector source: `torchtune_pkg/torchtune/torchtune/modules/transformer.py:404-416`
- htext old mismatch noted: `True`; strict official-section runner uses `HeimaOfficialAbstractProjection`.

## Special Tokens

|section|token|token id|before replacement embedding norm|remove/zero after replacement norm|embedding trainable default|
|---|---|---:|---:|---:|---|
|summary|`<THINKING_OF_SUMMARY>`|`151665`|`0.296455`|`0.0`|`True`|
|caption|`<THINKING_OF_CAPTION>`|`151666`|`0.296457`|`0.0`|`True`|
|reasoning|`<THINKING_OF_REASONING>`|`151667`|`0.296474`|`0.0`|`True`|

## Preliminary Conclusion

Hypothesis 1, latent extraction mismatch, is unlikely for the official-section runs audited here: the strict path aligns to Heima predictor-hidden extraction and official-shape projection/replacement. This does not explain the near-zero shuffle margin in the official H0 checkpoint.
