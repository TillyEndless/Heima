# Model A Self-Decoder

This branch adds an independent experiment path where Model A both produces and
decodes latent thinking states. It does not remove or change the existing
external Model B baseline.

## Relation To Tencent Hidden Decoding

Tencent Hidden Decoding is a same-forward multi-stream token expansion method.
It does not send a first forward's final hidden states into later independent
forwards. Its stream embeddings are full-vocabulary embedding tables, not this
project's CoT-stage special tokens.

This project only borrows three ideas:

- the same backbone can produce and read latent states;
- text losses can indirectly supervise latent states;
- intermediate latent markers do not necessarily need direct NTP supervision.

## Actual Method

The producer pass is unchanged:

```text
image + question + typed thinking tokens + answer
```

It computes `L_main` and extracts strict Heima predictor states:

```text
z_i = hidden_state[thinking_position_i - 1]
```

For each stage, a second text-only self-decoder forward uses the same Model A
parameter object:

```text
Question
Instruction for stage i
Latent slot <THINKING_OF_*>
Reasoning target text
```

The latent slot special token is only a locator. Its embedding is replaced by:

```text
adapter_i(z_i) + role_i
```

where `role_i` is optional and represents stage identity.

## Forward Modes

`sequential` is the reference mode:

```text
Forward 0: producer A-main
Forward 1..N: A-self-i
loss_total.backward()
optimizer.step()
```

`batched` stacks explanation samples along the batch dimension and is only a
performance optimization. Deterministic tests verify parity with sequential
losses and gradients.

## Label Modes

`text_only` masks question, instruction, and latent slot with `-100`; only the
target explanation text contributes to the self-decoder loss.

`latent_and_text` additionally labels the typed latent marker token. This is an
ablation for testing whether latent-marker NTP helps or merely teaches format.

## Safety Rules

- The self-decoder calls the same `model_a` object directly.
- It passes `inputs_embeds`, not `input_ids`.
- It passes no image, `pixel_values`, `image_grid_thw`, video inputs, or image
  grids during self decoding.
- The formal no-detach method uses `z_i` directly.
- The detach ablation uses `z_i.detach()`.
- The embedding table is never modified in place.
- All losses are summed before a single backward and optimizer step.
