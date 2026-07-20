# Model A Self-Decoder Audit

This document records the current external-interpreter path before adding the
independent `self_a` experiment branch. The existing `external_b` path must
remain runnable and checkpoint-compatible.

## Current Entry Points

- Whole-reasoning small VLM path:
  `scripts/run_data_small_vlm_l1.py`
- Official section path:
  `scripts/run_data_small_vlm_official_sections.py`
- Progressive section S0 path:
  `scripts/run_data_small_vlm_progressive_sections.py`
- Progressive frozen-interpreter path:
  `scripts/run_data_small_vlm_progressive_interpreters.py`
- Progressive joint Loss1 path:
  `scripts/run_data_small_vlm_progressive_joint_l1.py`

## Model A Initialization And Forward

- `scripts/run_data_small_vlm_official_sections.py::load_vlm_a`
  initializes Model A with:
  `AutoModelForImageTextToText.from_pretrained`.
- `scripts/run_data_small_vlm_official_sections.py::vlm_inputs`
  constructs an image-plus-text chat input.
- `scripts/run_data_small_vlm_official_sections.py::encoder_forward`
  calls:
  `model_a(**full, output_hidden_states=True, use_cache=False)`.

The current Model A producer forward sees image plus question and produces
`L_main`, logits, labels, and typed thinking hidden states.

## External Model B Initialization And Forward

- `scripts/run_data_small_vlm_official_sections.py::load_decoder_b`
  initializes Model B with:
  `AutoModelForCausalLM.from_pretrained`.
- `scripts/run_data_small_vlm_official_sections.py::decoder_forward`
  constructs `Question + Instruction + typed thinking slot + Target`.
- The current interpreter path replaces the typed thinking slot embedding and
  calls:
  `model_b(inputs_embeds=embeds, attention_mask=attention, use_cache=False)`.

Optimizer groups include Model A during joint training and include each external
Model B plus projector. `self_a` must not instantiate or optimize Model B.

## Typed Thinking Tokens

The official-section small VLM path registers:

- `<THINKING_OF_SUMMARY>`
- `<THINKING_OF_CAPTION>`
- `<THINKING_OF_REASONING>`

in `scripts/run_data_small_vlm_official_sections.py::load_vlm_a` and
`load_decoder_b`.

## Latent Extraction

- Extraction helper:
  `src/htext/heima_reuse.py::extract_thinking_state`
- Current mode:
  `mode="predictor"`
- Current semantics:
  `z_i = hidden_state[thinking_position_i - 1]`

This matches the strict Heima-repo aligned predictor state. The new self-decoder
branch must not change this extraction semantic.

## Projector And Replacement

- Official-compatible projector:
  `src/htext/heima_reuse.py::HeimaOfficialAbstractProjection`
- Official-compatible embedding replacement:
  `src/htext/heima_reuse.py::official_embedding_replacement`

The external-B path injects `projector(z)` at the typed thinking-token slot.
The new self-A path injects `adapter_i(z_i) + optional_role_i` into Model A's
own input embedding stream.

## Joint Detach And Ours Entry

- `scripts/run_data_small_vlm_official_sections.py::train_joint`
  calls `prepare_latent_for_decoder(z[s], detach)`.
- `detach=True` is the baseline/joint-detach condition.
- `detach=False` is the no-detach/Ours-L1 condition.

The only intended algorithmic difference is whether Loss1/self loss can
backpropagate through the producer latent.

## Prompt, Target, And Labels

In `scripts/run_data_small_vlm_official_sections.py::decoder_forward`:

- Prompt contains question, instruction, and the typed thinking slot.
- Target is one of `summary`, `caption`, `reasoning`.
- Default labels mask prompt/question/latent slot with `-100` and train only
  target text.
- The recent latent-marker ablation enables the typed latent slot label while
  keeping prompt/question ignored.

## Qwen2.5-VL Text-Only `inputs_embeds` Check

On `/data`, `Qwen2_5_VLForConditionalGeneration` was tested in eval mode with
no image tensors. The top-level forward accepted:

```python
base_embeds = model_a.get_input_embeddings()(input_ids)
out = model_a(
    inputs_embeds=base_embeds,
    attention_mask=attention_mask,
    use_cache=False,
    output_hidden_states=True,
)
```

The smoke test produced the same logits shape as the `input_ids` path and
`max_abs_diff = 0.0`. Therefore the self-A implementation should prefer the
top-level Qwen2.5-VL forward and pass no `input_ids`, image, `pixel_values`, or
`image_grid_thw` in the self-decoder forward.
