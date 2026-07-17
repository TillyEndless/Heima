# HText Heima Code-Reuse Migration

Status: in progress. This document separates functional alignment from official-code reuse.

## Official Heima Sources Checked

- `heima/configs/2_1-llama3_2_vision-11b-main_lora-pure_llm_decoder_lora-split_3_stages-fix_num-progressive.yaml`
- `heima/configs/2_5-llama3_2_vision-11b-main_lora-pure_llm_decoder_lora-split_3_stages-fix_num-progressive.yaml`
- `heima/main_python/1_1-organize_dataset-num_thinking_tokens-fix_num.py`
- `heima/main_python/2-training-pipeline-main_lora-pure_llm_decoder_lora-split_3_stages-fix_num-progressive.py`
- `torchtune_pkg/torchtune/torchtune/modules/transformer.py`
- `torchtune_pkg/torchtune/torchtune/modules/loss/ce_chunked_output_loss.py`
- `torchtune_pkg/torchtune/torchtune/data/_collate.py`
- `torchtune_pkg/torchtune/torchtune/data/_messages.py`

## Model Mapping

Official Heima:

- Encoder A: `Llama-3.2V-11B-cot`, `LLAMA3_VISION`, LoRA.
- Interpreter B: `Llama-3.1-8B-Instruct`, `LLAMA3`, LoRA.
- Projector: `abstract_projection` exists in Torchtune transformer construction.

Current HText resource adaptation:

- Encoder A: independent GPT-2 instance.
- Interpreter B: independent GPT-2 instance.
- Projector: lightweight `LayerNorm -> Linear`.

This is a model/resource adaptation, not official-parameter reuse.

## Reuse Already Applied

### 1. Torchtune CE loss

`src/htext/heima_reuse.py` now tries to import and use:

```python
from torchtune.modules.loss import CEWithChunkedOutputLoss
```

For Hugging Face logits, it applies the same shifted-label convention before calling the Torchtune loss with one chunk.

Official reference:

- `torchtune_pkg/torchtune/torchtune/modules/loss/ce_chunked_output_loss.py:13-83`
- `heima/main_python/2-training...py:1580-1594`

### 2. Shifted thinking-token mask

`src/htext/heima_reuse.py::heima_shifted_thinking_mask` mirrors the official mask:

```python
mask = tokens[:, 1:] == thinking_token_id
mask = cat(mask, final_false_column)
```

Official reference:

- `heima/main_python/2-training...py:1503-1532`

The lightweight configs now expose:

```yaml
heima_shifted_hidden: true
```

so HText can choose the official Heima hidden position semantics.

### 3. Single code path for detach vs non-detach

`train_joint_main_l1` now exposes:

```yaml
detach_encoder_latent: true   # Heima-style no Loss1-to-A ablation
detach_encoder_latent: false  # ours: Loss1 returns to A
```

This reduces the H1/H2 difference to a config switch in the joint trainer path.

## Still Adapter-Based

### Data builder

Official data scripts operate on LLaVA-CoT records with `SUMMARY/CAPTION/REASONING` spans and image fields. HText synthetic records are text-only arithmetic. The current synthetic generator remains an adapter.

Future cleanup:

- Extract official span replacement into a reusable pure function.
- Map whole CoT to a single `REASONING` section.
- Reuse decoder prompt construction where possible.

### Decoder/projector replacement

Official Torchtune transformer supports:

```python
forward(..., thinking_token=..., thinking_token_mask=...)
```

and replaces token embeddings in `torchtune/modules/transformer.py:669-701`.

HText uses Hugging Face GPT-2, whose forward API does not accept `thinking_token` and `thinking_token_mask`. Therefore HText still uses an adapter based on `inputs_embeds` replacement. This is functionally equivalent but not direct code reuse.

Important official detail:

- `abstract_projection` is constructed in `transformer.py:404-416`.
- The call applying it to `thinking_token` is commented out in `transformer.py:671-674` and `692-694`.

### Training recipe

Official Heima recipe includes FSDP, LoRA, Torchtune checkpointers, decoder optimizers, image collation, and multiple decoder branches. HText still uses a small Hugging Face trainer.

Reason:

- GPT-2 does not fit Torchtune's Llama/Vision builder interfaces.
- The current objective is single-stage text-only Heima adaptation on the existing remote GPT-2 model.

Future cleanup:

- Create a config-compatible Heima-lite recipe wrapper.
- Keep one training loop with `detach_encoder_latent` as the only method switch.
- Avoid adding any new loss or benchmark until this path is stable.

## Current Completion Judgment

Completed:

- Heima-overlap computation graph is implemented.
- Torchtune CE loss reuse is attempted directly.
- Heima shifted thinking-mask logic is mirrored and configurable.
- Loss1 detach vs non-detach is now a config switch in one joint path.

Not completed:

- Full official recipe reuse.
- Official dataset builder reuse.
- Official Torchtune decoder replacement reuse with GPT-2.
- LoRA/FSDP/checkpointer reuse.
- Progressive/recovering stages.

The current code is best described as:

> Heima-aligned lightweight prototype with partial Torchtune/Heima logic reuse.

It is not yet:

> A direct adaptation of the official Heima training recipe.

