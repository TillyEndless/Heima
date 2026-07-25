# Heima HF Decoder Intervention Result

## What Was Actually Loaded

- Base decoder: `/data/zxl/models/meta-llama-Llama-3.1-8B-Instruct`
- Official Heima HF adapters: `/data/zxl/models/official_heima_hf/decoder/{summary,caption,reasoning}`
- Adapter type: LoRA, `r=16`, `alpha=32`, target modules include attention, MLP, and `lm_head`.
- The run manually merged the LoRA deltas into the HF Llama-3.1-8B model because the available Python envs do not have `peft` installed.

## Important Limitation

The public Heima HF snapshot does **not** contain:

- `abstract_projector_summary.pth`
- `abstract_projector_caption.pth`
- `abstract_projector_reasoning.pth`

Official decode code expects these projector files. Therefore this run is **not** a full continuous-latent correct/shuffle intervention. It is a decoder-side slot ablation using the official decoder adapters.

## Conditions

- `normal`: query prompt + official reserved thinking token id
- `zero_slot`: same sequence length and same token positions, but the thinking slot embedding is set to zero
- `deleted_slot`: query prompt without the thinking token slot

Thinking token ids:

- summary: `128013`, decoded by base tokenizer as `<|reserved_special_token_5|>`
- caption: `128014`, decoded by base tokenizer as `<|reserved_special_token_6|>`
- reasoning: `128015`, decoded by base tokenizer as `<|reserved_special_token_7|>`

## Metrics, 8 Samples

|section|normal NLL|zero-slot NLL|deleted-slot NLL|zero delta|deleted delta|
|---|---:|---:|---:|---:|---:|
|summary|1.6961|1.6961|1.4747|0.0000|-0.2214|
|caption|1.4089|1.4089|1.3485|0.0000|-0.0603|
|reasoning|1.2114|1.2114|1.3158|0.0000|0.1044|

Full generations and per-sample NLL are in `reports/heima_hf_decoder_slot_intervention/*.jsonl`.

## Example

Question:
`Which country is highlighted? Context: N/A Options: (A) Solomon Islands (B) Nauru (C) Vanuatu (D) Fiji`

Gold summary:
`<SUMMARY> I will determine the highlighted country by examining its location on the map and comparing it with the given options. ... </SUMMARY>`

Normal generation:
`The thinking progress for the summary of the given question is: To solve the problem, I will examine the image to identify the highlighted country. I will then compare it with the options provided to determine the correct answer.`

Zero-slot generation:
identical to normal.

Deleted-slot generation:
same content, with the reserved special token emitted at the beginning.

## Interpretation

The exact equality of `normal` and `zero_slot` across all three sections means the decoder-side reconstruction is not sensitive to the content of the reserved thinking-token embedding in this audit. The decoder can generate plausible reconstruction text from the query/prompt and teacher-forced prefix alone.

This supports the shortcut hypothesis for the decoder reconstruction protocol, but it does not settle whether the missing official continuous latent projector would make correct latent outperform shuffle latent. To test that, we still need the official `abstract_projector_*.pth` files or a checkpoint export that includes them.
