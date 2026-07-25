# Heima HF Decoder Slot Intervention

Loaded local HF Llama-3.1-8B base and merged the official Heima decoder LoRA adapters section by section.

Limitation: public Heima HF snapshot does not include `abstract_projector_summary.pth`, `abstract_projector_caption.pth`, or `abstract_projector_reasoning.pth`, so this is a decoder thinking-slot ablation, not full A-latent correct/shuffle intervention.

|section|condition|NLL|delta vs normal|BLEU1|ROUGE-L|
|---|---|---:|---:|---:|---:|
|summary|normal|1.6961|0.0000|0.3896|0.4991|
|summary|zero_slot|1.6961|0.0000|0.3896|0.4991|
|summary|deleted_slot|1.4747|-0.2214|0.3773|0.4991|
|caption|normal|1.4089|0.0000|0.4568|0.4868|
|caption|zero_slot|1.4089|0.0000|0.4568|0.4868|
|caption|deleted_slot|1.3485|-0.0603|0.4377|0.4868|
|reasoning|normal|1.2114|0.0000|0.3647|0.2851|
|reasoning|zero_slot|1.2114|0.0000|0.3647|0.2851|
|reasoning|deleted_slot|1.3158|0.1044|0.3502|0.2705|

## Interpretation

If `zero_slot` or `deleted_slot` is close to `normal`, the official decoder adapter can reconstruct largely from query/prompt and teacher-forced target prefix without requiring the reserved thinking-token embedding. This does not test the missing continuous latent projector.
