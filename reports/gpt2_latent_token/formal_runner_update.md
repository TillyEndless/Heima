# GPT2 Formal G2/G3 Runner Update

Checked G3 trainable parameter report from the smoke rerun:

- `embed_tokens_trainable`: true
- `lm_head_trainable`: true
- `embed_tokens_has_grad`: true
- `lm_head_has_grad`: true

Added `scripts/run_gpt2_latent_ablation_formal.py` for the formal GPT2-small run.

Formal run behavior:

- Groups: G2 and G3 only.
- Same seed: 42.
- Same deterministic dataset split: one shared `data_split.json`.
- Same optimizer: AdamW, lr `1e-5`.
- Same steps: 5000.
- Saves model weights only at step500, step1000, step2500, step5000.
- Does not save optimizer, scheduler, or scaler state.
- After each checkpoint, runs intervention metrics on eval and generation eval on a fixed eval prefix.
- Outputs compact final JSON with `shuffle_margin`, `latent_acc`, and `generation_acc` for G2/G3.

The formal runner remains independent from Qwen2.5-VL, Heima Stage2, Model B, projector, role embeddings, cumulative latent, and Loss2.
