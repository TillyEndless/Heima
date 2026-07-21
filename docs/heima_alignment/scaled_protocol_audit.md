# Scaled Heima Protocol Audit

This branch aligns the small-model A+B Loss1 path to the official Heima protocol while explicitly allowing model-scale substitutions.

## Official Heima

- Encoder: Llama-3.2 Vision 11B / Xkev Llama-3.2V-11B-cot.
- Decoder: Llama-3.1 8B Instruct, with stage-specific summary/caption/reasoning decoder LoRA checkpoints.
- Data: Xkev/LLaVA-CoT-100k, prepared into Heima JSON files with summary/caption/reasoning replacement and pure-LLM decoder samples.
- Hidden extraction: predictor hidden. The official training script builds a mask with `batch["tokens"][:, 1:] == <THINKING>` and appends a false column, selecting `model.decoder.last_hidden_state` at position `p-1` for a thinking token at position `p`.
- Projector: official-compatible abstract projector is `Linear -> ReLU(inplace=True) -> Linear -> Dropout(0.0)`.
- Replacement: after token embedding lookup and before the transformer blocks.
- Loss: `CEWithChunkedOutputLoss`; decoder dataset uses `train_on_input: True`.

## Current Implementation

Previous small-model runs used Qwen2.5-VL-3B and Qwen2.5-0.5B with official section names, predictor hidden, projector/replacement helpers, and local/cumulative ablations. Some runs defaulted to micro data and text-only label ablations.

## Target Scaled Implementation

- `ALLOWED_MODEL_SCALE_DIFFERENCE`: keep Qwen2.5-VL-3B for Model A and Qwen2.5 small LLM for Model B.
- Full configuration defaults to official prepared LLaVA-CoT-100k paths. `--smoke` and `--dry-run` may use generated micro fixtures.
- Curriculum is explicit SFT, progressive summary, progressive caption, progressive reasoning, recover, then frozen-A stage-specific interpreter training.
- Main labels default to `main_label_mode=heima_ntp`: typed markers, unreplaced CoT text, and final answer participate in next-token prediction; prompt/image/question/padding are ignored.
- Heima baseline is named `heima_scaled_baseline`; `joint_detach` is not treated as the baseline because A continues changing under Main loss.
- Ours modes are separated into `ours_warm_b_fixed`, `ours_warm_b_joint`, `ours_cold_b_joint`, and `compute_matched_main_only`.

## Remaining Gaps

Full 100k training, VLMEvalKit benchmarks, and official metric reproduction are intentionally not launched in this implementation task. The new scripts create commandable stages and dry-run/smoke manifests only.
