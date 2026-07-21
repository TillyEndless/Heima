# Final Scaled Heima Alignment Report

## Completed Branches

1. `feat/heima-aligned-ab-loss1`
   - PR URL: https://github.com/TillyEndless/Heima/pull/new/feat/heima-aligned-ab-loss1
   - Protocol core commit: `e888b38e115cde4919e509f696a0f70eea97c365`
   - Tests: `13 passed`
   - Smoke: `/data/zxl/runs/heima_aligned_ab_loss1_protocol_smoke_20260721`

2. `feat/heima-aligned-ab-loss1-loss2`
   - PR URL: https://github.com/TillyEndless/Heima/pull/new/feat/heima-aligned-ab-loss1-loss2
   - Tests: `16 passed`
   - Smoke: `/data/zxl/runs/heima_aligned_ab_loss1_loss2_protocol_smoke_20260721`

3. `feat/heima-aligned-aonly-loss1`
   - PR URL: https://github.com/TillyEndless/Heima/pull/new/feat/heima-aligned-aonly-loss1
   - Tests: `16 passed`
   - Smoke: `/data/zxl/runs/heima_aligned_aonly_loss1_protocol_smoke_20260721`

## Fully Aligned Items

- Official typed tokens: `<THINKING_OF_SUMMARY>`, `<THINKING_OF_CAPTION>`, `<THINKING_OF_REASONING>`.
- Predictor hidden semantics: token position `p` uses hidden state `p-1`.
- Progressive stage order: explicit, summary, caption, reasoning, recover.
- Main label default: typed marker, unreplaced CoT, and answer are next-token supervised; prompt/question/image/padding are ignored.
- Stage-specific interpreter protocol and prompt shape.
- Official-compatible abstract projector: Linear, ReLU, Linear, Dropout(0).
- Embedding replacement after lookup, before transformer blocks.
- Paper and causal evaluator stages are separated.
- One-command wrappers with run-id, resume, dry-run, smoke, stage manifests, launch manifests, config hash, and no overwrite by default.

## Architecture-Driven Deviations

- Model A uses Qwen2.5-VL-3B rather than official 11B Llama vision model.
- Model B uses Qwen2.5 small LLM rather than Llama-3.1-8B-Instruct.
- HF model adapters are used for scaled models; official code uses Torchtune Llama builders.
- Full LoRA/FSDP parity is not executed in smoke mode.

## Not Run Because Of Compute/Data Scope

- Full LLaVA-CoT-100k explicit SFT.
- Full progressive/recover training.
- Full stage-specific interpreter training.
- MMStar, MMBench V1.1, MM-Vet, MathVista, AI2D, HallusionBench full evaluation.
- Official 4300 held-out decoder metric reproduction.

## Data/Benchmark Gaps

The branch defaults point to `/data/zxl/official_heima/datasets/LLaVA-CoT-100k`, but the task intentionally did not download or regenerate large data. If the prepared JSON files are missing, `pipeline.py` writes `data_missing_report.json` and stops outside smoke/dry-run.

## One-Command Formal Entrypoints

```bash
bash scripts/heima_aligned/run_all.sh \
  --config configs/heima_aligned/ab_loss1_qwen_vl3b.yaml \
  --mode heima_scaled_baseline \
  --run-id <RUN_ID>
```

```bash
bash scripts/heima_aligned/run_all.sh \
  --config configs/heima_aligned/ab_loss1_qwen_vl3b.yaml \
  --mode ours_warm_b_fixed \
  --run-id <RUN_ID>
```

```bash
bash scripts/heima_aligned/run_all.sh \
  --config configs/heima_aligned/ab_loss1_qwen_vl3b.yaml \
  --mode ours_warm_b_joint \
  --run-id <RUN_ID>
```

```bash
bash scripts/heima_aligned/run_all.sh \
  --config configs/heima_aligned/ab_loss1_loss2_qwen_vl3b.yaml \
  --mode ours_warm_b_joint_loss1_loss2 \
  --run-id <RUN_ID>
```

```bash
bash scripts/heima_aligned/run_all.sh \
  --config configs/heima_aligned/aonly_loss1_qwen_vl3b.yaml \
  --mode aonly_self_loss1 \
  --run-id <RUN_ID>
```

## Estimated Resources

- Smoke/dry-run: CPU only, negligible disk, under one minute.
- Scaled Qwen2.5-VL-3B + three Qwen2.5 small interpreters: expect one 48GB GPU per active stage for current implementation style, with roughly 10GiB per final checkpoint when saving full state dicts.
- Full official data and benchmark artifacts require separate disk approval before launch.

## Recommended Formal Order

1. Verify full Heima-prepared data and split hashes.
2. Run `heima_scaled_baseline` to completion.
3. Run `compute_matched_main_only`.
4. Run `ours_warm_b_fixed`.
5. Run `ours_warm_b_joint`.
6. Only if Loss1 has stable sample-specific signal, run Loss2 branch.
7. Run A-only self-decoder branch after A+B baseline is stable enough for fair comparison.
