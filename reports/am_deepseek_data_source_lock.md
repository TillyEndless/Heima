# AM-DeepSeek Data Source Lock Audit

## Why this exists

The recent Qwen-7B latent-reasoning runs were expected to use the pure-text AM-DeepSeek-R1-Distilled-1.4M derived dataset, but the active Qwen training scripts were still defaulting to an LLaVA-CoT-100k derived manifest. This caused decoder free generations from latent states to contain visual templates such as `The image shows ...`, because the textual CoT targets themselves were visual/LLaVA style.

## Fixed training defaults

The following Qwen scripts now default to the AM-only manifest path:

```text
/data2/zhouxiaoling/latent_cot/am_deepseek_runs/AM_DEEPSEEK_R1_DISTILLED_90K_TRAIN_5K_EVAL/manifest_train90k_eval5k_floor_nocap.json
```

Updated scripts:

- `scripts/qwen7b_stage1_r64_lr5_gate_sweep.py`
- `scripts/qwen7b_newdecoder_long.py`
- `scripts/qwen7b_newdecoder_long_lora64.py`
- `scripts/qwen7b_stage2_revised_ab_decoder.py` indirectly through `base.DEFAULT_MANIFEST`

Stage2 revised A+B default Stage1 checkpoint was also moved to an AM-named checkpoint path:

```text
/data2/zhouxiaoling/latent_cot/runs/AM_DEEPSEEK_QWEN7B_R64_LR5e-5_STAGE1_GATE_5K_10K_25K_50K_90K/checkpoints/stage1_step50000
```

## Hard guard

`load_split()` now fails closed unless the manifest is declared as AM-DeepSeek derived:

```text
a-m-team/AM-DeepSeek-R1-Distilled-1.4M
```

It rejects:

- manifest path/source/derived_from containing `LLaVA`, `LLaVA-CoT`, `official_heima`, `image-backed`, or `multimodal`;
- row-level `source`/`dataset` with those forbidden markers;
- a high sampled visual-CoT lexical rate, which catches accidental LLaVA-style target leakage.

Stage2 revised A+B also checks the Stage1 run `config.json`; legacy Stage1 checkpoints trained from LLaVA-CoT are rejected before model loading.

## Verification

`py_compile` passed for:

- `scripts/build_am_deepseek_latent_manifest.py`
- `scripts/qwen7b_stage1_r64_lr5_gate_sweep.py`
- `scripts/qwen7b_stage2_revised_ab_decoder.py`
- `scripts/qwen7b_newdecoder_long.py`
- `scripts/qwen7b_newdecoder_long_lora64.py`

The old LLaVA manifest is now rejected with:

```text
Refusing to load multimodal/LLaVA manifest for AM-only latent reasoning training
```

The old LLaVA Stage1@50K checkpoint is also rejected because its run config points to LLaVA-CoT.

## AM manifest status

Current available local AM data on this server:

```text
/weights2/zhouxiaoling/hf_cache/datasets--a-m-team--AM-DeepSeek-R1-Distilled-1.4M/snapshots/53531c06634904118a2dcd83961918c4d69d1cdf/am_0.9M_sample_1k.jsonl
```

A deterministic AM-only sample manifest has been built for smoke/protocol checks:

```text
/data2/zhouxiaoling/latent_cot/data/am_deepseek/manifests/am_deepseek_sample1k_train900_eval100_floor0p5_nocap.json
```

Manifest facts:

- train = 900
- eval = 100
- split hash = `fc49b1a2e309945f05b91fa2a8a3ad5c7b46e25fe5972499d7c7fb77dcfdd388`
- mean CoT token count = 1722.379
- mean latent count K = 860.945

This sample manifest is useful for code-path smoke tests only. It is not enough for the formal 90K/5K experiment.

The expected formal AM manifest is still:

```text
/data2/zhouxiaoling/latent_cot/am_deepseek_runs/AM_DEEPSEEK_R1_DISTILLED_90K_TRAIN_5K_EVAL/manifest_train90k_eval5k_floor_nocap.json
```

Online HF access from A100 timed out while probing the full dataset, so the formal 90K/5K AM split has not yet been materialized.

## What was not deleted

Existing LLaVA datasets, checkpoints, and reports were not physically deleted. They are historical artifacts and may be needed for audit/reproducibility. The fix removes them from the active Qwen latent-reasoning training path and makes accidental reuse fail loudly.
