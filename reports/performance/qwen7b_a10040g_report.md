# Qwen-7B A100 40GB Performance Benchmark

## Scope
- This is a performance audit, not a method conclusion or formal training run.
- Physical GPU1 only via `CUDA_VISIBLE_DEVICES=1`; GPU0 was not used.
- No Model B, no Loss2, no projector, no hidden cosine/MSE training loss.
- Stage2 microbenchmark is valid A-only self-decode: first forward extracts non-detached z_THINK; second forward injects z via `inputs_embeds`; `L_total=L_main+0.1*L_self_decode`.

## Historical Timing Audit
- Existing Qwen Stage1 reports had completed steps/losses but no per-step wall timestamps.
- `qwen7b_stage1_smoke1k_200` therefore has historical status `runtime_missing`; Stage1-current was not repeated per skip policy.
- Checkpoint mtimes are recorded only as low-confidence endpoints, not pure throughput.

## Measured Throughput

| config | stage | status | sec/step | steps/hour | tokens/s | supervised tok/s | peak allocated GB | peak reserved GB |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| S1-current | stage1 | skipped_existing_measured_run_missing_runtime |  |  |  |  | 0.00 | 0.00 |
| S1-small | stage1 | complete | 0.291 | 12391 | 867 | 630 | 24.92 | 26.65 |
| S1-medium | stage1 | complete | 0.341 | 10560 | 1638 | 1395 | 25.17 | 27.64 |
| S1-large | stage1 | complete | 0.392 | 9188 | 1743 | 1483 | 25.25 | 28.02 |
| S2-small | stage2 | complete | 0.588 | 6124 | 816 | 436 | 25.07 | 27.93 |
| S2-current | stage2 | complete | 0.614 | 5861 | 1359 | 828 | 25.43 | 29.00 |
| S2-medium | stage2 | complete | 0.673 | 5345 | 1552 | 1098 | 25.47 | 29.95 |
| GEN-self-decode-50 | semantic_eval | complete | 5.745 | 627 | 0 | 0 | 16.37 | 16.51 |

## Stage2 Feasibility
- S2-small/current/medium all ran on one A100 40GB without OOM.
- S2-current: 0.614s/step, 5861 steps/hour, peak allocated 25.43GB.
- `stage2_grad_nonzero_finite=True` for all S2 configs, so self-decode loss was not detached and did produce finite nonzero gradient on first-forward trainable parameters.

## Stage1 vs Stage2
- S2-small / S1-small time ratio: 2.02x
- S2-current / S1-medium time ratio: 1.80x
- Memory ratio in summary: 1.01x

## ETA By Bucket

| basis | 1k h | 10k h | 100k h | 500k h | 900k h |
|---|---:|---:|---:|---:|---:|
| S1-small | 0.09 | 0.93 | 9.28 | 46.40 | 83.53 |
| S1-medium | 0.11 | 1.09 | 10.89 | 54.45 | 98.01 |
| S1-large | 0.13 | 1.25 | 12.52 | 62.58 | 112.65 |
| S2-small | 0.19 | 1.88 | 18.78 | 93.89 | 169.00 |
| S2-current | 0.20 | 1.96 | 19.62 | 98.11 | 176.59 |
| S2-medium | 0.22 | 2.15 | 21.51 | 107.57 | 193.62 |

## Two-Stage 60/40 ETA

| samples | realistic +15% hours | basis |
|---:|---:|---|
| 1000 | 0.14 | 60% S1-medium + 40% S2-current |
| 10000 | 1.44 | 60% S1-medium + 40% S2-current |
| 100000 | 14.38 | 60% S1-medium + 40% S2-current |
| 500000 | 71.91 | 60% S1-medium + 40% S2-current |
| 900000 | 129.44 | 60% S1-medium + 40% S2-current |

## Semantic Evaluation Benchmark
- Free generation measured: 5.745s/case, peak allocated 16.37GB.
- Frozen encoder semantic cosine time: not tested; no second 7B was kept resident on GPU.
- Estimated free-generation time: 50 cases 0.08h, 200 cases 0.32h, 1000 cases 1.60h.

## Missing Or Not Tested
- Model load time: missing in profiler output; benchmark step times exclude setup.
- Checkpoint save time: not tested in new profiler to avoid writing repeated 6GB adapters; historical checkpoint duration missing.
- GPU utilization: only point-sampled outside the measured loop; not reliable as per-step utilization.

## Hardware Decision
- Single A100 40GB can run capped Stage1 and capped A-only Stage2 microbenchmarks.
- For the tested capped buckets, memory is not at the absolute 40GB limit during measured steps (~25.5GB allocated, ~30GB reserved), but previous full adapter save/reload path nearly filled memory and reload OOMed.
- Main risk for formal runs is long oracle-K CoT length and large `modules_to_save=[embed_tokens,lm_head]` checkpoints, not this capped benchmark loop.
- 8x4090 should not be assumed linear. It may be worth renting only when moving to 100k+/500k formal runs, longer sequence buckets, multi-seed, or true Stage2 evaluation throughput.

