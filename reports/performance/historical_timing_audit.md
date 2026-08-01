# Historical Timing Audit

Historical Qwen smoke reports exist, but they did not log per-step wall timestamps. Checkpoint mtime is recorded only as low-confidence endpoint evidence, not pure training throughput.

## qwen7b_stage1_overfit32
- completed_steps: 30
- config: Stage1 max_q=256, max_latent=128, max_answer=128, LoRA BF16 GC microbatch=1
- runtime: missing
- timestamp_confidence: low
- checkpoint_saved_time: 2026-07-31T18:24:15.999248+00:00

## qwen7b_stage1_smoke1k_200
- completed_steps: 200
- config: Stage1 max_q=256, max_latent=128, max_answer=128, LoRA BF16 GC microbatch=1
- runtime: missing
- timestamp_confidence: low
- checkpoint_saved_time: 2026-07-31T18:27:26.100273+00:00

