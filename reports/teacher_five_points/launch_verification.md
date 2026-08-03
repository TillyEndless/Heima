# Launch Verification

Branch: exp/teacher-five-points-validation-20260803
Commit: 4458c8b feat: add teacher five-point latent cot validation suite

## 4090 server

Host: 4090 / 706124b042cd.

Started tmux workers:

- teacher5_gpu0: B0_DIRECT queue, current job S2K, waiting for GPU0.
- teacher5_gpu1: B1_EXPLICIT_COT queue, current job S2K, waiting for GPU1.
- teacher5_gpu2: P0_CURRENT_O3_DYNAMIC queue, current job D32, waiting for GPU2.
- teacher5_gpu3: P1_FIXED_K32 queue, current job D32, waiting for GPU3.
- teacher5_gpu4: P2_FIXED_K64 queue, current job D32, waiting for GPU4.
- teacher5_gpu5: P3_PROGRESSIVE_4_TYPED queue, current job D32, waiting for GPU5.
- teacher5_gpu6: P4_ONESHOT_4_TYPED queue, current job D32, waiting for GPU6.
- teacher5_gpu7: P5_PROGRESSIVE_4_REPEATED then P6 queue, current job D32, waiting for GPU7.

All 4090 GPUs were occupied by pre-existing jobs during verification. Workers are sleeping and did not preempt or kill anything.

Monitor: teacher_five_monitor.

## A100 server

A100 queue was not launched because ssh to 10.71.106.248:20024 failed. See missing_host_report.json.

## Training caveat

This launch is not yet scientifically valid full training. On the reachable 4090 server, the AM-DeepSeek-R1-Distilled-1.4M S2K/S10K/S50K source was not found. The worker will record DATA_MISSING for those scales after a GPU becomes free. Full protocol trainers beyond the audited D32/O3 code path still need implementation before large-scale jobs can produce results.
