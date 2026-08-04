# Stage2 Resume Launch Plan

```json
{
  "time": "2026-08-04T23:30:06+0800",
  "branch": "exp/stage2-overnight-20260804-002656",
  "commit": "9b70279280a46abd78e987e303cfff08bd8f619a",
  "manifest": "/data2/zhouxiaoling/latent_cot/Heima-overnight-qwen7b/data/processed_large/qwen7b_stage1/smoke1k.json",
  "samples": 1000,
  "queues": [
    "configs/stage2_overnight/resume_gpu1_queue.csv",
    "configs/stage2_overnight/resume_gpu0_queue.csv"
  ],
  "io_safety": {
    "save_best": false,
    "save_final": true,
    "checkpoint_save_lock": true,
    "eval_every": 200,
    "eval_n": 64,
    "gpu_poll_seconds": 120
  },
  "note": "Resume launches only S1K because S2K/S10K manifests were not found locally; GPU1 starts first, GPU0 waits until free."
}
```
