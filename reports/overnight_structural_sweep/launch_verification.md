# Launch Verification

Branch: exp/overnight-stage1-structural-20260803
Commit: 2ca886e feat: add distributed structural latent cot sweep

## 4090 Server

Host: 4090 / 706124b042cd
Worktree: /data/zxl/Heima-qwen7b-overnight-20260803

Eight lane tmux sessions are running:

- overnight_lane_0: FIXED-K32 on GPU0, waiting because GPU busy.
- overnight_lane_1: FIXED-K64 on GPU1, waiting because GPU busy.
- overnight_lane_2: FIXED-K-P50 on GPU2, waiting because GPU busy.
- overnight_lane_3: PROGRESSIVE-4-TYPED on GPU3, waiting because GPU busy.
- overnight_lane_4: PROGRESSIVE-8-TYPED on GPU4, waiting because GPU busy.
- overnight_lane_5: PROGRESSIVE-4-REPEATED on GPU5, waiting because GPU busy.
- overnight_lane_6: PROGRESSIVE-4-TYPED-NO-LATENT-NTP on GPU6, waiting because GPU busy.
- overnight_lane_7: ONESHOT-4-TYPED on GPU7, waiting because GPU busy.

Monitor session: overnight_monitor.

Verification after 2 minutes showed all GPUs occupied by pre-existing processes and all workers logging gpu-busy sleep, so no other user process was killed or preempted.

## A100 Server

Host: 10.71.106.248:20024.

A100 queue was not launched because SSH repeatedly failed with: Connection timed out during banner exchange. This is recorded as a missing host instead of blocking the 4090 queue.

## Important Limitation

The distributed scheduler and worker queues are launched, but run_protocol.py currently only performs unit_smoke and then fail-fast records failed_protocol_trainer_not_implemented for non-smoke jobs. Full structural protocol trainers still need implementation before these queues can produce scientific training results.
