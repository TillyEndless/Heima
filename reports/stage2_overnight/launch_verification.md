# Stage2 Overnight Launch Verification

- time: 2026-08-04T00:51:04+0800
- branch: `exp/stage2-overnight-20260804-002656`
- commit: `3f4ce2398a6eda320c059293a00d0e978ebd5053`
- worktree: `/data2/zhouxiaoling/latent_cot/Heima-qwen7b-stage2-overnight-20260804-002656`
- O3 step475: `/data2/zhouxiaoling/latent_cot/Heima-qwen7b-stage-validity-pilot/checkpoints/stage1_boundary_o3_dynamic/best_adapter`
- O3 step500: `/data2/zhouxiaoling/latent_cot/Heima-qwen7b-stage-validity-pilot/checkpoints/stage1_boundary_o3_dynamic/final_adapter`

## Tmux

```
stage2_cpu_audit: 1 windows (created Tue Aug  4 00:44:27 2026)
stage2_gpu0: 1 windows (created Tue Aug  4 00:44:27 2026)
stage2_gpu1: 1 windows (created Tue Aug  4 00:44:27 2026)
stage2_monitor: 1 windows (created Tue Aug  4 00:44:27 2026)
```

## GPU

```
0, NVIDIA H200 NVL, 27797, 143771, 0
1, NVIDIA H200 NVL, 28831, 143771, 0
```

## Processes

```
1618595 tmux new-session -d -s progressive_gpu0 cd "/data2/zhouxiaoling/latent_cot/Heima-qwen7b-progressive-suite-20260803-171939" && REPO_ROOT="/data2/zhouxiaoling/latent_cot/Heima-qwen7b-progressive-suite-20260803-171939" bash scripts/progressive_suite/gpu_worker.sh 0 configs/progressive_suite/gpu0_queue.csv
1898461 bash scripts/stage2_overnight/gpu_worker.sh 0 configs/stage2_overnight/gpu0_queue.csv
1898464 bash scripts/stage2_overnight/gpu_worker.sh 1 configs/stage2_overnight/gpu1_queue.csv
1898469 bash -c cd "/data2/zhouxiaoling/latent_cot/Heima-qwen7b-stage2-overnight-20260804-002656" && while true; do /data2/zhouxiaoling/envs/latentcot/bin/python scripts/stage2_overnight/stage2_runner.py gpt2-audit; sleep 3600; done
1898504 /data2/zhouxiaoling/envs/latentcot/bin/python scripts/stage2_overnight/stage2_runner.py train --job D32_A0_M0_475_CONT --mode M0 --adapter-kind o3_475 --steps 300 --lambda-decode 0.0
1898508 /data2/zhouxiaoling/envs/latentcot/bin/python scripts/stage2_overnight/stage2_runner.py train --job D32_B0_M1_JOINT_LAMBDA0p1 --mode M1 --adapter-kind none --steps 800 --lambda-decode 0.1
1904874 /bin/sh -c pgrep -af 'stage2_runner.py train|gpu_worker.sh|stage2_runner.py gpt2-audit|stage2_monitor' || true
```

## Current Jobs

- GPU0: `D32_A0_M0_475_CONT` running at/after step 20
- GPU1: `D32_B0_M1_JOINT_LAMBDA0p1` running at/after step 20
- CPU audit: `stage2_cpu_audit` started
- Monitor: `stage2_monitor` started

## Not Started / Deferred

- S2K/S10K jobs were not launched in this bootstrap because this worktree only has `data/manifests/stage1_debug32_no_truncation.json`; no real S2K/S10K split was available locally.
- Full semantic/intervention evaluators are scaffolded as reports but not complete scientific evaluators yet; current launched queue is D32 exploratory Stage2.
- Gradient smoke passed for finite A gradients, but decode-only split negative-control is marked deferred.
