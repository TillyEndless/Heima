# Progressive Latent Suite Launch Status

- time: 2026-08-03 17:46:06 +0800
- worktree: `/data2/zhouxiaoling/latent_cot/Heima-qwen7b-progressive-suite-20260803-171939`
- branch: `exp/progressive-latent-suite-20260803-171939`
- commit: `5a6c194c587c566246bae552ddc41a96dea291c9`
- tmux: `progressive_gpu0`, `progressive_gpu1`, `progressive_monitor`
- model: `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`
- GPUs:
```
0, NVIDIA H200 NVL, 103723, 143771, 97
1, NVIDIA H200 NVL, 60109, 143771, 0
```
- training processes:
```
1618637 /data2/zhouxiaoling/envs/latentcot/bin/python scripts/progressive_suite/train_protocol.py train --protocol P2_ONESHOT_4_TYPED --scale D32 --steps 600
1618638 /data2/zhouxiaoling/envs/latentcot/bin/python scripts/progressive_suite/train_protocol.py train --protocol P1_PROGRESSIVE_4_TYPED --scale D32 --steps 600
1621207 /bin/sh -c pgrep -af "train_protocol.py train" || true
```
## Data Audit

- source rows: 1000
- filtered rows under token cap: 191
- dropped too long: 809
- max sequence token cap: 1024
- D32: 32 train / 32 eval, split hash `8d397504a1771de696924cae2a17fbc6820b4ced700cd494374ae5e5354e3e55`
- S2K/S10K/S50K: DATA_INSUFFICIENT on current H200 local data cache

## Current Worker Status

### P1_PROGRESSIVE_4_TYPED_D32.json
```json
{
  "stage": "P1_PROGRESSIVE_4_TYPED_D32",
  "stage_id": 0,
  "status": "running",
  "step": 20,
  "time": 1785750281.3970578,
  "total_loss": 1.281184434890747
}
```
### P2_ONESHOT_4_TYPED_D32.json
```json
{
  "stage": "P2_ONESHOT_4_TYPED_D32",
  "stage_id": 4,
  "status": "running",
  "step": 20,
  "time": 1785750280.0495522,
  "total_loss": 13.376303672790527
}
```
### P4_ONESHOT_4_REPEATED_D32.json
```json
{
  "stage": "P4_ONESHOT_4_REPEATED_D32",
  "stage_id": 4,
  "status": "running",
  "step": 20,
  "time": 1785749825.7865582,
  "total_loss": 18.343473434448242
}
```
### worker_gpu0.json
```json
{
  "time": "2026-08-03T17:43:33+08:00",
  "gpu": 0,
  "protocol": "P1_PROGRESSIVE_4_TYPED",
  "scale": "D32",
  "state": "running"
}
```
### worker_gpu1.json
```json
{
  "time": "2026-08-03T17:43:33+08:00",
  "gpu": 1,
  "protocol": "P2_ONESHOT_4_TYPED",
  "scale": "D32",
  "state": "running"
}
```

## Notes

This is a D32 smoke/protocol-validation run only. Earlier failed P1/P2 attempts are retained in logs/reports for audit; the current launch is the one after atomic split writes and 1024-token cap. No historical checkpoints were modified.
