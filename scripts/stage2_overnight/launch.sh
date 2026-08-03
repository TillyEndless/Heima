#!/usr/bin/env bash
set -euo pipefail
export REPO_ROOT="${REPO_ROOT:-$(pwd)}" PY="${PY:-/data2/zhouxiaoling/envs/latentcot/bin/python}"
mkdir -p configs/stage2_overnight logs/stage2_overnight reports/stage2_overnight status/stage2_overnight
cat > configs/stage2_overnight/gpu0_queue.csv <<EOF
job,mode,adapter,steps,lambda
D32_A0_M0_475_CONT,M0,o3_475,300,0.0
D32_A1_M2_475_LAMBDA0p1,M2,o3_475,300,0.1
D32_D_M2_475_LAMBDA0p03,M2,o3_475,300,0.03
EOF
cat > configs/stage2_overnight/gpu1_queue.csv <<EOF
job,mode,adapter,steps,lambda
D32_B0_M1_JOINT_LAMBDA0p1,M1,none,800,0.1
D32_C0_M0_500_CONT,M0,o3_500,300,0.0
D32_C1_M2_500_LAMBDA0p1,M2,o3_500,300,0.1
D32_D_M2_475_LAMBDA0p3,M2,o3_475,300,0.3
EOF
"$PY" scripts/stage2_overnight/stage2_runner.py inventory
BR=$(git branch --show-current); SHA=$(git rev-parse HEAD)
cat > reports/stage2_overnight/task_registry.json <<EOF
{"branch":"$BR","commit":"$SHA","gpu0_queue":"configs/stage2_overnight/gpu0_queue.csv","gpu1_queue":"configs/stage2_overnight/gpu1_queue.csv","note":"D32 Stage2 queue launched; S2K/S10K not launched because current worktree only has debug32 manifest locally."}
EOF
tmux kill-session -t stage2_gpu0 2>/dev/null || true; tmux kill-session -t stage2_gpu1 2>/dev/null || true; tmux kill-session -t stage2_cpu_audit 2>/dev/null || true; tmux kill-session -t stage2_monitor 2>/dev/null || true
tmux new-session -d -s stage2_gpu0 "cd \"$REPO_ROOT\" && REPO_ROOT=\"$REPO_ROOT\" bash scripts/stage2_overnight/gpu_worker.sh 0 configs/stage2_overnight/gpu0_queue.csv"
tmux new-session -d -s stage2_gpu1 "cd \"$REPO_ROOT\" && REPO_ROOT=\"$REPO_ROOT\" bash scripts/stage2_overnight/gpu_worker.sh 1 configs/stage2_overnight/gpu1_queue.csv"
tmux new-session -d -s stage2_cpu_audit "cd \"$REPO_ROOT\" && while true; do $PY scripts/stage2_overnight/stage2_runner.py gpt2-audit; sleep 3600; done"
tmux new-session -d -s stage2_monitor "cd \"$REPO_ROOT\" && while true; do date -Is > status/stage2_overnight/monitor_heartbeat.txt; nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits > status/stage2_overnight/gpu_snapshot.txt; $PY scripts/stage2_overnight/stage2_runner.py aggregate >/dev/null 2>&1 || true; sleep 300; done"
cat > reports/stage2_overnight/launch_manifest.json <<EOF
{"time":"$(date -Is)","branch":"$BR","commit":"$SHA","worktree":"$REPO_ROOT","sessions":["stage2_gpu0","stage2_gpu1","stage2_cpu_audit","stage2_monitor"]}
EOF
tmux ls | grep stage2
