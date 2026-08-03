#!/usr/bin/env bash
set -euo pipefail
export REPO_ROOT="${REPO_ROOT:-$(pwd)}" PY="${PY:-/data2/zhouxiaoling/envs/latentcot/bin/python}"
mkdir -p configs/progressive_suite logs/progressive_suite status/progressive_suite reports/progressive_suite
cat > configs/progressive_suite/gpu0_queue.csv <<EOF
protocol,scale,steps
P1_PROGRESSIVE_4_TYPED,D32,600
P1_PROGRESSIVE_4_TYPED,S2K,1200
P1_PROGRESSIVE_4_TYPED,S10K,3000
P1_PROGRESSIVE_4_TYPED,S50K,6000
P3_PROGRESSIVE_4_REPEATED,D32,600
P3_PROGRESSIVE_4_REPEATED,S2K,1200
P3_PROGRESSIVE_4_REPEATED,S10K,3000
FIXED_K64,D32,500
FIXED_K64,S2K,1200
FIXED_K64,S10K,3000
EOF
cat > configs/progressive_suite/gpu1_queue.csv <<EOF
protocol,scale,steps
P2_ONESHOT_4_TYPED,D32,600
P2_ONESHOT_4_TYPED,S2K,1200
P2_ONESHOT_4_TYPED,S10K,3000
P2_ONESHOT_4_TYPED,S50K,6000
P4_ONESHOT_4_REPEATED,D32,600
P4_ONESHOT_4_REPEATED,S2K,1200
P4_ONESHOT_4_REPEATED,S10K,3000
P5_PROGRESSIVE_4_TYPED_NO_LATENT_NTP,D32,600
P5_PROGRESSIVE_4_TYPED_NO_LATENT_NTP,S2K,1200
P5_PROGRESSIVE_4_TYPED_NO_LATENT_NTP,S10K,3000
FIXED_K32,D32,500
FIXED_K32,S2K,1200
FIXED_K32,S10K,3000
EOF
"$PY" scripts/progressive_suite/train_protocol.py audit
BR=$(git branch --show-current); SHA=$(git rev-parse HEAD)
cat > reports/progressive_suite/task_registry.json <<EOF
{"branch":"$BR","commit":"$SHA","gpu0_queue":"configs/progressive_suite/gpu0_queue.csv","gpu1_queue":"configs/progressive_suite/gpu1_queue.csv"}
EOF
tmux kill-session -t progressive_gpu0 2>/dev/null || true; tmux kill-session -t progressive_gpu1 2>/dev/null || true; tmux kill-session -t progressive_monitor 2>/dev/null || true
tmux new-session -d -s progressive_gpu0 "cd \"$REPO_ROOT\" && REPO_ROOT=\"$REPO_ROOT\" bash scripts/progressive_suite/gpu_worker.sh 0 configs/progressive_suite/gpu0_queue.csv"
tmux new-session -d -s progressive_gpu1 "cd \"$REPO_ROOT\" && REPO_ROOT=\"$REPO_ROOT\" bash scripts/progressive_suite/gpu_worker.sh 1 configs/progressive_suite/gpu1_queue.csv"
tmux new-session -d -s progressive_monitor "cd \"$REPO_ROOT\" && while true; do date -Is > status/progressive_suite/monitor_heartbeat.txt; nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits > status/progressive_suite/gpu_snapshot.txt; sleep 300; done"
cat > reports/progressive_suite/launch_manifest.json <<EOF
{"time":"$(date -Is)","branch":"$BR","commit":"$SHA","worktree":"$REPO_ROOT","sessions":["progressive_gpu0","progressive_gpu1","progressive_monitor"]}
EOF
tmux ls | grep progressive
