#!/usr/bin/env bash
set -euo pipefail
ROOT="${REPO_ROOT:-/data/zxl/Heima-qwen7b-overnight-20260803}"
cd "$ROOT"
python3 - <<'PY2' > reports/overnight_structural_sweep/launch_manifest.json
import json,time
cfg=json.load(open('configs/overnight_structural_sweep/lanes.json'))
print(json.dumps({'time':time.time(),'lanes':cfg['lanes'],'status':'launching'},indent=2))
PY2
python3 - <<'PY2' | while IFS=: read -r lane gpu proto; do
import json
cfg=json.load(open('configs/overnight_structural_sweep/lanes.json'))
for l in cfg['lanes']:
 print(f"{l['lane']}:{l['gpu']}:{l['protocol']}")
PY2
  sess="overnight_lane_${lane}"
  tmux has-session -t "$sess" 2>/dev/null && continue
  tmux new-session -d -s "$sess" "bash $ROOT/scripts/overnight_structural_sweep/gpu_worker.sh $lane $gpu $proto"
done
