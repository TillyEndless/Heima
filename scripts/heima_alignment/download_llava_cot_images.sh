#!/usr/bin/env bash
set -euo pipefail
DATA_ROOT=${DATA_ROOT:-/data/zxl/official_heima/datasets/LLaVA-CoT-100k}
REPO=${REPO:-Xkev/LLaVA-CoT-100k}
REVISION=${REVISION:-2b1db57dccb04ba8c33fc9b418a0f1d6cd328de3}
PYTHON_BIN=${PYTHON:-python3}
MIN_FREE_BYTES=$((200*1024*1024*1024))
FREE=$(df -B1 /data | awk 'NR==2{print $4}')
if [[ "$FREE" -lt "$MIN_FREE_BYTES" ]]; then
  echo "Need at least 200GB free before downloading image parts; have ${FREE} bytes" >&2
  exit 1
fi
mkdir -p "$DATA_ROOT/parts"
export HF_HOME=${HF_HOME:-/data/zxl/hf}
export HF_HUB_CACHE=${HF_HUB_CACHE:-/data/zxl/hf/hub}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-/data/zxl/hf/datasets}
export HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET:-1}
parts=(aa ab ac ad ae af ag ah ai aj ak al am an ao ap)
for suffix in "${parts[@]}"; do
  file="image.zip.part-${suffix}"
  echo "Downloading ${file} from ${REPO}@${REVISION}"
  hf download "$REPO" "$file" --repo-type dataset --revision "$REVISION" --local-dir "$DATA_ROOT/parts"
done
cat "$DATA_ROOT"/parts/image.zip.part-* > "$DATA_ROOT/image.zip"
python - <<PY
from pathlib import Path
import hashlib, zipfile, json
root=Path('$DATA_ROOT')
zip_path=root/'image.zip'
h=hashlib.sha256(zip_path.read_bytes()).hexdigest()
with zipfile.ZipFile(zip_path) as zf:
    bad=zf.testzip()
    names=[n for n in zf.namelist() if n.lower().endswith(('.jpg','.jpeg','.png'))]
    if bad: raise SystemExit(f'zip corrupt at {bad}')
(root/'image_zip_verification.json').write_text(json.dumps({'sha256':h,'image_count':len(names),'zip_size':zip_path.stat().st_size},indent=2)+'\n')
print('verified', h, len(names))
PY
