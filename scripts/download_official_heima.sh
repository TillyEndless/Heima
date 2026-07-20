#!/usr/bin/env bash
set -Eeuo pipefail

export PATH=/data/zxl/conda_envs/nlp-final/bin:$PATH

ROOT=/data/zxl/official_heima
DATA_ROOT="$ROOT/datasets/LLaVA-CoT-100k"
MODEL_ROOT="$ROOT/models"
HEIMA_ROOT="$ROOT/checkpoints/shawnricecake-Heima"
LOG_ROOT="$ROOT/logs"

export HF_HOME=/data/zxl/hf
export HF_HUB_CACHE=/data/zxl/hf/hub
export HF_DATASETS_CACHE=/data/zxl/hf/datasets
export TRANSFORMERS_CACHE=/data/zxl/hf/transformers
export HF_XET_CACHE=/data/zxl/hf/xet
export XDG_CACHE_HOME=/data/zxl/.cache
export TORCH_HOME=/data/zxl/.cache/torch
export TMPDIR=/data/zxl/tmp
export TEMP=/data/zxl/tmp
export TMP=/data/zxl/tmp

export HF_HUB_ETAG_TIMEOUT=60
export HF_HUB_DOWNLOAD_TIMEOUT=600
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false

mkdir -p \
  "$ROOT" \
  "$DATA_ROOT" \
  "$MODEL_ROOT" \
  "$HEIMA_ROOT" \
  "$LOG_ROOT" \
  "$HF_HOME" \
  "$HF_HUB_CACHE" \
  "$HF_DATASETS_CACHE" \
  "$HF_XET_CACHE" \
  "$XDG_CACHE_HOME" \
  "$TORCH_HOME" \
  "$TMPDIR"

exec > >(tee -a "$LOG_ROOT/download_$(date +%Y%m%d_%H%M%S).log") 2>&1

echo "============================================================"
echo "Official Heima resource download"
echo "Started: $(date -Is)"
echo "============================================================"

command -v hf >/dev/null
command -v unzip >/dev/null
hf auth whoami || echo "WARNING: hf auth whoami failed; continuing because repository resolution/downloads will verify access."

df -h /data
python -c 'import huggingface_hub; print("huggingface_hub:", huggingface_hub.__version__)'

echo
echo "[1/8] Resolving repository revisions..."

python - <<'PY' > "$ROOT/revisions.env.tmp"
from huggingface_hub import HfApi

api = HfApi()

repos = [
    ("VLM_REV", lambda: api.model_info("Xkev/Llama-3.2V-11B-cot").sha),
    ("DATA_REV", lambda: api.dataset_info("Xkev/LLaVA-CoT-100k").sha),
    ("HEIMA_REV", lambda: api.model_info("shawnricecake/Heima").sha),
    ("LLM_REV", lambda: api.model_info("meta-llama/Llama-3.1-8B-Instruct").sha),
]

for name, getter in repos:
    value = getter()
    if not value or len(value) != 40:
        raise RuntimeError(f"Invalid revision for {name}: {value!r}")
    print(f"{name}={value}")
PY

mv "$ROOT/revisions.env.tmp" "$ROOT/revisions.env"
# shellcheck disable=SC1090
source "$ROOT/revisions.env"
export VLM_REV DATA_REV HEIMA_REV LLM_REV

cat "$ROOT/revisions.env"

python - <<'PY'
import json
import os
from pathlib import Path

out = {
    "Xkev/Llama-3.2V-11B-cot": os.environ["VLM_REV"],
    "Xkev/LLaVA-CoT-100k": os.environ["DATA_REV"],
    "shawnricecake/Heima": os.environ["HEIMA_REV"],
    "meta-llama/Llama-3.1-8B-Instruct": os.environ["LLM_REV"],
}

path = Path("/data/zxl/official_heima/revisions.json")
path.write_text(json.dumps(out, indent=2) + "\n")
print(path)
PY

echo
echo "[2/8] Downloading dataset metadata and train.jsonl..."

hf download Xkev/LLaVA-CoT-100k \
  train.jsonl README.md .gitattributes \
  --repo-type dataset \
  --revision "$DATA_REV" \
  --local-dir "$DATA_ROOT"

ls -lh "$DATA_ROOT"/train.jsonl
df -h /data

echo
echo "[3/8] Downloading and assembling image.zip sequentially..."

python - <<'PY'
from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import tempfile
import time
from pathlib import Path

repo_id = "Xkev/LLaVA-CoT-100k"
revision = os.environ["DATA_REV"]

root = Path("/data/zxl/official_heima/datasets/LLaVA-CoT-100k")
parts_dir = root / ".download_parts"
archive = root / "image.zip"
state_path = root / "image_assembly_state.json"

parts_dir.mkdir(parents=True, exist_ok=True)
filenames = [f"image.zip.part-a{chr(ord('a') + i)}" for i in range(16)]


def save_state(state: dict) -> None:
    temp = state_path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(state, indent=2) + "\n")
    os.replace(temp, state_path)


def token() -> str | None:
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        value = os.environ.get(name)
        if value:
            return value.strip()
    token_path = Path(os.environ.get("HF_HOME", "/data/zxl/hf")) / "token"
    if token_path.exists():
        value = token_path.read_text().strip()
        if value:
            return value
    return None


def curl_download(filename: str, destination: Path) -> None:
    url = f"https://huggingface.co/datasets/{repo_id}/resolve/{revision}/{filename}"
    config_lines = [
        "location",
        "fail",
        "show-error",
        "continue-at = -",
        "connect-timeout = 30",
        "speed-limit = 1024",
        "speed-time = 300",
        f'output = "{destination}"',
        f'url = "{url}"',
    ]
    hf_token = token()
    if hf_token:
        config_lines.append(f'header = "Authorization: Bearer {hf_token}"')

    with tempfile.NamedTemporaryFile("w", delete=False, dir=parts_dir, prefix=".curl_", suffix=".conf") as handle:
        config_path = Path(handle.name)
        handle.write("\n".join(config_lines) + "\n")
    config_path.chmod(stat.S_IRUSR | stat.S_IWUSR)

    attempt = 0
    try:
        while True:
            attempt += 1
            before = destination.stat().st_size if destination.exists() else 0
            print(f"curl resume attempt {attempt}: {filename}, existing={before} bytes", flush=True)
            result = subprocess.run(["curl", "--config", str(config_path)], check=False)
            after = destination.stat().st_size if destination.exists() else 0
            print(f"curl attempt {attempt} exit={result.returncode}: {filename}, bytes={after}", flush=True)
            if result.returncode == 0:
                break
            if after < before:
                raise RuntimeError(f"curl truncated {filename}: before={before}, after={after}")
            time.sleep(20)
    finally:
        config_path.unlink(missing_ok=True)


if state_path.exists():
    state = json.loads(state_path.read_text())
else:
    if archive.exists() and archive.stat().st_size != 0:
        raise RuntimeError(f"{archive} exists but no assembly state was found. Inspect the archive before continuing.")
    state = {"revision": revision, "done": [], "zip_size": 0, "pending": None}
    save_state(state)

if state["revision"] != revision:
    raise RuntimeError(f"Revision mismatch: state={state['revision']} current={revision}")

archive.touch(exist_ok=True)

pending = state.get("pending")
if pending:
    start = int(pending["start_offset"])
    part_size = int(pending["part_size"])
    current = archive.stat().st_size
    filename = pending["filename"]
    if current == start:
        print(f"Previous append of {filename} did not begin; retrying.")
        state["pending"] = None
        save_state(state)
    elif current == start + part_size:
        print(f"Previous append of {filename} completed; finalizing state.")
        if filename not in state["done"]:
            state["done"].append(filename)
        state["zip_size"] = current
        state["pending"] = None
        save_state(state)
        existing = parts_dir / filename
        if existing.exists():
            existing.unlink()
    else:
        raise RuntimeError(f"Ambiguous interrupted archive state for {filename}: archive size={current}, expected {start} or {start + part_size}")

for index, filename in enumerate(filenames, start=1):
    if filename in state["done"]:
        print(f"[{index:02d}/16] already assembled: {filename}")
        continue

    current_size = archive.stat().st_size
    if current_size != int(state["zip_size"]):
        raise RuntimeError(f"Archive size differs from state: actual={current_size}, recorded={state['zip_size']}")

    print(f"[{index:02d}/16] downloading {filename}")
    local_path = parts_dir / filename
    curl_download(filename, local_path)
    part_size = local_path.stat().st_size
    print(f"[{index:02d}/16] appending {filename}: {part_size / 2**30:.2f} GiB")

    state["pending"] = {"filename": filename, "start_offset": current_size, "part_size": part_size}
    save_state(state)

    with local_path.open("rb") as src, archive.open("ab") as dst:
        shutil.copyfileobj(src, dst, length=64 * 1024 * 1024)
        dst.flush()
        os.fsync(dst.fileno())

    expected_size = current_size + part_size
    actual_size = archive.stat().st_size
    if actual_size != expected_size:
        raise RuntimeError(f"Append size mismatch for {filename}: actual={actual_size}, expected={expected_size}")

    state["done"].append(filename)
    state["zip_size"] = actual_size
    state["pending"] = None
    save_state(state)
    local_path.unlink()
    print(f"[{index:02d}/16] complete; archive now {actual_size / 2**30:.2f} GiB")

if set(state["done"]) != set(filenames):
    missing = sorted(set(filenames) - set(state["done"]))
    raise RuntimeError(f"Assembly incomplete; missing: {missing}")

print(f"Complete archive: {archive}")
print(f"Archive bytes: {archive.stat().st_size}")
PY

ls -lh "$DATA_ROOT/image.zip"
df -h /data

echo
echo "[4/8] Testing ZIP and checking extraction space..."

unzip -tq "$DATA_ROOT/image.zip"

python - <<'PY'
from pathlib import Path
import shutil
import zipfile

archive = Path("/data/zxl/official_heima/datasets/LLaVA-CoT-100k/image.zip")
root = archive.parent

with zipfile.ZipFile(archive) as zf:
    members = [x for x in zf.infolist() if not x.is_dir()]
    uncompressed = sum(x.file_size for x in members)
    compressed = archive.stat().st_size

free = shutil.disk_usage(root).free
margin = 30 * 2**30
needed = uncompressed + margin

print(f"Archive size:       {compressed / 2**30:.2f} GiB")
print(f"Expanded file size: {uncompressed / 2**30:.2f} GiB")
print(f"File count:         {len(members):,}")
print(f"Current free:       {free / 2**30:.2f} GiB")
print(f"Required + margin:  {needed / 2**30:.2f} GiB")

if free < needed:
    raise SystemExit("Insufficient space to extract safely. Move image.zip or free more space before continuing.")
PY

echo
echo "[5/8] Extracting image archive..."

unzip -q "$DATA_ROOT/image.zip" -d "$DATA_ROOT"

echo "Verifying every ZIP member exists with the expected size..."

python - <<'PY'
from pathlib import Path
import zipfile

root = Path("/data/zxl/official_heima/datasets/LLaVA-CoT-100k")
archive = root / "image.zip"
missing = []
wrong_size = []

with zipfile.ZipFile(archive) as zf:
    files = [x for x in zf.infolist() if not x.is_dir()]
    for info in files:
        destination = root / info.filename
        if not destination.is_file():
            missing.append(info.filename)
        elif destination.stat().st_size != info.file_size:
            wrong_size.append({"path": info.filename, "expected": info.file_size, "actual": destination.stat().st_size})

if missing or wrong_size:
    raise RuntimeError(f"Extraction verification failed: missing={len(missing)}, wrong_size={len(wrong_size)}")

print(f"Verified {len(files):,} extracted files.")
PY

rm -f "$DATA_ROOT/image.zip"

find "$DATA_ROOT" -type f ! -path '*/.cache/*' -printf '%P\t%s\n' | LC_ALL=C sort > "$DATA_ROOT/file_sizes_manifest.tsv"

echo "Dataset usage:"
du -sh "$DATA_ROOT"
df -h /data

python - <<'PY'
import shutil
free = shutil.disk_usage("/data").free
minimum = 80 * 2**30
print(f"Free before model downloads: {free / 2**30:.2f} GiB")
if free < minimum:
    raise SystemExit("Less than 80 GiB remains. Do not download the models yet; free additional disk space.")
PY

echo
echo "[6/8] Downloading Xkev/Llama-3.2V-11B-cot..."

VLM_DIR="$MODEL_ROOT/Xkev-Llama-3.2V-11B-cot"
hf download Xkev/Llama-3.2V-11B-cot --revision "$VLM_REV" --local-dir "$VLM_DIR"

du -sh "$VLM_DIR"
df -h /data

echo
echo "[7/8] Downloading Meta Llama-3.1-8B-Instruct HF files..."

LLM_DIR="$MODEL_ROOT/meta-llama-Llama-3.1-8B-Instruct"
hf download meta-llama/Llama-3.1-8B-Instruct \
  LICENSE USE_POLICY.md config.json generation_config.json \
  model-00001-of-00004.safetensors model-00002-of-00004.safetensors \
  model-00003-of-00004.safetensors model-00004-of-00004.safetensors \
  model.safetensors.index.json special_tokens_map.json tokenizer.json tokenizer_config.json \
  --revision "$LLM_REV" \
  --local-dir "$LLM_DIR"

du -sh "$LLM_DIR"
df -h /data

echo
echo "[8/8] Downloading shawnricecake/Heima checkpoints..."

hf download shawnricecake/Heima --revision "$HEIMA_REV" --local-dir "$HEIMA_ROOT"

du -sh "$HEIMA_ROOT"
df -h /data

echo
echo "Generating model/checkpoint SHA256 manifests..."

find "$VLM_DIR" -type f ! -path '*/.cache/*' -print0 | LC_ALL=C sort -z | xargs -0 sha256sum > "$ROOT/vlm_sha256.txt"
find "$LLM_DIR" -type f ! -path '*/.cache/*' -print0 | LC_ALL=C sort -z | xargs -0 sha256sum > "$ROOT/llm_sha256.txt"
find "$HEIMA_ROOT" -type f ! -path '*/.cache/*' -print0 | LC_ALL=C sort -z | xargs -0 sha256sum > "$ROOT/heima_checkpoints_sha256.txt"

echo
echo "============================================================"
echo "Download completed successfully: $(date -Is)"
echo "============================================================"

echo
echo "Revisions:"
cat "$ROOT/revisions.env"

echo
echo "Resource sizes:"
du -sh "$DATA_ROOT" "$VLM_DIR" "$LLM_DIR" "$HEIMA_ROOT"

echo
df -h /data
