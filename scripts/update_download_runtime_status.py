#!/usr/bin/env python3
from __future__ import annotations

import json
import pathlib
import subprocess
import time


def run(cmd: str, timeout: int = 20) -> dict:
    try:
        proc = subprocess.run(cmd, shell=True, text=True, capture_output=True, timeout=timeout)
        return {"returncode": proc.returncode, "stdout": proc.stdout.strip(), "stderr": proc.stderr.strip()}
    except Exception as exc:
        return {"returncode": None, "stdout": "", "stderr": repr(exc)}


def main() -> None:
    out = pathlib.Path("/data/zxl/Heima/reports/official_heima_reproduction")
    out.mkdir(parents=True, exist_ok=True)
    data_root = pathlib.Path("/data/zxl/official_heima/datasets/LLaVA-CoT-100k")
    dl_dir = data_root / ".download_parts" / ".cache" / "huggingface" / "download"
    incompletes = []
    if dl_dir.exists():
        for path in sorted(dl_dir.glob("*.incomplete")):
            stat = path.stat()
            incompletes.append({"path": str(path), "size_bytes": stat.st_size, "mtime": stat.st_mtime})
    state_path = data_root / "image_assembly_state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else None
    image_zip = data_root / "image.zip"
    status = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "running_part_aa_non_xet_download",
        "script_mode": "HF_HUB_DISABLE_XET=1; HF_XET_HIGH_PERFORMANCE absent",
        "hf_auth": run(
            "PATH=/data/zxl/conda_envs/nlp-final/bin:$PATH "
            "HF_HOME=/data/zxl/hf "
            "http_proxy=http://127.0.0.1:17890 https_proxy=http://127.0.0.1:17890 "
            "HTTP_PROXY=http://127.0.0.1:17890 HTTPS_PROXY=http://127.0.0.1:17890 "
            "hf auth whoami"
        ),
        "active_log": run(
            "find /data/zxl/official_heima/logs -type f -name 'download_*.log' "
            "-printf '%T@ %p\\n' | sort -nr | head -1 | cut -d' ' -f2-"
        ),
        "download_processes": run("pgrep -af 'download_official_heima|python -' || true"),
        "image_zip_size_bytes": image_zip.stat().st_size if image_zip.exists() else None,
        "assembly_state": state,
        "part_incomplete_files": incompletes,
        "disk": run("df -h /data && du -sh /data/zxl/official_heima /data/zxl/hf 2>/dev/null"),
        "note": (
            "image.zip remains 0 until the whole part is downloaded and appended. "
            "The active incomplete file is the expected hf_hub_download temporary file."
        ),
    }
    (out / "download_runtime_status.json").write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
    print(json.dumps(status, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
