#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import struct
import zlib
from pathlib import Path


LOCAL_HEADER = b"PK\x03\x04"


def load_needed(manifest: Path) -> set[str]:
    return {json.loads(line)["image"] for line in manifest.open(encoding="utf-8")}


def normalize(name: str) -> str:
    for prefix in ("image_files/", "images/"):
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip-part", type=Path, default=Path("/data/zxl/official_heima/datasets/LLaVA-CoT-100k/.download_parts/image.zip.part-aa"))
    parser.add_argument("--manifest", type=Path, default=Path("/data/zxl/official_heima/micro_subsets/chartqa_sqa_v1/image_manifest.jsonl"))
    parser.add_argument("--out-root", type=Path, default=Path("/data/zxl/official_heima/micro_subsets/chartqa_sqa_v1/image_files"))
    parser.add_argument("--report", type=Path, default=Path("/data/zxl/official_heima/micro_subsets/chartqa_sqa_v1/image_extraction_report.json"))
    args = parser.parse_args()

    needed = load_needed(args.manifest)
    extracted: list[str] = []
    seen: list[str] = []
    unsupported: list[dict] = []

    with args.zip_part.open("rb") as handle:
        while True:
            sig = handle.read(4)
            if not sig:
                break
            if sig != LOCAL_HEADER:
                chunk = sig + handle.read(1024 * 1024)
                idx = chunk.find(LOCAL_HEADER, 1)
                if idx < 0:
                    break
                handle.seek(handle.tell() - len(chunk) + idx)
                continue

            header = handle.read(26)
            if len(header) < 26:
                break
            _ver, flag, method, _mt, _md, _crc, csize, _usize, name_len, extra_len = struct.unpack("<HHHHHIIIHH", header)
            name = handle.read(name_len).decode("utf-8", errors="replace")
            handle.seek(extra_len, 1)
            rel = normalize(name)

            if flag & 0x08:
                unsupported.append({"name": name, "reason": "data_descriptor_unknown_size"})
                break
            if csize == 0xFFFFFFFF:
                unsupported.append({"name": name, "reason": "zip64_not_supported_in_fast_extractor"})
                break

            should_extract = rel in needed
            if should_extract:
                compressed = handle.read(csize)
                if method == 0:
                    data = compressed
                elif method == 8:
                    data = zlib.decompress(compressed, -zlib.MAX_WBITS)
                else:
                    unsupported.append({"name": name, "reason": f"unsupported_compression_{method}"})
                    continue
                write_bytes(args.out_root / rel, data)
                extracted.append(rel)
                seen.append(name)
            else:
                handle.seek(csize, 1)

            if len(extracted) == len(needed):
                break

    missing = sorted(needed - set(extracted))
    report = {
        "zip_part": str(args.zip_part),
        "manifest": str(args.manifest),
        "out_root": str(args.out_root),
        "needed": len(needed),
        "extracted": len(extracted),
        "missing": len(missing),
        "extracted_by_task": {
            "chartqa": sum(1 for x in extracted if x.startswith("chartqa/")),
            "sqa": sum(1 for x in extracted if x.startswith("sqa/")),
        },
        "missing_by_task": {
            "chartqa": sum(1 for x in missing if x.startswith("chartqa/")),
            "sqa": sum(1 for x in missing if x.startswith("sqa/")),
        },
        "first_missing": missing[:20],
        "unsupported": unsupported[:20],
    }
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
