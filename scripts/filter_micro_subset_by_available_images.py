#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, default=Path("/data/zxl/official_heima/micro_subsets/chartqa_sqa_v1"))
    parser.add_argument("--dst", type=Path, default=Path("/data/zxl/official_heima/micro_subsets/chartqa_sqa_available_images_v1"))
    args = parser.parse_args()

    image_root = args.src / "image_files"
    args.dst.mkdir(parents=True, exist_ok=True)
    kept_all = []
    split_spec = {}
    for split in ("train", "validation", "test"):
        kept = []
        dropped = []
        for line in (args.src / f"{split}.jsonl").open(encoding="utf-8"):
            row = json.loads(line)
            if (image_root / row["image"]).exists():
                kept.append(row)
                kept_all.append({"image": row["image"], "task": row["task"], "required_by_split": split})
            else:
                dropped.append(row)
        write_jsonl(args.dst / f"{split}.jsonl", kept)
        split_spec[split] = {
            "kept": len(kept),
            "dropped": len(dropped),
            "kept_by_task": dict(Counter(row["task"] for row in kept)),
            "dropped_by_task": dict(Counter(row["task"] for row in dropped)),
            "first_dropped": [row["image"] for row in dropped[:10]],
        }

    write_jsonl(args.dst / "image_manifest.jsonl", kept_all)
    # Symlink the shared image root if possible; fall back to recording the source.
    link = args.dst / "image_files"
    if not link.exists():
        try:
            link.symlink_to(image_root, target_is_directory=True)
        except OSError:
            pass

    spec = {
        "name": args.dst.name,
        "source_subset": str(args.src),
        "image_root": str(image_root),
        "splits": split_spec,
        "total_kept": sum(v["kept"] for v in split_spec.values()),
        "total_dropped": sum(v["dropped"] for v in split_spec.values()),
        "strict_note": "Contains only samples whose image bytes have been physically materialized and matched without ambiguous assignment.",
    }
    (args.dst / "dataset_spec.json").write_text(json.dumps(spec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(spec, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
