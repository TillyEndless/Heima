#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pyarrow.parquet as pq


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def parse_micro_question(text: str) -> tuple[str, tuple[str, ...]]:
    stem = text.split(" Context:", 1)[0]
    matches = re.findall(r"\(([A-Z])\)\s*([^()]+?)(?=\s*\([A-Z]\)|$)", text)
    choices = tuple(normalize(choice) for _label, choice in matches)
    return normalize(stem), choices


def write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_bytes(data)
    temp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset-root", type=Path, default=Path("/data/zxl/official_heima/micro_subsets/chartqa_sqa_v1"))
    parser.add_argument("--parquet", type=Path, default=Path("/data/zxl/official_heima/source_datasets/scienceqa/data/train-00000-of-00001-1028f23e353fbe3e.parquet"))
    parser.add_argument("--out-root", type=Path, default=Path("/data/zxl/official_heima/micro_subsets/chartqa_sqa_v1/image_files"))
    parser.add_argument("--report", type=Path, default=Path("/data/zxl/official_heima/micro_subsets/chartqa_sqa_v1/sqa_image_extraction_report.json"))
    args = parser.parse_args()

    needed = []
    for split in ("train", "validation", "test"):
        for line in (args.subset_root / f"{split}.jsonl").open(encoding="utf-8"):
            row = json.loads(line)
            if row["task"] == "sqa":
                stem, choices = parse_micro_question(row["question"])
                needed.append({"split": split, "image": row["image"], "stem": stem, "choices": choices, "id": row["id"]})

    table = pq.read_table(args.parquet, columns=["image", "question", "choices"])
    by_key: dict[tuple[str, tuple[str, ...]], list[bytes]] = {}
    for i in range(table.num_rows):
        image = table.column("image")[i].as_py()
        if not image or not image.get("bytes"):
            continue
        question = table.column("question")[i].as_py() or ""
        choices = tuple(normalize(x) for x in (table.column("choices")[i].as_py() or []))
        key = (normalize(question), choices)
        by_key.setdefault(key, []).append(image["bytes"])

    extracted = []
    missing = []
    ambiguous = []
    for item in needed:
        key = (item["stem"], item["choices"])
        matches = by_key.get(key, [])
        if len(matches) == 1:
            write_bytes(args.out_root / item["image"], matches[0])
            extracted.append(item["image"])
        elif not matches:
            missing.append(item)
        else:
            ambiguous.append({**item, "matches": len(matches)})

    report = {
        "parquet": str(args.parquet),
        "subset_root": str(args.subset_root),
        "needed_sqa": len(needed),
        "extracted": len(extracted),
        "missing": len(missing),
        "ambiguous": len(ambiguous),
        "first_missing": missing[:10],
        "first_ambiguous": ambiguous[:10],
    }
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
