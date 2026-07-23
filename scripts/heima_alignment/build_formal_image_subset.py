#!/usr/bin/env python3
"""Build the Model-A-only Loss1 formal image-backed split.

The script never downloads the full upstream image archive. It scans local image
roots, optionally extracts only requested files from an already present local
zip/zip-part, writes a download manifest for remaining images, and refuses to
mark the split ready unless every selected row has an accessible image and all
Heima section targets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import shutil
import struct
import zlib
from collections import Counter
from pathlib import Path
from statistics import mean

SECTIONS = ("summary", "caption", "reasoning")
TAG_RE = {
    "summary": re.compile(r"<SUMMARY>\s*(.*?)\s*</SUMMARY>", re.S),
    "caption": re.compile(r"<CAPTION>\s*(.*?)\s*</CAPTION>", re.S),
    "reasoning": re.compile(r"<REASONING>\s*(.*?)\s*</REASONING>", re.S),
    "answer": re.compile(r"<CONCLUSION>\s*(.*?)\s*</CONCLUSION>", re.S),
}
LOCAL_HEADER = b"PK\x03\x04"


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def extract_sections(text: str) -> dict[str, str] | None:
    out: dict[str, str] = {}
    for key, pattern in TAG_RE.items():
        match = pattern.search(text or "")
        if not match:
            return None
        value = clean(match.group(1))
        if not value:
            return None
        out[key] = value
    return out


def iter_rows(path: Path):
    with path.open(encoding="utf-8") as handle:
        for record_index, line in enumerate(handle):
            obj = json.loads(line)
            image = clean(obj.get("image", ""))
            conversations = obj.get("conversations", [])
            yielded = False
            for turn_idx in range(0, len(conversations) - 1, 2):
                human = conversations[turn_idx]
                assistant = conversations[turn_idx + 1]
                if human.get("from") != "human" or assistant.get("from") != "gpt":
                    continue
                yielded = True
                sections = extract_sections(assistant.get("value", ""))
                question = clean(human.get("value", ""))
                if not image or not question or sections is None:
                    yield None, {"image": bool(image), "question": bool(question), "sections": sections is not None}
                    continue
                row = {
                    "id": f"{obj.get(id, record_index)}::turn{turn_idx // 2}",
                    "source_record_id": obj.get("id"),
                    "source_record_index": record_index,
                    "turn_index": turn_idx // 2,
                    "task": image.split("/", 1)[0],
                    "image": image,
                    "question": question,
                    "summary": sections["summary"],
                    "caption": sections["caption"],
                    "reasoning": sections["reasoning"],
                    "answer": sections["answer"],
                    "whole_cot": clean("\n".join(sections[s] for s in SECTIONS)),
                }
                yield row, {"image": True, "question": True, "sections": True}
            if not yielded:
                yield None, {"image": bool(image), "question": False, "sections": False}


def normalize_zip_name(name: str) -> str:
    for prefix in ("image_files/", "images/", "./"):
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def build_file_index(roots: list[Path]) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file():
                rel = path.relative_to(root).as_posix()
                index.setdefault(rel, path)
    return index


def copy_or_link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    try:
        dst.symlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def extract_selected_from_zip_stream(zip_path: Path, needed: set[str], out_root: Path) -> set[str]:
    if not zip_path.exists() or zip_path.stat().st_size == 0 or not needed:
        return set()
    extracted: set[str] = set()
    with zip_path.open("rb") as handle:
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
            rel = normalize_zip_name(name)
            if flag & 0x08 or csize == 0xFFFFFFFF:
                break
            if rel in needed:
                compressed = handle.read(csize)
                if method == 0:
                    data = compressed
                elif method == 8:
                    data = zlib.decompress(compressed, -zlib.MAX_WBITS)
                else:
                    continue
                dst = out_root / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(data)
                extracted.add(rel)
                if len(extracted) == len(needed):
                    break
            else:
                handle.seek(csize, 1)
    return extracted


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def split_hash(train: list[dict], eval_rows: list[dict]) -> str:
    h = hashlib.sha256()
    for name, rows in (("train", train), ("eval", eval_rows)):
        h.update(name.encode())
        for row in rows:
            h.update(json.dumps(row, ensure_ascii=False, sort_keys=True).encode())
            h.update(b"\n")
    return h.hexdigest()


def token_stats(rows: list[dict]) -> dict[str, dict[str, float]]:
    stats = {}
    for field in ("question", "summary", "caption", "reasoning", "answer"):
        lengths = [len(str(row[field]).split()) for row in rows]
        stats[field] = {
            "min": min(lengths) if lengths else 0,
            "max": max(lengths) if lengths else 0,
            "mean": round(mean(lengths), 2) if lengths else 0.0,
        }
    return stats


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("/data/zxl/official_heima/datasets/LLaVA-CoT-100k/train.jsonl"))
    parser.add_argument("--out-dir", type=Path, default=Path("/data/zxl/runs/model_a_only_loss1_formal"))
    parser.add_argument("--train-size", type=int, default=5000)
    parser.add_argument("--eval-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-root", action="append", type=Path, default=[])
    parser.add_argument("--zip-source", action="append", type=Path, default=[])
    args = parser.parse_args()

    default_roots = [
        args.out_dir / "image_files",
        Path("/data/zxl/official_heima/datasets/LLaVA-CoT-100k/image_files"),
        Path("/data/zxl/official_heima/micro_subsets/chartqa_sqa_available_images_v1/image_files"),
        Path("/data/zxl/official_heima/micro_subsets/chartqa_sqa_v1/image_files"),
    ]
    roots = list(dict.fromkeys(args.image_root + default_roots))
    zip_sources = list(dict.fromkeys(args.zip_source + [
        Path("/data/zxl/official_heima/datasets/LLaVA-CoT-100k/image.zip"),
        Path("/data/zxl/official_heima/datasets/LLaVA-CoT-100k/.download_parts/image.zip.part-aa"),
    ]))

    total_jsonl = 0
    completeness = Counter()
    parsed_rows: list[dict] = []
    for row, present in iter_rows(args.input):
        total_jsonl += 1
        for key, ok in present.items():
            completeness[f"{key}_{'ok' if ok else 'missing'}"] += 1
        if row is not None:
            parsed_rows.append(row)

    rng = random.Random(args.seed)
    rng.shuffle(parsed_rows)
    image_index = build_file_index(roots)
    wanted = [row["image"] for row in parsed_rows]
    missing_before = [rel for rel in wanted if rel not in image_index]

    restored_from_zip = set()
    for zip_path in zip_sources:
        needed_now = set(missing_before) - restored_from_zip
        restored_from_zip |= extract_selected_from_zip_stream(zip_path, needed_now, args.out_dir / "image_files")
        if restored_from_zip:
            image_index = build_file_index(roots)
            missing_before = [rel for rel in wanted if rel not in image_index]

    selected: list[dict] = []
    missing_images: list[str] = []
    for row in parsed_rows:
        src = image_index.get(row["image"])
        if src is None:
            missing_images.append(row["image"])
            continue
        dst = args.out_dir / "image_files" / row["image"]
        copy_or_link(src, dst)
        selected.append(row)
        if len(selected) >= args.train_size + args.eval_size:
            break

    train = selected[: args.train_size]
    eval_rows = selected[args.train_size : args.train_size + args.eval_size]
    ready = len(train) == args.train_size and len(eval_rows) == args.eval_size
    split_id = split_hash(train, eval_rows)
    split_dir = args.out_dir / "formal_split"
    if ready:
        write_jsonl(split_dir / "train.jsonl", train)
        write_jsonl(split_dir / "validation.jsonl", eval_rows)

    data_split = {
        "status": "ready" if ready else "insufficient_accessible_images",
        "seed": args.seed,
        "train_size_target": args.train_size,
        "eval_size_target": args.eval_size,
        "selected_train_count": len(train),
        "selected_eval_count": len(eval_rows),
        "available_usable": len(selected),
        "image_root": str(args.out_dir / "image_files"),
        "dataset_path": str(split_dir),
        "source_jsonl": str(args.input),
        "split_hash": split_id,
        "train": train,
        "eval": eval_rows,
    }
    write_json(args.out_dir / "data_split.json", data_split)

    audit = {
        "status": data_split["status"],
        "total_jsonl_count": total_jsonl,
        "parsed_complete_count": len(parsed_rows),
        "selected_train_count": len(train),
        "selected_eval_count": len(eval_rows),
        "required_total": args.train_size + args.eval_size,
        "available_usable": len(selected),
        "image_access_rate": round(len(selected) / max(1, len(parsed_rows)), 6),
        "missing_images": len(missing_images),
        "first_missing_images": missing_images[:50],
        "section_completeness": dict(completeness),
        "task_counts_selected": dict(Counter(row["task"] for row in selected)),
        "token_statistics": token_stats(selected),
        "split_hash": split_id,
        "local_image_roots": [str(root) for root in roots],
        "zip_sources": [{"path": str(path), "exists": path.exists(), "size": path.stat().st_size if path.exists() else None} for path in zip_sources],
        "restored_from_zip_count": len(restored_from_zip),
        "notes": [
            "No full image.zip download is performed by this script.",
            "Images are symlinked into the formal run image root when possible.",
        ],
    }
    write_json(Path("reports/model_a_only_loss1_formal_dataset_audit.json"), audit)

    manifest_rows = [{"image": row["image"], "task": row["task"], "reason": "missing_local_image"} for row in parsed_rows if row["image"] not in image_index]
    write_jsonl(args.out_dir / "download_manifest.jsonl", manifest_rows)
    if not ready:
        failed = {
            "status": "failed",
            "reason": "insufficient_accessible_images",
            "required_train": args.train_size,
            "required_eval": args.eval_size,
            "available_usable": len(selected),
            "download_manifest": str(args.out_dir / "download_manifest.jsonl"),
            "dataset_audit": "reports/model_a_only_loss1_formal_dataset_audit.json",
        }
        write_json(Path("reports/formal_subset_failed.json"), failed)
        print(json.dumps(failed, indent=2, ensure_ascii=False, sort_keys=True))
        return 1

    print(json.dumps(audit, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
