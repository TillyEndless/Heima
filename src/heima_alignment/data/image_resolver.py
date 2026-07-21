from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import json
from typing import Iterable, Sequence

IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png'}

@dataclass(frozen=True)
class ImageResolution:
    image_field: str
    resolved_path: str
    root_used: str
    sha256_16: str

class ImageResolver:
    def __init__(self, dataset_root: str | Path, extra_roots: Sequence[str | Path] = ()): 
        self.dataset_root = Path(dataset_root)
        official = self.dataset_root.parents[1] if len(self.dataset_root.parents) > 1 else self.dataset_root.parent
        roots = [
            self.dataset_root / 'images',
            self.dataset_root / 'image_files',
            official / 'images',
            official / 'image_files',
            Path('/data/zxl/official_heima/micro_subsets/chartqa_sqa_v1/image_files'),
            Path('/data/zxl/official_heima/micro_subsets/chartqa_sqa_available_images_v1'),
            Path('/data/zxl/hf'),
            Path('/root/.cache/huggingface'),
        ]
        roots.extend(Path(x) for x in extra_roots)
        self.roots = []
        seen = set()
        for root in roots:
            key = str(root)
            if key not in seen:
                self.roots.append(root); seen.add(key)

    def resolve(self, image_field: str) -> ImageResolution:
        p = Path(image_field)
        candidates = [p] if p.is_absolute() else [root / p for root in self.roots]
        for cand in candidates:
            if cand.exists() and cand.is_file() and cand.suffix.lower() in IMAGE_SUFFIXES:
                return ImageResolution(image_field, str(cand.resolve()), str(self._matched_root(cand)), file_sha256_16(cand))
        raise FileNotFoundError(f'cannot resolve image field {image_field!r}; tried {len(candidates)} roots')

    def _matched_root(self, path: Path) -> Path:
        for root in self.roots:
            try:
                path.resolve().relative_to(root.resolve())
                return root
            except Exception:
                continue
        return path.parent

    def statistics(self, image_fields: Iterable[str]) -> dict:
        total = 0; ok = 0; missing = []; by_root = {}
        for field in image_fields:
            total += 1
            try:
                res = self.resolve(field)
                ok += 1
                by_root[res.root_used] = by_root.get(res.root_used, 0) + 1
            except FileNotFoundError:
                if len(missing) < 20:
                    missing.append(field)
        return {'total': total, 'resolved': ok, 'missing': total-ok, 'availability': ok/max(total,1), 'by_root': by_root, 'missing_examples': missing}

def file_sha256_16(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()[:16]

def dump_resolution_stats(path: str | Path, stats: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(stats, indent=2, ensure_ascii=False, sort_keys=True) + '\n', encoding='utf-8')
