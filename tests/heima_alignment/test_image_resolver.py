from pathlib import Path
import pytest
from src.heima_alignment.data.image_resolver import ImageResolver


def test_image_resolver_deterministic(tmp_path: Path):
    root = tmp_path / 'dataset'; imgdir = root / 'image_files' / 'a'; imgdir.mkdir(parents=True)
    img = imgdir / 'x.png'; img.write_bytes(b'png')
    resolver = ImageResolver(root)
    a = resolver.resolve('a/x.png')
    b = resolver.resolve('a/x.png')
    assert a.resolved_path == b.resolved_path
    assert a.sha256_16 == b.sha256_16


def test_image_resolver_missing_raises(tmp_path: Path):
    resolver = ImageResolver(tmp_path)
    with pytest.raises(FileNotFoundError):
        resolver.resolve('missing.png')


def test_resolution_statistics(tmp_path: Path):
    root = tmp_path / 'dataset'; (root / 'images').mkdir(parents=True)
    (root / 'images' / 'ok.jpg').write_bytes(b'jpg')
    stats = ImageResolver(root).statistics(['ok.jpg', 'bad.jpg'])
    assert stats['resolved'] == 1
    assert stats['missing'] == 1
    assert stats['availability'] == 0.5
