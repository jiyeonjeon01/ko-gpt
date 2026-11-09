"""filecache: 이 저장소 파일의 캐시만 반환하고, 저장소 밖·.venv·심볼릭 링크로 빠져나가는 경로는 거부"""

import os
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import filecache as fc  # noqa: E402


@pytest.fixture
def proj_dir():
    with tempfile.TemporaryDirectory(dir=ROOT, prefix=".test_filecache_") as d:
        yield Path(d)


def test_drop_removes_cached_pages_of_project_file(proj_dir):
    f = proj_dir / "a.bin"
    f.write_bytes(os.urandom(4 << 20))  # 쓴 직후: 더러운 페이지 → fsync 후 반환되어야 함
    assert fc.resident_bytes(f) > 0
    assert fc.drop_cache(f) == {str(f): "ok"}
    assert fc.resident_bytes(f) == 0
    f.read_bytes()
    assert fc.resident_bytes(f) > 0
    assert fc.drop_cache_under(proj_dir) == {"files": 1, "ok": 1, "failed": {}}
    assert fc.resident_bytes(f) == 0


def test_refuses_outside_project(tmp_path, proj_dir):
    outside = tmp_path / "x.bin"
    outside.write_bytes(b"x" * 8192)
    before = fc.resident_bytes(outside)
    r = fc.drop_cache(outside, "/etc/hostname")
    assert all("저장소 밖" in v for v in r.values())
    assert fc.resident_bytes(outside) == before  # 건드리지 않음
    link = proj_dir / "link.bin"
    link.symlink_to(outside)  # 저장소 안 링크라도 실제 파일이 밖이면 거부
    assert "저장소 밖" in fc.drop_cache(link)[str(link)]
    assert fc.drop_cache_under(proj_dir)["files"] == 0  # 링크는 건너뜀


def test_refuses_venv_and_missing(proj_dir):
    venv_file = next((p for p in (ROOT / ".venv").rglob("*.py") if p.is_file() and not p.is_symlink()), None)
    if venv_file is not None:
        assert ".venv" in fc.drop_cache(venv_file)[str(venv_file)]
    assert "일반 파일" in fc.drop_cache(proj_dir / "none.bin")[str(proj_dir / "none.bin")]
