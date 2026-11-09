"""무결성 검사용 SHA-256 (sha256sum과 같은 출력 형식). 파일마다 해시 직후 그 파일의 캐시를 반환한다 (filecache.py)

    python3 scripts/hash_tree.py runs --exclude runs/phase2 > before.sha
    python3 scripts/hash_tree.py runs --exclude runs/phase2 | diff - before.sha

Checkpoint 전체를 읽는 무결성 검사가 WSL 파일 캐시를 수 GB 늘려 Windows 메모리를 압박한 일(2026-11-09 11:47)이 있어 만들었다.
"""

import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from filecache import drop_cache  # noqa: E402


def sha256_drop(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    drop_cache(path)
    return h.hexdigest()


def files(paths: list[Path], exclude: list[Path]) -> list[Path]:
    out = []
    ex = [e.resolve() for e in exclude]
    for p in paths:
        cand = [p] if p.is_file() else [f for f in p.rglob("*") if f.is_file()]
        for f in cand:
            r = f.resolve()
            if any(r == e or e in r.parents for e in ex):
                continue
            out.append(f)
    return sorted(out)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("paths", nargs="+", type=Path)
    p.add_argument("--exclude", nargs="*", type=Path, default=[])
    a = p.parse_args(argv)
    for f in files(a.paths, a.exclude):
        print(f"{sha256_drop(f)}  {f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
