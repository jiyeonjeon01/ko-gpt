"""이 프로젝트가 직접 읽거나 쓴 파일의 페이지 캐시를 커널에 돌려준다 (메모리_원인조사_보고서 대책 B)

WSL2에서는 VM 안의 파일 캐시도 Windows 쪽 메모리(vmmemWSL commit)를 차지한다.
그래서 학습·생성·데이터 생성이 다 쓴 큰 파일은 캐시에서 뺀다.

- posix_fadvise(POSIX_FADV_DONTNEED)는 **그 파일의 캐시 페이지만** 대상으로 한다.
  - 다른 프로젝트·Docker의 캐시, VM 전체 캐시(drop_caches)는 건드리지 않는다.
  - 쓴 파일은 먼저 fsync한다. 더러운 페이지는 버려지지 않기 때문이다.
- **저장소 밖의 경로는 거부한다:** 심볼릭 링크를 풀어서 판단한다. `.venv` 안도 거부한다. 라이브러리는 다른 프로세스와 공유될 수 있기 때문이다.
- 실패해도 학습을 멈추지 않는다. 캐시 반환은 최적화일 뿐이므로 결과만 돌려준다.
"""

import ctypes
import ctypes.util
import mmap
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
_EXCLUDED = (ROOT / ".venv",)


def project_file(path: Path) -> Path:
    """저장소 안의 일반 파일이면 실제 경로, 아니면 ValueError"""
    real = Path(os.path.realpath(path))
    if not (real == ROOT or ROOT in real.parents):
        raise ValueError(f"{path}: 저장소 밖 파일은 캐시 반환 대상이 아니다")
    if any(real == e or e in real.parents for e in _EXCLUDED):
        raise ValueError(f"{path}: .venv 안 파일은 대상이 아니다")
    if not real.is_file():
        raise ValueError(f"{path}: 일반 파일이 아니다")
    return real


def drop_cache(*paths) -> dict[str, str]:
    """파일마다 fsync + POSIX_FADV_DONTNEED. 반환: {경로: "ok" | 거부·실패 이유}"""
    out = {}
    for p in paths:
        try:
            real = project_file(Path(p))
            fd = os.open(real, os.O_RDONLY)
            try:
                os.fsync(fd)
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(fd)
            out[str(p)] = "ok"
        except (OSError, ValueError) as e:
            out[str(p)] = f"{type(e).__name__}: {e}"
    return out


_libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
_libc.mmap.restype = ctypes.c_void_p
_libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
_libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
_libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
_PROT_READ, _MAP_SHARED, _MAP_FAILED = 1, 1, ctypes.c_void_p(-1).value


def resident_bytes(path: Path) -> int:
    """파일 중 페이지 캐시에 올라와 있는 바이트 수 (mmap + mincore, 페이지를 읽어 들이지 않음). 테스트·측정용"""
    size = os.path.getsize(path)
    if size == 0:
        return 0
    page = mmap.PAGESIZE
    n = (size + page - 1) // page
    fd = os.open(path, os.O_RDONLY)
    try:
        addr = _libc.mmap(None, size, _PROT_READ, _MAP_SHARED, fd, 0)
        if addr in (None, _MAP_FAILED):
            raise OSError(ctypes.get_errno(), "mmap 실패")
        try:
            vec = (ctypes.c_ubyte * n)()
            if _libc.mincore(addr, size, vec) != 0:
                raise OSError(ctypes.get_errno(), "mincore 실패")
        finally:
            _libc.munmap(addr, size)
    finally:
        os.close(fd)
    return sum(v & 1 for v in vec) * page


def drop_cache_under(*dirs) -> dict:
    """저장소 안 폴더들의 모든 일반 파일에 drop_cache. 읽은 파일 목록을 하나하나 넘기기 어려운 데이터 분석 스크립트용.
    캐시에 없는 파일은 아무 일도 없다. 반환: {"files": n, "ok": n, "failed": {경로: 이유}}"""
    files = [f for d in dirs for f in sorted(Path(d).rglob("*")) if f.is_file() and not f.is_symlink()]
    res = drop_cache(*files)
    failed = {k: v for k, v in res.items() if v != "ok"}
    return {"files": len(files), "ok": len(files) - len(failed), "failed": failed}
