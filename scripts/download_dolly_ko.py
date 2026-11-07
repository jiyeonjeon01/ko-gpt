"""Phase 2 데이터: S4 dolly-ko 다운로드 (승인 2026-11-07, 15,212,862 바이트 파일 1개)

gated(자동 승인) 데이터셋이라 HF 로그인이 필요하다. 토큰은 사용자가 직접 넣는다:
    1) 브라우저에서 https://huggingface.co/datasets/nlpai-lab/databricks-dolly-15k-ko 접속 → 로그인 → 약관 동의
    2) 터미널에서 `uv run hf auth login` (토큰은 화면·로그·코드에 남기지 않는다)
    3) uv run python scripts/download_dolly_ko.py

이 스크립트는 토큰을 읽거나 출력하지 않는다. huggingface_hub가 저장된 로그인 정보를 스스로 쓴다.
revision과 파일 크기를 고정하고, 다르면 저장하지 않는다.
"""

import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import GatedRepoError, HfHubHTTPError

REPO = "nlpai-lab/databricks-dolly-15k-ko"
REVISION = "183b7a25c302efac83478ee5ef9ab14e105af16e"  # 2026-11-07 HF API로 확인
FILES = {"databricks-dolly-15k-ko.jsonl": 15_212_862, "README.md": 7_944}
OUT = Path("data/sft/S4_dolly_ko")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    manifest_path = OUT / "manifest.json"
    if manifest_path.exists():
        print(f"{manifest_path}가 이미 있다. 다시 받지 않는다")
        return 0
    OUT.mkdir(parents=True, exist_ok=True)
    local = []
    for name, size in FILES.items():
        try:
            path = Path(hf_hub_download(REPO, name, repo_type="dataset", revision=REVISION, local_dir=OUT))
        except GatedRepoError:
            print("접근 거부: 데이터셋 페이지에서 약관에 동의했는지, `uv run hf auth login`으로 로그인했는지 확인")
            return 2
        except HfHubHTTPError as e:
            print(f"다운로드 실패: {type(e).__name__} (상태 코드 {getattr(e.response, 'status_code', '?')})")
            return 2
        if path.stat().st_size != size:
            path.unlink()
            print(f"{name}: 크기 {path.stat().st_size if path.exists() else '?'} ≠ 예상 {size}. 저장하지 않음")
            return 3
        local.append({"path": str(path), "bytes": size, "sha256": sha256(path)})
    shutil.rmtree(OUT / ".cache", ignore_errors=True)  # hf_hub_download 메타데이터
    manifest = {"name": "S4_dolly_ko", "repo": REPO, "revision": REVISION, "license": "CC-BY-SA 3.0",
                "generator": "human (Dolly 15k) + DeepL 번역 (데이터셋 설명)", "group": "A",
                "downloaded": time.strftime("%Y-%m-%d %H:%M:%S"), "files": local}
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
