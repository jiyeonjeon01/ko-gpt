"""dolly-ko 품질 검사: 행 수·필드, 언어(한글 비율·번역 안 된 영어), 중복, 빈 값, 길이, 범주, 원본 영어 Dolly와의 대응

    uv run python scripts/dolly_ko_check.py --out runs/phase2/prep/dolly_ko_check.json
"""

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from eval_overlap import normalize  # noqa: E402

KO = ROOT / "data/sft/S4_dolly_ko/databricks-dolly-15k-ko.jsonl"
EN = ROOT / "data/sample/S10_dolly/databricks-dolly-15k.jsonl"
HANGUL = re.compile(r"[가-힣]")
LATIN = re.compile(r"[A-Za-z]")


def lang_ratio(text: str) -> tuple[float, float]:
    """어절 단위 (한글이 들어 있는 어절 비율, 라틴 문자만 있는 어절 비율). 글자 단위는 영어 단어가 길어 한국어를 과소평가한다"""
    words = [w for w in text.split() if any(c.isalpha() for c in w)]
    if not words:
        return 0.0, 0.0
    ko = sum(1 for w in words if HANGUL.search(w))
    en = sum(1 for w in words if LATIN.search(w) and not HANGUL.search(w))
    return ko / len(words), en / len(words)


def classify(text: str) -> str:
    """어절 기준 ko: 한글 어절 ≥ 50%, en: 라틴 어절 ≥ 80%이고 한글 어절 < 5%, mixed: 나머지, empty"""
    if not text.strip():
        return "empty"
    h, l = lang_ratio(text)
    if h >= 0.5:
        return "ko"
    if l >= 0.8 and h < 0.05:
        return "en"
    return "mixed"


def check(rows: list[dict], en_rows: list[dict] | None = None, seed: int = 0) -> dict:
    res = {"rows": len(rows), "fields": Counter(tuple(sorted(r)) for r in rows).most_common(3)}
    res["fields"] = [[list(k), v] for k, v in res["fields"]]
    for f in ("instruction", "context", "response"):
        res[f"{f}_lang"] = dict(Counter(classify(r.get(f) or "") for r in rows))
    res["category"] = dict(Counter(r.get("category") for r in rows))
    key = lambda r: hashlib.sha1(normalize((r.get("instruction") or "") + "\x00" + (r.get("response") or "")).encode()).hexdigest()  # noqa: E731
    pairs = Counter(key(r) for r in rows)
    res["exact_duplicate_pairs"] = sum(c - 1 for c in pairs.values() if c > 1)
    ins = Counter(normalize(r.get("instruction") or "") for r in rows)
    res["duplicate_instructions"] = sum(c - 1 for c in ins.values() if c > 1)
    resp_len = sorted(len(r.get("response") or "") for r in rows)
    res["response_chars"] = {"median": resp_len[len(resp_len) // 2], "p95": resp_len[int(len(resp_len) * 0.95)],
                             "max": resp_len[-1], "under_5": sum(1 for x in resp_len if x < 5)}
    res["untranslated_examples"] = [r.get("instruction", "")[:80] for r in rows
                                    if classify(r.get("response") or "") == "en"][:5]
    res["leftover_markup"] = sum(1 for r in rows if re.search(r"\[\d+\]|&[a-z]+;|<[a-z/]+>", r.get("response") or ""))
    if en_rows is not None:
        res["english_rows"] = len(en_rows)
        res["category_matches_english"] = Counter(r.get("category") for r in rows) == Counter(r.get("category") for r in en_rows)
    rng = random.Random(seed)
    res["random_sample_for_review"] = [{"instruction": r.get("instruction", "")[:120], "response": r.get("response", "")[:200]}
                                       for r in rng.sample(rows, min(10, len(rows)))]
    return res


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if not KO.exists():
        print(f"{KO}가 없다. scripts/download_dolly_ko.py를 먼저 실행한다")
        return 1
    rows = [json.loads(x) for x in KO.read_text(encoding="utf-8").splitlines() if x.strip()]
    en = [json.loads(x) for x in EN.read_text(encoding="utf-8").splitlines() if x.strip()] if EN.exists() else None
    res = check(rows, en)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "random_sample_for_review"}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    finally:
        # 읽은 원본·쓴 결과가 WSL 파일 캐시로 남아 Windows 메모리를 차지하지 않게 뺀다 (이 저장소의 data/만, filecache.py)
        _r = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(_r))
        from filecache import drop_cache_under
        drop_cache_under(_r / "data" / "sample", _r / "data" / "sft")
