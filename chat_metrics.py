"""대화 평가 자동 지표 (eval/chat120/rubric.md 1절 정의 그대로)

    uv run python chat_metrics.py eval/chat120/frozen_v1.json pre=runs/phase2/sft42m/gen_pre post=runs/phase2/sft42m/gen_post \
        --out runs/phase2/sft42m/metrics.json

지표: 정지율, 언어 일치율, 반복 붕괴율, distinct-2, 평균 길이, 정답 포함률 — 전체와 범주별
"""

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

HANGUL = re.compile(r"[가-힣]")
LATIN = re.compile(r"[A-Za-z]")


def detect_language(text: str) -> str:
    """rubric: 한글 음절 비율 ≥ 30% → ko, 라틴 문자 ≥ 50%이고 한글 < 10% → en, 나머지 other (비율은 글자 중)"""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return "other"
    ko = sum(1 for c in letters if HANGUL.match(c)) / len(letters)
    en = sum(1 for c in letters if LATIN.match(c)) / len(letters)
    if ko >= 0.3:
        return "ko"
    if en >= 0.5 and ko < 0.1:
        return "en"
    return "other"


def collapsed(ids: list[int], n: int = 4, limit: int = 3) -> bool:
    """같은 n-gram이 limit번 넘게 나오면 반복 붕괴"""
    grams = Counter(tuple(ids[i:i + n]) for i in range(len(ids) - n + 1))
    return any(c > limit for c in grams.values())


def distinct2(ids: list[int]) -> float | None:
    bigrams = [tuple(ids[i:i + 2]) for i in range(len(ids) - 1)]
    return len(set(bigrams)) / len(bigrams) if bigrams else None


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", s).lower()


def contains_answer(text: str, accept: list[str]) -> bool:
    t = _norm(text)
    return any(_norm(a) in t for a in accept)


def item_metrics(item: dict, gen: dict) -> dict:
    ids = gen["completion_ids"]
    m = {"stopped": gen["stop_reason"] == "im_end", "stop_reason": gen["stop_reason"],
         "language": detect_language(gen["completion"]), "collapsed": collapsed(ids), "distinct2": distinct2(ids),
         "length": len(ids)}
    m["language_match"] = m["language"] == item["expected_language"]
    ref = item["reference"]
    if ref["kind"] == "exact":
        m["answer_included"] = contains_answer(gen["completion"], ref["accept"])
    return m


def aggregate(rows: list[dict]) -> dict:
    n = len(rows)
    out = {"n": n,
           "stop_rate": sum(r["stopped"] for r in rows) / n,
           "language_match_rate": sum(r["language_match"] for r in rows) / n,
           "collapse_rate": sum(r["collapsed"] for r in rows) / n,
           "mean_length": sum(r["length"] for r in rows) / n,
           "stop_reasons": dict(Counter(r["stop_reason"] for r in rows))}
    d2 = [r["distinct2"] for r in rows if r["distinct2"] is not None]
    out["distinct2_mean"] = sum(d2) / len(d2) if d2 else None
    ex = [r["answer_included"] for r in rows if "answer_included" in r]
    out["answer_included_rate"] = sum(ex) / len(ex) if ex else None
    out["answer_included_n"] = len(ex)
    return out


def evaluate_run(items: list[dict], gens: dict[str, dict]) -> dict:
    missing = [it["id"] for it in items if it["id"] not in gens]
    if missing:
        raise ValueError(f"생성 결과에 없는 문항 {missing[:5]} 등 {len(missing)}개")
    per = {it["id"]: item_metrics(it, gens[it["id"]]) for it in items}
    by_cat = defaultdict(list)
    for it in items:
        by_cat[it["category"]].append(per[it["id"]])
    return {"overall": aggregate(list(per.values())), "by_category": {c: aggregate(v) for c, v in by_cat.items()},
            "items": per}


def load_generations(d: Path) -> tuple[dict, dict]:
    gens = {}
    for line in (d / "generations.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        gens[r["id"]] = r
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    return gens, meta


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("eval_set", type=Path)
    p.add_argument("runs", nargs="+", help="이름=생성 폴더")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    if a.out.exists():
        raise FileExistsError(f"{a.out}가 이미 있다")
    items = json.loads(a.eval_set.read_text(encoding="utf-8"))["items"]
    result = {"eval_set": str(a.eval_set), "runs": {}}
    for spec in a.runs:
        name, path = spec.split("=", 1)
        gens, meta = load_generations(Path(path))
        res = evaluate_run(items, gens)
        res["meta"] = meta
        result["runs"][name] = res
        o = res["overall"]
        print(f"{name}: 정지율 {o['stop_rate']:.1%} | 언어 일치 {o['language_match_rate']:.1%} | 반복 붕괴 {o['collapse_rate']:.1%}"
              f" | distinct-2 {o['distinct2_mean'] if o['distinct2_mean'] is None else round(o['distinct2_mean'], 3)}"
              f" | 평균 {o['mean_length']:.0f}토큰 | 정답 포함 {o['answer_included_rate']}")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
