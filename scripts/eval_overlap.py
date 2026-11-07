"""평가 문항과 학습·평가 데이터의 중복 검사 (문자 3-gram)

평가 문항은 짧아서(대부분 10~40자) 단어 13-gram 검사(prepare_sample.py)로는 잡히지 않는다.
정규화(NFC, 소문자, 공백·문장부호 제거) 뒤 문자 3-gram으로 비교한다.

  비슷함 = Jaccard ≥ 0.5  또는  (문항 3-gram의 80% 이상이 상대 텍스트에 있고, 상대가 문항보다 4배 이하로 김)

같은 기준(is_near_duplicate)을 SFT 데이터에서 평가 문항을 뺄 때 쓴다.

    uv run python scripts/eval_overlap.py eval/chat120/draft_v0.1.json --out runs/phase2/prep/eval_overlap.json
"""

import argparse
import gzip
import json
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

JACCARD_MIN, CONTAIN_MIN, LENGTH_RATIO_MAX = 0.5, 0.8, 4.0
_DROP = re.compile(r"[\s\W_]+", re.UNICODE)
SAMPLE = Path("data/sample")


def normalize(text: str) -> str:
    return _DROP.sub("", unicodedata.normalize("NFC", text).lower())


def grams(text: str, n: int = 3) -> set[str]:
    t = normalize(text)
    if len(t) < n:
        return {t} if t else set()
    return {t[i:i + n] for i in range(len(t) - n + 1)}


def scores(a: set[str], b: set[str]) -> tuple[float, float]:
    """(Jaccard, a의 b 포함률)"""
    if not a or not b:
        return 0.0, 0.0
    inter = len(a & b)
    return inter / len(a | b), inter / len(a)


def flagged(jac: float, contain: float, len_a: int, len_b: int) -> bool:
    return jac >= JACCARD_MIN or (contain >= CONTAIN_MIN and len_b <= LENGTH_RATIO_MAX * max(len_a, 1))


def is_near_duplicate(prompt: str, text: str) -> bool:
    a, b = grams(prompt), grams(text)
    jac, con = scores(a, b)
    return normalize(prompt) == normalize(text) or flagged(jac, con, len(a), len(b))


def user_text(item: dict) -> str:
    """평가 문항의 사용자 발화 전체 (다중 턴이면 이어 붙임)"""
    return "\n".join(m["content"] for m in item["messages"] if m["role"] == "user")


# ---- 비교 대상 ----


def corpus_sft():
    import pyarrow.parquet as pq
    for f in ("train", "test"):
        for r in pq.read_table(SAMPLE / "S1_aya" / "data" / f"{f}-00000-of-00001.parquet").to_pylist():
            if r["language_code"] in ("kor", "eng"):
                yield "S1_aya", r["inputs"]
    with gzip.open(SAMPLE / "S2_oasst2" / "2023-11-05_oasst2_ready.trees.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            tree = json.loads(line)
            if tree["prompt"].get("lang") not in ("ko", "en"):
                continue
            stack = [tree["prompt"]]
            while stack:
                node = stack.pop()
                if node["role"] == "prompter":
                    yield "S2_oasst2", node["text"]
                stack.extend(node.get("replies", []))
    for line in open(SAMPLE / "S10_dolly" / "databricks-dolly-15k.jsonl", encoding="utf-8"):
        yield "S10_dolly", json.loads(line)["instruction"]
    ko = Path("data/sft/S4_dolly_ko/databricks-dolly-15k-ko.jsonl")
    if ko.exists():
        for line in open(ko, encoding="utf-8"):
            yield "S4_dolly_ko", json.loads(line).get("instruction", "")


def corpus_eval():
    import pyarrow.parquet as pq
    for f in sorted((SAMPLE / "E_kobest").rglob("*.jsonl")):
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            for v in r.values():
                if isinstance(v, str):
                    yield f"KoBEST/{f.parent.name}", v
    for name, cols in (("E_hellaswag", ("ctx",)), ("E_arc_easy", ("question",)), ("E_piqa", ("goal",))):
        for f in sorted((SAMPLE / name).rglob("*.parquet")):
            for r in pq.read_table(f).to_pylist():
                for c in cols:
                    if r.get(c):
                        yield name, r[c]
    ko20 = json.loads(Path("eval/prompts_ko20.json").read_text(encoding="utf-8"))
    for p in ko20["prompts"]:
        yield "ko20", p["text"]


def scan(items: list[dict], corpus, top: int = 3) -> dict:
    """문항별 가장 비슷한 텍스트 top개와 기준 초과 여부"""
    pg = {it["id"]: grams(user_text(it)) for it in items}
    pnorm = {it["id"]: normalize(user_text(it)) for it in items}
    index = defaultdict(list)
    for iid, g in pg.items():
        for x in g:
            index[x].append(iid)
    best = {iid: [] for iid in pg}
    counts = defaultdict(int)
    for source, text in corpus:
        counts[source] += 1
        tg = grams(text)
        hit = defaultdict(int)
        for x in tg:
            for iid in index.get(x, ()):
                hit[iid] += 1
        for iid, inter in hit.items():
            a = pg[iid]
            jac, con = inter / len(a | tg), inter / len(a)
            if con < 0.3:
                continue
            row = {"source": source, "text": text[:200], "jaccard": round(jac, 3), "contain": round(con, 3),
                   "flag": pnorm[iid] == normalize(text) or flagged(jac, con, len(a), len(tg))}
            lst = best[iid]
            lst.append(row)
            lst.sort(key=lambda r: (r["flag"], r["jaccard"], r["contain"]), reverse=True)
            del lst[top:]
    return {"scanned": dict(counts), "matches": best}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("eval_set", type=Path)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    items = json.loads(a.eval_set.read_text(encoding="utf-8"))["items"]
    # 세트 안 중복
    internal = []
    for i, x in enumerate(items):
        for y in items[i + 1:]:
            if is_near_duplicate(user_text(x), user_text(y)):
                internal.append([x["id"], y["id"]])
    sft = scan(items, corpus_sft())
    ev = scan(items, corpus_eval())
    flagged_sft = {k: v for k, v in sft["matches"].items() if any(r["flag"] for r in v)}
    flagged_ev = {k: v for k, v in ev["matches"].items() if any(r["flag"] for r in v)}
    report = {"eval_set": str(a.eval_set), "rule": {"jaccard_min": JACCARD_MIN, "contain_min": CONTAIN_MIN,
                                                    "length_ratio_max": LENGTH_RATIO_MAX, "ngram": 3},
              "internal_duplicates": internal, "sft_scanned": sft["scanned"], "eval_scanned": ev["scanned"],
              "flagged_vs_sft": flagged_sft, "flagged_vs_eval": flagged_ev,
              "top_vs_sft": sft["matches"], "top_vs_eval": ev["matches"]}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"세트 안 중복 {len(internal)}쌍 | SFT 후보와 겹침 {len(flagged_sft)}문항 {sorted(flagged_sft)} "
          f"| 평가 데이터와 겹침 {len(flagged_ev)}문항 {sorted(flagged_ev)}")
    print("검사한 텍스트:", sft["scanned"], ev["scanned"])


if __name__ == "__main__":
    sys.exit(main())
