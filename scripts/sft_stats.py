"""A 그룹 SFT 표본의 규모 통계 (설계안 v0.2 4-2절: 규모와 비율은 토큰화 후 결정)

포함: S1 aya, S2 oasst2(ready trees, 각 트리에서 순위 1위 답변을 따라간 경로 하나), S10 dolly(영어)
제외: S4 dolly-ko (Phase 0 다운로드 제외 결정) → 한국어 통계는 불완전하다

    uv run --group data python scripts/sft_stats.py --tokenizer runs/phase0/tokenizers/v32768/tokenizer.json
"""

import argparse
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sft import render_conversation
from tokenizer import encode, load_tokenizer

SAMPLE = Path("data/sample")
MAX_LEN = 1024


def aya():
    for f in ("data/train-00000-of-00001.parquet", "data/test-00000-of-00001.parquet"):
        for r in pq.read_table(SAMPLE / "S1_aya" / f).to_pylist():
            lang = {"kor": "ko", "eng": "en"}.get(r["language_code"], "other")
            yield "S1_aya", lang, [{"role": "user", "content": r["inputs"]},
                                   {"role": "assistant", "content": r["targets"]}]


def oasst2():
    with gzip.open(SAMPLE / "S2_oasst2" / "2023-11-05_oasst2_ready.trees.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            tree = json.loads(line)
            node, messages = tree["prompt"], []
            while node is not None:
                messages.append({"role": "user" if node["role"] == "prompter" else "assistant",
                                 "content": node["text"]})
                # 사용자 후속 질문에는 rank가 없는 경우가 많다 (2단계 이후 rank 없음 확인). rank가 있으면 1위, 없으면 첫 답
                replies = [r for r in node.get("replies", []) if not r.get("deleted")]
                ranked = [r for r in replies if r.get("rank") is not None]
                node = min(ranked, key=lambda r: r["rank"]) if ranked else (replies[0] if replies else None)
            while messages and messages[-1]["role"] != "assistant":
                messages.pop()
            if messages:
                lang = tree["prompt"].get("lang", "other")
                yield "S2_oasst2", lang if lang in ("ko", "en") else "other", messages


def dolly():
    with open(SAMPLE / "S10_dolly" / "databricks-dolly-15k.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            user = r["instruction"] + (f"\n\n{r['context']}" if r["context"] else "")
            yield "S10_dolly", "en", [{"role": "user", "content": user},
                                      {"role": "assistant", "content": r["response"]}]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", type=Path, required=True)
    args = p.parse_args()
    tok = load_tokenizer(args.tokenizer)
    stat = defaultdict(lambda: defaultdict(int))
    for gen in (aya, oasst2, dolly):
        for source, lang, messages in gen():
            if lang == "other":
                stat[(source, "other")]["conversations"] += 1
                continue
            try:
                ids, mask = render_conversation(messages, lambda t: encode(tok, t))
            except ValueError:
                stat[(source, lang)]["invalid"] += 1
                continue
            s = stat[(source, lang)]
            if len(ids) > MAX_LEN:
                s["over_1024_excluded"] += 1
                continue
            s["conversations"] += 1
            s["turns"] += sum(m["role"] == "assistant" for m in messages)
            s["tokens"] += len(ids)
            s["loss_tokens"] += sum(mask)
            s["max_len"] = max(s["max_len"], len(ids))
    out = {}
    for (source, lang), s in sorted(stat.items()):
        s = dict(s)
        if s.get("conversations") and lang != "other":
            s["avg_len"] = round(s["tokens"] / s["conversations"], 1)
        out[f"{source}/{lang}"] = s
    totals = defaultdict(lambda: defaultdict(int))
    for key, s in out.items():
        lang = key.split("/")[1]
        for k in ("conversations", "tokens", "loss_tokens"):
            totals[lang][k] += s.get(k, 0)
    result = {"by_source": out, "by_language": {k: dict(v) for k, v in totals.items()},
              "note": "S4 dolly-ko는 Phase 0에서 제외되어 한국어 통계가 불완전하다. other = 한국어·영어 외 언어(사용 안 함)"}
    Path("runs/phase0/sft_stats.json").write_text(json.dumps(result, ensure_ascii=False, indent=1))
    print(json.dumps(result, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
