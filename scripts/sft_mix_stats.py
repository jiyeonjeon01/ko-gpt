"""Phase 2 SFT 후보 데이터의 출처·언어별 대화 수, 응답(Loss) 토큰 수, 평가 문항 유사 제외 수

    uv run python scripts/sft_mix_stats.py --out runs/phase2/prep/sft_mix_stats.json

- 토큰 수는 sft.render_conversation(설계안 4-3 템플릿)으로 센다. 1,024 토큰 초과 대화는 제외한다 (설계안 4-2).
- 평가 문항(eval/chat120/*.json, eval/prompts_ko20.json)과 비슷한 사용자 발화가 있는 대화는 제외하고 수를 센다.
- dolly-ko(data/sft/S4_dolly_ko)와 S11(data/sft/S11/*.jsonl)은 있으면 포함한다.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from eval_overlap import grams, normalize, scores, flagged  # noqa: E402
from s11_convert import load_eval_prompts  # noqa: E402
from sft import render_conversation  # noqa: E402
from sft_stats import aya, dolly, oasst2  # noqa: E402
from tokenizer import encode, load_tokenizer  # noqa: E402

MAX_LEN = 1024
DOLLY_KO = ROOT / "data/sft/S4_dolly_ko/databricks-dolly-15k-ko.jsonl"
S11_DIR = ROOT / "data/sft/S11"


def dolly_ko():
    if not DOLLY_KO.exists():
        return
    for line in open(DOLLY_KO, encoding="utf-8"):
        r = json.loads(line)
        ctx = r.get("context") or ""
        user = r["instruction"] + (f"\n\n{ctx}" if ctx else "")
        yield "S4_dolly_ko", "ko", [{"role": "user", "content": user}, {"role": "assistant", "content": r["response"]}]


def s11():
    for f in sorted(S11_DIR.glob("*.jsonl")) if S11_DIR.exists() else []:
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            yield "S11_direct", "ko", r["messages"]


class EvalGuard:
    def __init__(self, prompts: list[tuple[str, str]]):
        self.p = [(eid, grams(t), normalize(t)) for eid, t in prompts]

    def hit(self, messages: list[dict]) -> str | None:
        for m in messages:
            if m["role"] != "user":
                continue
            tg, tn = grams(m["content"]), normalize(m["content"])
            for eid, a, an in self.p:
                j, c = scores(a, tg)
                if an == tn or flagged(j, c, len(a), len(tg)):
                    return eid
        return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", type=Path, default=ROOT / "data/full/tokenizer/tokenizer.json")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    tok = load_tokenizer(a.tokenizer)
    guard = EvalGuard(load_eval_prompts(ROOT))
    stat = defaultdict(lambda: defaultdict(int))
    for gen in (aya, oasst2, dolly, dolly_ko, s11):
        for source, lang, messages in gen():
            if lang == "other":
                continue
            s = stat[f"{source}/{lang}"]
            if guard.hit(messages):
                s["excluded_eval_overlap"] += 1
                continue
            ids, mask = render_conversation(messages, lambda t: encode(tok, t))
            if len(ids) > MAX_LEN:
                s["excluded_over_1024"] += 1
                continue
            s["conversations"] += 1
            s["multi_turn"] += sum(m["role"] == "assistant" for m in messages) > 1
            s["tokens"] += len(ids)
            s["loss_tokens"] += sum(mask)
    out = {k: dict(v) for k, v in sorted(stat.items())}
    by_lang = defaultdict(lambda: defaultdict(int))
    for k, v in out.items():
        for f in ("conversations", "tokens", "loss_tokens", "multi_turn"):
            by_lang[k.split("/")[1]][f] += v.get(f, 0)
    result = {"by_source": out, "by_language": {k: dict(v) for k, v in by_lang.items()},
              "dolly_ko_present": DOLLY_KO.exists(), "s11_present": any(S11_DIR.glob("*.jsonl")) if S11_DIR.exists() else False}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
