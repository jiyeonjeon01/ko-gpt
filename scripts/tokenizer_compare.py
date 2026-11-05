"""Tokenizer 후보(24k/32k/48k) 학습과 출처별 압축률 비교 (설계안 v0.2 5-4, 6-2절)

학습 샘플: 정제된 표본의 train 분할에서 고정 seed로 문서를 섞어 언어 비율(바이트 기준)을 맞춘다.
    한국어 50% (P1 25, P2 15, P3 10), 영어 30% (P4 15, P5 7.5, P6 7.5), 코드 15% (P8a), 수학 5% (P7)
    표본에 코드가 모자라면 있는 만큼만 쓰고 실제 비율을 기록한다.
측정: 각 출처의 val 분할 전체. 글자/토큰, 바이트/토큰, 한글 음절당 토큰 수(한국어 출처).

    uv run --group data python scripts/tokenizer_compare.py --total-mb 1000
"""

import argparse
import hashlib
import json
import random
import re
import resource
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tokenizer import decode, encode, save_tokenizer, train_tokenizer

CLEAN = Path("data/sample/clean")
OUT = Path("runs/phase0/tokenizers")
SHARES = {"P1_fineweb2_ko": 0.25, "P2_wikipedia_ko": 0.15, "P3_finepdfs_ko": 0.10,
          "P4_fineweb_edu": 0.15, "P5_cosmopedia_v2": 0.075, "P6_wikipedia_en": 0.075,
          "P8a_opc_algorithmic": 0.15, "P7_finemath_4plus": 0.05}
VOCABS = [24576, 32768, 49152]
SEED = 20261105
_HANGUL = re.compile(r"[가-힣]")

EXAMPLES = [
    "안녕하세요! 오늘 날씨가 정말 좋네요. 점심은 뭐 드셨어요?",
    "시험을 망쳐서 너무 속상해. 어떻게 해야 할지 모르겠어.",
    "광합성은 식물이 빛 에너지를 이용해 이산화탄소와 물로 포도당을 만드는 과정입니다.",
    "대한민국의 수도는 서울특별시이며, 인구는 약 940만 명이다.",
    "파이썬으로 1부터 10까지 더하는 코드를 알려줘.",
    "What is your name? I'm a small language model.",
]


def iter_split(name: str, split: str, chosen: set[int] | None = None):
    """출처의 한 분할을 문서 단위로 흘려 준다 (전체를 메모리에 올리지 않음). chosen이 있으면 그 순번만"""
    i = 0
    for batch in pq.ParquetFile(CLEAN / f"{name}.parquet").iter_batches(batch_size=2000, columns=["split", "text"]):
        for sp, text in zip(batch["split"].to_pylist(), batch["text"].to_pylist()):
            if sp != split:
                continue
            if chosen is None or i in chosen:
                yield text
            i += 1


def plan_training_sample(total_bytes: int) -> tuple[dict[str, set[int]], dict]:
    """출처마다 train 문서 순번을 고정 seed로 섞어 바이트 예산까지 고른다. 문서 본문은 들고 있지 않는다.

    BPE 학습은 단어 빈도만 세므로 문서 순서가 결과에 영향을 주지 않는다 (흘려 넣어도 같은 Tokenizer).
    """
    chosen, info = {}, {}
    h = hashlib.sha256()
    for name, share in SHARES.items():
        sizes = [len(t.encode("utf-8")) for t in iter_split(name, "train")]
        order = list(range(len(sizes)))
        random.Random(f"{SEED}:{name}").shuffle(order)
        budget, used, pick = int(total_bytes * share), 0, []
        for i in order:
            if used >= budget:
                break
            pick.append(i)
            used += sizes[i]
        chosen[name] = set(pick)
        for i in sorted(pick):
            h.update(f"{name}:{i}".encode())
        info[name] = {"target_bytes": budget, "bytes": used, "docs": len(pick),
                      "available_train_docs": len(sizes), "short": used < budget}
    total = sum(v["bytes"] for v in info.values())
    for v in info.values():
        v["actual_share"] = round(v["bytes"] / total, 4)
    return chosen, {"sources": info, "total_bytes": total, "docs": sum(v["docs"] for v in info.values()),
                    "sha256_of_doc_ids": h.hexdigest(), "split_seed_note": "문서 순번은 clean parquet의 train 분할 내 순서"}


def training_docs(chosen: dict[str, set[int]]):
    for name, idx in chosen.items():
        yield from iter_split(name, "train", idx)


def measure(tok) -> dict:
    out = {}
    for name in SHARES:
        texts = list(iter_split(name, "val"))
        chars = sum(len(t) for t in texts)
        nbytes = sum(len(t.encode("utf-8")) for t in texts)
        hangul = sum(len(_HANGUL.findall(t)) for t in texts)
        ids = tok.encode_batch(texts, add_special_tokens=False)
        tokens = sum(len(e.ids) for e in ids)
        assert all(decode(tok, e.ids) == t for e, t in zip(ids[:200], texts[:200])), f"{name} 왕복 실패"
        out[name] = {"docs": len(texts), "chars": chars, "tokens": tokens,
                     "chars_per_token": round(chars / tokens, 3), "bytes_per_token": round(nbytes / tokens, 3),
                     "hangul_share": round(hangul / max(chars, 1), 3)}
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--total-mb", type=int, default=1000)
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    chosen, sample_info = plan_training_sample(args.total_mb * 10**6)
    print(f"학습 샘플: {sample_info['total_bytes'] / 1e6:.1f}MB, {sample_info['docs']:,} 문서", flush=True)
    report = {"training_sample": sample_info, "vocabs": {}}
    for v in VOCABS:
        t0 = time.time()
        tok = train_tokenizer(training_docs(chosen), vocab_size=v)
        seconds = time.time() - t0
        sha = save_tokenizer(tok, OUT / f"v{v}" / "tokenizer.json",
                             {"phase": 0, "training_sample": sample_info["sha256_of_doc_ids"]})
        report["vocabs"][v] = {
            "train_seconds": round(seconds, 1), "sha256": sha,
            "max_rss_mib_so_far": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
            "val": measure(tok),
            "examples": [{"text": e, "tokens": len(encode(tok, e)),
                          "pieces": [decode(tok, [i]) for i in encode(tok, e)]} for e in EXAMPLES],
        }
        print(f"V={v}: {seconds:.0f}s, " + ", ".join(
            f"{k.split('_')[0]} {m['chars_per_token']}" for k, m in report["vocabs"][v]["val"].items()), flush=True)
        (Path("runs/phase0") / "tokenizer_compare.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
