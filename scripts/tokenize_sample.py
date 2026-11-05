"""정제된 표본을 고정 Tokenizer로 토큰 샤드(uint16)로 만들고, 고유 토큰·반복 노출 예측을 계산한다 (설계안 v0.2 7-3절)

- 문서마다 <|endoftext|> + 토큰. 출처·분할마다 .bin 하나
- data.json: 학습 루프용 데이터 설정 (가중치 = 설계안 4-1 목표 비율, 반복 상한 = 4-1)
- 전체 출처 고유 토큰 추정 = 표본 토큰 × (전체 크기 / 표본 크기). 크기 기준은 출처마다 표에 적었다

    uv run --group data python scripts/tokenize_sample.py --tokenizer runs/phase0/tokenizers/v32768/tokenizer.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from text import EOT_ID
from tokenizer import file_sha256, load_tokenizer

CLEAN = Path("data/sample/clean")
TOKENS = Path("data/sample/tokens")

# 설계안 v0.2 4-1: 목표 토큰(B), 반복 상한. P2는 "고유 토큰 × 최대 2회"라 목표를 고유 토큰으로 정한다(None)
TARGETS = {
    "P1_fineweb2_ko": (0.85, 1.0), "P2_wikipedia_ko": (None, 2.0), "P3_finepdfs_ko": (0.25, 1.0),
    "P4_fineweb_edu": (0.60, 1.0), "P5_cosmopedia_v2": (0.20, 1.0), "P6_wikipedia_en": (0.20, 1.0),
    "P7_finemath_4plus": (0.12, 1.0), "P8a_opc_algorithmic": (0.30, 2.0),
}
# 전체 출처 크기 / 표본 크기 (외삽 배율의 근거). 행 수는 2026-11-04 datasets-server, 파일 크기는 HF API 확인값
#   표본 행 수는 download manifest, 정제 전 문서 수(docs_in)를 쓴다
FULL_ROWS = {
    "P1_fineweb2_ko": 60_874_355,  # kor_Hang train 전체
    "P2_wikipedia_ko": 647_897,  # 전체를 받았으므로 배율 1
    "P3_finepdfs_ko": 1_092_492,
    "P4_fineweb_edu": None,  # sample-10BT의 일부만 쓰므로 외삽하지 않음 (원본이 1T 이상이라 부족할 일 없음)
    "P5_cosmopedia_v2": 39_134_000,
    "P6_wikipedia_en": 6_407_814,
    "P7_finemath_4plus": 6_699_493,
    "P8a_opc_algorithmic": 5_322_920,
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--block-size", type=int, default=1024)
    args = p.parse_args()
    tok = load_tokenizer(args.tokenizer)
    if tok.get_vocab_size() > 65536:
        raise ValueError("uint16 범위를 넘는 Vocabulary")
    TOKENS.mkdir(parents=True, exist_ok=True)
    stats = json.loads(Path("runs/phase0/clean_stats.json").read_text())["sources"]
    data_sources, report = [], {}
    total_target = sum(t for t, _ in TARGETS.values() if t)

    for name, (target_b, max_repeat) in TARGETS.items():
        pf = pq.ParquetFile(CLEAN / f"{name}.parquet")
        counts = {}
        shards = {}
        for split in ("train", "val", "test"):
            path = TOKENS / f"{name}_{split}.bin"
            if path.exists():
                raise FileExistsError(f"{path}가 이미 있다")
            n_tok = n_docs = 0
            with open(path, "wb") as f:
                for batch in pf.iter_batches(batch_size=2000, columns=["split", "text"]):
                    texts = [t for s, t in zip(batch["split"].to_pylist(), batch["text"].to_pylist()) if s == split]
                    for enc in tok.encode_batch(texts, add_special_tokens=False):
                        arr = np.asarray([EOT_ID] + enc.ids, dtype=np.uint16)
                        arr.tofile(f)
                        n_tok += len(arr)
                    n_docs += len(texts)
            counts[split] = {"tokens": n_tok, "docs": n_docs}
            shards[split] = [{"path": str(path), "tokens": n_tok, "sha256": file_sha256(path)}]
        s = stats[name]
        train_tok = counts["train"]["tokens"] + counts["val"]["tokens"] + counts["test"]["tokens"]
        scale = FULL_ROWS[name] / s["docs_in"] if FULL_ROWS[name] else None
        est_unique = train_tok * scale if scale else None
        target = (target_b * 1e9) if target_b else (est_unique * max_repeat if est_unique else None)
        report[name] = {
            "sample_tokens": train_tok, "sample_docs_kept": s["docs_kept"], "keep_rate": round(s["keep_rate"], 4),
            "chars_per_token": round(s["chars_kept"] / train_tok, 3),
            "extrapolation_scale": round(scale, 2) if scale else None,
            "est_full_unique_tokens_B": round(est_unique / 1e9, 3) if est_unique else None,
            "target_tokens_B": round(target / 1e9, 3) if target else None,
            "max_repeat": max_repeat,
            "planned_repeat": round(target / est_unique, 3) if (target and est_unique) else None,
            "splits": counts,
        }
        # 표본 smoke용 가중치: 목표 비율. P2는 4-1 계산값(고유 × 2)을 쓴다
        weight = target_b if target_b else (target / 1e9 if target else 0.2)
        data_sources.append({"name": name, "weight": weight, "max_repeat": max_repeat,
                             "train": shards["train"], "val": shards["val"]})
        print(f"{name}: 표본 {train_tok / 1e6:,.1f}M 토큰, 전체 추정 "
              f"{report[name]['est_full_unique_tokens_B']}B, 계획 반복 {report[name]['planned_repeat']}", flush=True)

    data = {"tokenizer": str(args.tokenizer), "tokenizer_sha256": file_sha256(args.tokenizer),
            "block_size": args.block_size, "sources": data_sources,
            "note": "Phase 0 표본 smoke용. 반복 상한은 표본 고유 토큰 기준이라 본학습과 다르다"}
    Path("runs/phase0/data_sample.json").write_text(json.dumps(data, ensure_ascii=False, indent=1))
    Path("runs/phase0/token_stats.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
