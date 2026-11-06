"""전체 정제본을 고정 Tokenizer로 샤드(uint16, 최대 1억 토큰)로 만들고 Phase 1용 데이터 설정을 쓴다 (설계안 v0.3 4-1, 7-1절)

- 문서마다 <|endoftext|> + 토큰. 출처·분할마다 샤드 여러 개 (train_000.bin, ...)
- runs/phase0b/data_full.json: 가중치 = v0.3 목표 토큰, 반복 상한 = v0.3 (P2·P8a 2회, 나머지 1회)
- runs/phase0b/token_stats.json: 출처별 고유 토큰, 계획 반복 배수

    uv run --group data python scripts/tokenize_full.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from text import EOT_ID
from tokenizer import file_sha256, load_tokenizer

CLEAN = Path("data/full/clean")
TOKENS = Path("data/full/tokens")
TOKENIZER = Path("data/full/tokenizer/tokenizer.json")
SHARD_TOKENS = 100_000_000
BLOCK_SIZE = 1024
# v0.3 4-1: 목표 토큰(B), 반복 상한
TARGETS = {
    "P1_fineweb2_ko": (0.75, 1.0), "P2_wikipedia_ko": (0.58, 2.0), "P3_finepdfs_ko": (0.25, 1.0),
    "P4_fineweb_edu": (0.60, 1.0), "P5_cosmopedia_v2": (0.20, 1.0), "P6_wikipedia_en": (0.20, 1.0),
    "P8a_opc_algorithmic": (0.30, 2.0), "P7_finemath_4plus": (0.12, 1.0),
}


class ShardWriter:
    def __init__(self, prefix: Path):
        self.prefix, self.index, self.count, self.f, self.shards = prefix, 0, 0, None, []

    def write(self, arr: np.ndarray) -> None:
        if self.f is None or self.count >= SHARD_TOKENS:
            self.close()
            path = self.prefix.with_name(f"{self.prefix.name}_{self.index:03d}.bin")
            if path.exists():
                raise FileExistsError(path)
            self.f, self.count, self.path = open(path, "wb"), 0, path
            self.index += 1
        arr.tofile(self.f)
        self.count += len(arr)

    def close(self) -> None:
        if self.f is not None:
            self.f.close()
            self.shards.append({"path": str(self.path), "tokens": self.count, "sha256": file_sha256(self.path)})
            self.f = None


def main() -> None:
    tok = load_tokenizer(TOKENIZER)
    TOKENS.mkdir(parents=True, exist_ok=True)
    sources, report = [], {}
    for name, (target_b, max_repeat) in TARGETS.items():
        writers = {s: ShardWriter(TOKENS / f"{name}_{s}") for s in ("train", "val", "test")}
        docs = {s: 0 for s in writers}
        for batch in pq.ParquetFile(CLEAN / f"{name}.parquet").iter_batches(batch_size=4000, columns=["split", "text"]):
            splits, texts = batch["split"].to_pylist(), batch["text"].to_pylist()
            for split, enc in zip(splits, tok.encode_batch(texts, add_special_tokens=False)):
                writers[split].write(np.asarray([EOT_ID] + enc.ids, dtype=np.uint16))
                docs[split] += 1
        for w in writers.values():
            w.close()
        counts = {s: sum(x["tokens"] for x in w.shards) for s, w in writers.items()}
        target = target_b * 1e9
        report[name] = {"train_tokens": counts["train"], "val_tokens": counts["val"], "test_tokens": counts["test"],
                        "docs": docs, "target_tokens": int(target), "max_repeat": max_repeat,
                        "planned_repeat": round(target / counts["train"], 3),
                        "target_reachable": target <= counts["train"] * max_repeat}
        sources.append({"name": name, "weight": target_b, "max_repeat": max_repeat,
                        "train": writers["train"].shards, "val": writers["val"].shards})
        print(f"{name}: train {counts['train'] / 1e6:,.1f}M 토큰 ({len(writers['train'].shards)} 샤드), "
              f"계획 반복 {report[name]['planned_repeat']} / 상한 {max_repeat}", flush=True)
        # 테스트 분할 샤드는 평가용으로 남기되 학습 설정에는 넣지 않는다
        report[name]["test_shards"] = writers["test"].shards
    data = {"tokenizer": str(TOKENIZER), "tokenizer_sha256": file_sha256(TOKENIZER), "block_size": BLOCK_SIZE,
            "sources": sources, "note": "설계안 v0.3 4-1. 가중치 = 목표 토큰(B)"}
    Path("runs/phase0b/data_full.json").write_text(json.dumps(data, ensure_ascii=False, indent=1))
    Path("runs/phase0b/token_stats.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    unreachable = [k for k, v in report.items() if not v["target_reachable"]]
    if unreachable:
        print(f"주의: 반복 상한으로 목표를 채울 수 없는 출처 {unreachable}", file=sys.stderr)


if __name__ == "__main__":
    main()
