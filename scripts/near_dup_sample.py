"""출처 내·출처 간 근사 중복 비율 (MinHash 표본). prepare_sample.py가 쓰고, 단독으로도 다시 잴 수 있다.

표본 안에서만 짝을 찾으므로 실제 비율의 하한이다.

    uv run --group data python scripts/near_dup_sample.py   # data/sample/clean에서 다시 표본추출해 clean_stats.json 갱신
"""

import json
import random
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cleaning import MinHasher, estimated_jaccard, lsh_candidate_pairs

SAMPLE_DOCS = 20_000
THRESHOLD = 0.8
NUM_PERM, BANDS = 128, 32  # 밴드당 4행: 추정 Jaccard 0.8에서 후보가 될 확률 ≈ 1 - (1 - 0.8^4)^32 ≈ 1.0
CROSS_PAIRS = [("P1_fineweb2_ko", "P2_wikipedia_ko"), ("P1_fineweb2_ko", "P3_finepdfs_ko"),
               ("P3_finepdfs_ko", "P2_wikipedia_ko"), ("P4_fineweb_edu", "P6_wikipedia_en"),
               ("P4_fineweb_edu", "P5_cosmopedia_v2"), ("P4_fineweb_edu", "P7_finemath_4plus")]
METHOD = (f"MinHash {NUM_PERM} perm(64비트 multiply-xorshift), 문자 5-gram, 앞 3000자, LSH {BANDS}밴드×"
          f"{NUM_PERM // BANDS}행, 추정 Jaccard ≥ {THRESHOLD}, 출처당 최대 {SAMPLE_DOCS:,} 문서 표본 안에서만 짝 탐색 (하한)")


def near_dup_report(samples: dict[str, list[str]]) -> dict:
    hasher = MinHasher(num_perm=NUM_PERM)
    sigs = {}
    for name, texts in samples.items():
        sigs[name] = np.stack([hasher.signature(t) for t in texts])
        print(f"MinHash {name}: {len(texts):,} 문서", flush=True)
    within = {}
    for name, sig in sigs.items():
        pairs = [p for p in lsh_candidate_pairs(sig, bands=BANDS)
                 if estimated_jaccard(sig[p[0]], sig[p[1]]) >= THRESHOLD]
        docs = {d for p in pairs for d in p}
        within[name] = {"sample_docs": len(sig), "near_dup_pairs": len(pairs), "docs_in_pairs": len(docs),
                        "rate_lower_bound": round(len(docs) / max(len(sig), 1), 5)}
    cross = {}
    for a, b in CROSS_PAIRS:
        sa, sb = sigs[a], sigs[b]
        both = np.concatenate([sa, sb])
        pairs = [p for p in lsh_candidate_pairs(both, bands=BANDS)
                 if (p[0] < len(sa)) != (p[1] < len(sa))
                 and estimated_jaccard(both[p[0]], both[p[1]]) >= THRESHOLD]
        docs_a = {min(p) for p in pairs}
        cross[f"{a}~{b}"] = {"pairs": len(pairs), "docs_of_first_with_match": len(docs_a),
                             "rate_of_first": round(len(docs_a) / len(sa), 5)}
    return {"near_dup_within_sample": within, "near_dup_cross_sample": cross, "near_dup_method": METHOD}


def main() -> None:
    stats_path = Path("runs/phase0/clean_stats.json")
    stats = json.loads(stats_path.read_text())
    samples = {}
    for name in stats["sources"]:
        texts = pq.read_table(f"data/sample/clean/{name}.parquet", columns=["text"])["text"].to_pylist()
        idx = random.Random(f"neardup:{name}").sample(range(len(texts)), min(SAMPLE_DOCS, len(texts)))
        samples[name] = [texts[i] for i in idx]
        del texts
    report = near_dup_report(samples)
    stats.update(report)
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=1))
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
