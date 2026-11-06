"""출처 하나의 전체 근사 중복 제거 (MinHash LSH). Phase 0b에서 코드(P8a)에 적용 (설계안 v0.3 7-2절)

- 64 perm 서명(uint32로 저장), LSH 16밴드 × 4행. 추정 Jaccard 0.8에서 후보가 될 확률 ≈ 1 - (1 - 0.8^4)^16 ≈ 0.9998
- 밴드마다 같은 버킷의 대표(가장 앞 문서)와 추정 Jaccard ≥ 0.8이면 뒤 문서를 제거한다 (분할과 무관: val·test 누수 방지)
- 원본은 <출처>.pre_neardup.parquet로 남긴다

    uv run --group data python scripts/dedup_near.py --clean data/full/clean --source P8a_opc_algorithmic
"""

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cleaning import MinHasher

NUM_PERM, BANDS, THRESHOLD = 64, 16, 0.8
_hasher = MinHasher(num_perm=NUM_PERM)


def signatures(texts: list[str]) -> np.ndarray:
    return np.stack([_hasher.signature(t) for t in texts]).astype(np.uint32)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--clean", type=Path, required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--procs", type=int, default=8)
    args = p.parse_args()
    path = args.clean / f"{args.source}.parquet"
    backup = args.clean / f"{args.source}.pre_neardup.parquet"
    if backup.exists():
        raise FileExistsError(f"{backup}가 이미 있다 (이미 처리됨)")
    t0 = time.time()
    pf = pq.ParquetFile(path)
    chunks = (b["text"].to_pylist() for b in pf.iter_batches(batch_size=5000, columns=["text"]))
    with Pool(args.procs) as pool:
        sig = np.concatenate(list(pool.imap(signatures, chunks)))
    n = len(sig)
    print(f"서명 {n:,}개, {time.time() - t0:.0f}s", flush=True)

    rows = NUM_PERM // BANDS
    dup = np.zeros(n, dtype=bool)
    for b in range(BANDS):
        band = np.ascontiguousarray(sig[:, b * rows : (b + 1) * rows]).view(np.dtype((np.void, 4 * rows))).ravel()
        _, first, inverse = np.unique(band, return_index=True, return_inverse=True)
        rep = first[inverse]  # 각 문서가 속한 버킷의 가장 앞 문서
        cand = np.nonzero(rep != np.arange(n))[0]
        sim = (sig[cand] == sig[rep[cand]]).mean(axis=1)
        dup[cand[sim >= THRESHOLD]] = True
        print(f"  밴드 {b + 1}/{BANDS}: 후보 {len(cand):,}, 누적 제거 {dup.sum():,}", flush=True)

    table = pq.read_table(path)
    os.replace(path, backup)
    keep = table.filter(~dup)
    pq.write_table(keep, path, compression="zstd")
    splits = keep["split"].to_pylist()
    stats = {"source": args.source, "docs_before": n, "near_dup_removed": int(dup.sum()),
             "removed_rate": round(float(dup.mean()), 5), "docs_after": keep.num_rows,
             "after_splits": {s: splits.count(s) for s in ("train", "val", "test")},
             "method": f"MinHash {NUM_PERM} perm, LSH {BANDS}×{rows}, 버킷 대표와 추정 Jaccard ≥ {THRESHOLD}",
             "seconds": round(time.time() - t0, 1)}
    out = Path("runs/phase0b") / f"near_dup_{args.source}.json"
    out.write_text(json.dumps(stats, ensure_ascii=False, indent=1))
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
