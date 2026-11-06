"""최종 Tokenizer (V=32,768) 학습과 고정 (설계안 v0.3 D1)

- 학습 샘플: 전체 정제본(data/full/clean)의 train 분할에서 설계 비율(한국어 50, 영어 30, 코드 15, 수학 5) 1GB
- 확인: 같은 val 문서에서 Phase 0 Tokenizer와 압축률 비교. 한국어 출처가 ±2% 안이어야 한다 (벗어나면 종료 코드 1, 고정하지 않음)
- 저장: data/full/tokenizer/tokenizer.json (덮어쓰기 거부)

    uv run --group data python scripts/train_final_tokenizer.py
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import tokenizer_compare as tc
from tokenizer import load_tokenizer, save_tokenizer, train_tokenizer

VOCAB = 32768
OUT = Path("data/full/tokenizer/tokenizer.json")
PHASE0 = Path("runs/phase0/tokenizers/v32768/tokenizer.json")
KOREAN = ("P1_fineweb2_ko", "P2_wikipedia_ko", "P3_finepdfs_ko")
TOLERANCE = 0.02


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"{OUT}가 이미 있다 (Tokenizer는 고정)")
    tc.CLEAN = Path("data/full/clean")
    chosen, info = tc.plan_training_sample(10**9)
    print(f"학습 샘플 {info['total_bytes'] / 1e6:.1f}MB, {info['docs']:,} 문서: "
          + ", ".join(f"{k} {v['actual_share']:.3f}" for k, v in info["sources"].items()), flush=True)
    t0 = time.time()
    tok = train_tokenizer(tc.training_docs(chosen), vocab_size=VOCAB)
    seconds = time.time() - t0
    final = tc.measure(tok)
    phase0 = tc.measure(load_tokenizer(PHASE0))
    compare = {name: {"phase0": phase0[name]["chars_per_token"], "final": final[name]["chars_per_token"],
                      "ratio": round(final[name]["chars_per_token"] / phase0[name]["chars_per_token"], 4)}
               for name in final}
    ok = all(abs(compare[k]["ratio"] - 1) <= TOLERANCE for k in KOREAN)
    report = {"vocab": VOCAB, "train_seconds": round(seconds, 1), "training_sample": info, "val": final,
              "compare_with_phase0": compare, "korean_within_2pct": ok}
    Path("runs/phase0b/final_tokenizer.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    for k, v in compare.items():
        print(f"  {k}: Phase 0 {v['phase0']} → 최종 {v['final']} ({v['ratio']:.3f})")
    if not ok:
        print("중단: 한국어 압축률이 Phase 0 값의 ±2%를 벗어났다. 고정하지 않았다", file=sys.stderr)
        sys.exit(1)
    sha = save_tokenizer(tok, OUT, {"phase": "0b", "training_sample": info["sha256_of_doc_ids"],
                                    "training_shares": {k: v["actual_share"] for k, v in info["sources"].items()}})
    print(f"고정: {OUT} sha256 {sha} ({seconds:.0f}s)")


if __name__ == "__main__":
    main()
