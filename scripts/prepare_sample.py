"""표본(Phase 0)·전체(Phase 0b) 정제와 통계 (설계안 v0.2 7-1 ~ 7-3절)

전처리 → 출처별 필터 → 개인정보 치환 → 정확 중복(출처 내·출처 간) → 평가 세트 13-gram 오염 → 분할(문서 ID 해시)
→ data/sample/clean/<출처>.parquet 와 runs/phase0/clean_stats.json
근사 중복(MinHash)은 near_dup_sample.py가 출처마다 최대 20,000 문서 표본에서 잰다 (표본 안에서만 짝을 찾으므로 하한값).

    uv run --group data python scripts/prepare_sample.py                # Phase 0 표본
    uv run --group data python scripts/prepare_sample.py --stage full   # Phase 0b: 표본 + data/0b 추가분 → data/full/clean
"""

import argparse
import gzip
import hashlib
import json
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cleaning import filter_code, filter_ko_web, filter_length_only, filter_pdf, filter_wiki, mask_pii, text_digest
from near_dup_sample import SAMPLE_DOCS as NEAR_DUP_SAMPLE, near_dup_report
from text import preprocess

SAMPLE = Path("data/sample")
CLEAN = SAMPLE / "clean"
STATS = Path("runs/phase0/clean_stats.json")
SPLIT_SEED = "ko-gpt-split-v1"
NGRAM = 13

# 출처별: 파일 형식, 텍스트·ID·URL·제목 필드, 필터, 언어
PRETRAIN = {
    "P1_fineweb2_ko": dict(text="text", id="id", url="url", filter=lambda r: filter_ko_web(r["text"], r.get("url"))),
    "P2_wikipedia_ko": dict(text="text", id="id", url="url", title="title",
                            filter=lambda r: filter_wiki(r["text"], r.get("title"), "ko")),
    "P3_finepdfs_ko": dict(text="text", id="id", url="url", filter=lambda r: filter_pdf(r["text"])),
    "P4_fineweb_edu": dict(text="text", id="id", url="url", filter=lambda r: filter_length_only(r["text"])),
    "P5_cosmopedia_v2": dict(text="text", id=None, filter=lambda r: filter_length_only(r["text"])),
    "P6_wikipedia_en": dict(text="text", id="id", url="url", title="title",
                            filter=lambda r: filter_wiki(r["text"], r.get("title"), "en")),
    "P7_finemath_4plus": dict(text="text", id="url", url="url", filter=lambda r: filter_length_only(r["text"])),
    "P8a_opc_algorithmic": dict(text="text", id=None, filter=lambda r: filter_code(r["text"])),
}

_WORD = re.compile(r"\w+")


def split_of(doc_id: str) -> str:
    """문서 ID 해시로 고정 분할: val 0.5%, test 0.5%, 나머지 train"""
    h = int.from_bytes(hashlib.sha256(f"{SPLIT_SEED}:{doc_id}".encode()).digest()[:8], "big") % 1000
    return "val" if h < 5 else "test" if h < 10 else "train"


def input_files(stage: str) -> dict[str, list[Path]]:
    """출처별 입력 파일. full은 표본 파일 뒤에 0b 추가분을 붙인다 (표본의 문서 순번과 분할이 그대로 유지됨)"""
    manifest = json.loads((SAMPLE / "manifest.json").read_text())
    files = {name: [SAMPLE / f["path"] for f in manifest["sources"][name]["local_files"]] for name in PRETRAIN}
    if stage == "full":
        extra = json.loads(Path("data/0b/manifest.json").read_text())["items"]
        for item in sorted(extra.values(), key=lambda i: i["file"]):
            files[item["source"]].append(Path("data/0b") / item["local"])
    return files


def iter_records(paths: list[Path]):
    for path in paths:
        if path.suffix == ".parquet":
            pf = pq.ParquetFile(path)
            for batch in pf.iter_batches(batch_size=2000):
                yield from batch.to_pylist()
        else:
            opener = gzip.open if path.suffix == ".gz" else open
            with opener(path, "rt", encoding="utf-8") as fh:
                for line in fh:
                    yield json.loads(line)


def word_ngrams(text: str, n: int = NGRAM):
    words = _WORD.findall(text.lower())
    for i in range(len(words) - n + 1):
        yield hash(" ".join(words[i : i + n]))


def eval_ngrams() -> tuple[set[int], dict]:
    """평가 세트 텍스트의 13-gram (단어 단위, 소문자)"""
    texts = []
    for f in (SAMPLE / "E_kobest").rglob("*.jsonl"):
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            texts.append(" ".join(str(v) for v in r.values() if isinstance(v, str)))
    for name in ("E_hellaswag", "E_arc_easy", "E_piqa"):
        for f in (SAMPLE / name).rglob("*.parquet"):
            for r in pq.read_table(f).to_pylist():
                texts.append(" ".join(str(v) for v in r.values() if isinstance(v, (str, dict, list))))
    grams = set()
    for t in texts:
        grams.update(word_ngrams(t))
    return grams, {"eval_texts": len(texts), "eval_13grams": len(grams)}


def flush(rows: list, writer, name: str):
    """rows를 Parquet에 쓰고 비운다 (출처 전체를 메모리에 올리지 않기 위함)"""
    if rows:
        table = pa.Table.from_pylist(rows, schema=pa.schema([("doc_id", pa.string()), ("split", pa.string()),
                                                               ("text", pa.string())]))
        if writer is None:
            writer = pq.ParquetWriter(CLEAN / f"{name}.parquet", table.schema, compression="zstd")
        writer.write_table(table)
        rows.clear()
    return writer


def main() -> None:
    global CLEAN, STATS
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["sample", "full"], default="sample")
    args = parser.parse_args()
    if args.stage == "full":
        CLEAN, STATS = Path("data/full/clean"), Path("runs/phase0b/clean_stats.json")
    if any(CLEAN.glob("*.parquet")):
        raise FileExistsError(f"{CLEAN}에 이미 정제 결과가 있다 (덮어쓰지 않음)")
    files = input_files(args.stage)
    CLEAN.mkdir(parents=True, exist_ok=True)
    STATS.parent.mkdir(parents=True, exist_ok=True)
    contam_grams, contam_info = eval_ngrams()
    print(f"평가 세트 13-gram {contam_info}", flush=True)
    first_source: dict[bytes, str] = {}  # 정확 중복: 다이제스트 → 처음 나온 출처
    stats = {"stage": args.stage, "split_seed": SPLIT_SEED, "contamination": contam_info, "sources": {},
             "inputs": {k: [str(p) for p in v] for k, v in files.items()}}
    kept_texts: dict[str, list[str]] = {}

    for name, spec in PRETRAIN.items():
        t0 = time.time()
        s = Counter()
        reasons, pii, cross_dup = Counter(), Counter(), Counter()
        rows, reservoir = [], []
        writer = None
        rng_res = random.Random(name)
        for i, r in enumerate(iter_records(files[name])):
            raw = r.get(spec["text"]) or ""
            s["docs_in"] += 1
            s["chars_in"] += len(raw)
            text = preprocess(raw)
            if "< |" in text and "<|" in raw:
                s["special_string_defused"] += 1
            reason = spec["filter"]({**r, "text": text})
            if reason:
                reasons[reason] += 1
                continue
            text, counts = mask_pii(text)
            pii.update(counts)
            digest = text_digest(text)
            if digest in first_source:
                if first_source[digest] == name:
                    reasons["exact_dup_within"] += 1
                else:
                    reasons["exact_dup_cross"] += 1
                    cross_dup[first_source[digest]] += 1
                continue
            first_source[digest] = name
            if any(g in contam_grams for g in word_ngrams(text)):
                reasons["eval_contamination"] += 1
                continue
            doc_id = str(r.get(spec["id"]) if spec["id"] else f"{name}:{i}")
            split = split_of(doc_id)
            rows.append({"doc_id": doc_id, "split": split, "text": text})
            s[f"docs_{split}"] += 1
            s["docs_kept"] += 1
            s["chars_kept"] += len(text)
            s["bytes_kept"] += len(text.encode("utf-8"))
            # 근사 중복 표본: 저장소 표본추출(reservoir)로 메모리에 최대 NEAR_DUP_SAMPLE개만 둔다
            if len(reservoir) < NEAR_DUP_SAMPLE:
                reservoir.append(text)
            else:
                j = rng_res.randrange(s["docs_kept"])
                if j < NEAR_DUP_SAMPLE:
                    reservoir[j] = text
            if len(rows) >= 5000:
                writer = flush(rows, writer, name)
        writer = flush(rows, writer, name)
        writer.close()
        stats["sources"][name] = {**s, "drop_reasons": dict(reasons), "pii_masked": dict(pii),
                                  "exact_dup_cross_from": dict(cross_dup),
                                  "keep_rate": s["docs_kept"] / max(s["docs_in"], 1),
                                  "seconds": round(time.time() - t0, 1)}
        kept_texts[name] = reservoir
        print(f"{name}: {s['docs_in']:,} → {s['docs_kept']:,} 문서 ({s['docs_kept'] / max(s['docs_in'], 1):.1%}), "
              f"{s['chars_kept'] / 1e6:,.1f}M자 | 탈락 {dict(reasons)} | {time.time() - t0:.0f}s", flush=True)
        STATS.write_text(json.dumps(stats, ensure_ascii=False, indent=1))

    # 근사 중복 (출처 내 표본, 출처 쌍 표본)
    stats.update(near_dup_report(kept_texts))
    STATS.write_text(json.dumps(stats, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
