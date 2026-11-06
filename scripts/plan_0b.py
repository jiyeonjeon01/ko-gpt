"""Phase 0b 다운로드 계획: 설계안 v0.3 4-1 목표를 채우는 데 필요한 추가 파일·row group과 정확한 크기 (조회만)

필요 문서 수 = 목표 토큰 × 1.1(여유) / (Phase 0 표본의 입력 문서당 토큰). 입력 문서당 토큰은 정제 통과율이 반영된 값이다.
이미 받은 Phase 0 표본은 그대로 쓰고, 추가분은 다른 파일에서 층화 무작위로 고른다 (한 파일 편중 방지).
크기는 Parquet 메타데이터에서 필요한 열의 압축 크기 합 (cache_type="none" 시험에서 수신량 ≈ 1.05배).

    uv run --group data python scripts/plan_0b.py
"""

import json
import math
import sys
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem

sys.path.insert(0, str(Path(__file__).resolve().parent))
from download_sample import SEED, SOURCES, full_revision, stratified_row_groups

TARGET_B = {"P1_fineweb2_ko": 0.75, "P3_finepdfs_ko": 0.25, "P4_fineweb_edu": 0.60, "P5_cosmopedia_v2": 0.20,
            "P6_wikipedia_en": 0.20, "P7_finemath_4plus": 0.12}
MARGIN = 1.1
# 추가분을 가져올 파일 (Phase 0 표본과 다른 파일)
EXTRA_FILES = {
    "P1_fineweb2_ko": [f"data/kor_Hang/train/{i:03d}_00000.parquet" for i in range(1, 5)],
    "P3_finepdfs_ko": ["data/kor_Hang/train/001_00000.parquet"],
    "P4_fineweb_edu": [f"sample/10BT/{i:03d}_00000.parquet" for i in range(1, 6)],
    "P5_cosmopedia_v2": [f"cosmopedia-v2/train-{i:05d}-of-00104.parquet" for i in range(1, 4)],
}


def main() -> None:
    api, fs = HfApi(), HfFileSystem()
    clean = json.loads(Path("runs/phase0/clean_stats.json").read_text())["sources"]
    tokens = json.loads(Path("runs/phase0/token_stats.json").read_text())
    src_by_name = {s["name"]: s for s in SOURCES}
    plan, total = [], 0

    for name, target in TARGET_B.items():
        per_doc = tokens[name]["sample_tokens"] / clean[name]["docs_in"]  # 입력 문서당 토큰 (통과율 반영)
        need_docs = math.ceil(target * 1e9 * MARGIN / per_doc)
        have_docs = clean[name]["docs_in"]
        src = src_by_name[name]
        entry = {"name": name, "target_B": target, "tokens_per_input_doc": round(per_doc, 1),
                 "need_input_docs": need_docs, "have_input_docs": have_docs, "extra": []}
        missing = need_docs - have_docs
        if missing <= 0:
            entry["note"] = "Phase 0 표본으로 충분"
        elif name == "P6_wikipedia_en":
            # 파일 단위: 다음 파일 하나 (약 15만 문서)
            path = "20231101.en/train-00001-of-00041.parquet"
            rev = full_revision(src["repo"], src["revision"])
            size = api.get_paths_info(src["repo"], [path], repo_type="dataset", revision=rev)[0].size
            entry["extra"].append({"repo": src["repo"], "revision": rev, "file": path, "bytes": size})
        else:
            rev = full_revision(src["repo"], src["revision"])
            files = EXTRA_FILES[name]
            metas = [pq.ParquetFile(fs.open(f"datasets/{src['repo']}@{rev}/{p}", cache_type="none")).metadata
                     for p in files]
            rows_per_rg = metas[0].row_group(0).num_rows
            rg_needed = math.ceil(missing / rows_per_rg)
            per_file = [rg_needed // len(files) + (i < rg_needed % len(files)) for i in range(len(files))]
            for path, md, k in zip(files, metas, per_file):
                chosen = stratified_row_groups(md.num_row_groups, k, SEED + 1)
                cols = set(src["columns"])
                size = sum(md.row_group(i).column(j).total_compressed_size for i in chosen
                           for j in range(md.num_columns) if md.schema.column(j).name in cols)
                entry["extra"].append({"repo": src["repo"], "revision": rev, "file": path,
                                       "row_groups_total": md.num_row_groups, "row_groups": chosen,
                                       "rows": sum(md.row_group(i).num_rows for i in chosen), "bytes": size})
        entry["extra_bytes"] = sum(e["bytes"] for e in entry["extra"])
        total += entry["extra_bytes"]
        plan.append(entry)
        print(f"{name}: 필요 {need_docs:,} / 보유 {have_docs:,} 문서 → 추가 {entry['extra_bytes'] / 1e6:,.1f} MB "
              f"({len(entry['extra'])}개 파일)", flush=True)

    # P8a 코드: 근사 중복 제거 후 고유 토큰을 확보하기 위해 나머지 53개 파일 전부
    src = src_by_name["P8a_opc_algorithmic"]
    rev = full_revision(src["repo"], src["revision"])
    files = [f for f in api.list_repo_tree(src["repo"], repo_type="dataset", revision=rev,
                                           path_in_repo="algorithmic_corpus")
             if f.path.endswith(".jsonl") and f.path != "algorithmic_corpus/1.jsonl"]
    code = {"name": "P8a_opc_algorithmic", "note": "나머지 파일 전부 (근사 중복 제거 전 0.99B 추정, 표본 근사 중복 22.6% 이상)",
            "extra": [{"repo": src["repo"], "revision": rev, "file": f.path, "bytes": f.size} for f in files]}
    code["extra_bytes"] = sum(f.size for f in files)
    total += code["extra_bytes"]
    plan.append(code)
    print(f"P8a_opc_algorithmic: 추가 {len(files)}개 파일 {code['extra_bytes'] / 1e6:,.1f} MB")
    print(f"합계 추가 {total / 1e9:.2f} GB (필요한 열의 압축 크기 기준)")
    Path("runs/phase0b").mkdir(parents=True, exist_ok=True)
    Path("runs/phase0b/plan.json").write_text(json.dumps({"seed": SEED + 1, "margin": MARGIN, "sources": plan,
                                                          "total_bytes": total}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
