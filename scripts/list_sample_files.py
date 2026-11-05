"""Phase 0 소량 표본 다운로드 계획: 파일 목록, 크기, 고정 Revision을 조회만 한다 (다운로드하지 않음)

    uv run --group data python scripts/list_sample_files.py > runs/phase0/sample_plan.json
"""

import json
import sys

from huggingface_hub import HfApi

# (이름, 저장소, 경로 접두어, 고를 파일 수, 용도). 파일 수 None = 접두어 아래 전부
PLAN = [
    ("P1 fineweb-2 ko", "HuggingFaceFW/fineweb-2", "data/kor_Hang/train/", 1, "pretrain"),
    ("P2 wikipedia ko", "wikimedia/wikipedia", "20231101.ko/", None, "pretrain"),
    ("P3 finepdfs ko", "HuggingFaceFW/finepdfs", "data/kor_Hang/train/", 1, "pretrain"),
    ("P4 fineweb-edu", "HuggingFaceFW/fineweb-edu", "sample/10BT/", 1, "pretrain"),
    ("P5 cosmopedia-v2", "HuggingFaceTB/smollm-corpus", "cosmopedia-v2/", 1, "pretrain"),
    ("P6 wikipedia en", "wikimedia/wikipedia", "20231101.en/", 1, "pretrain"),
    ("P7 finemath-4plus", "HuggingFaceTB/finemath", "finemath-4plus/", 1, "pretrain"),
    ("P8a opc algorithmic", "OpenCoder-LLM/opc-annealing-corpus", "algorithmic_corpus/", 1, "pretrain"),
    ("S1 aya", "CohereLabs/aya_dataset", "", None, "sft"),
    ("S2 oasst2", "OpenAssistant/oasst2", "", None, "sft"),
    ("S4 dolly-ko", "nlpai-lab/databricks-dolly-15k-ko", "", None, "sft"),
    ("S10 dolly", "databricks/databricks-dolly-15k", "", None, "sft"),
    ("E kobest", "skt/kobest_v1", "", None, "eval"),
    ("E hellaswag", "Rowan/hellaswag", "", None, "eval"),
    ("E arc", "allenai/ai2_arc", "ARC-Easy/", None, "eval"),
    ("E piqa", "ybisk/piqa", "", None, "eval"),
]
DATA_EXT = (".parquet", ".jsonl", ".json", ".jsonl.gz", ".json.gz", ".csv", ".txt")


def main() -> None:
    api = HfApi()
    out = []
    for name, repo, prefix, count, purpose in PLAN:
        try:
            info = api.dataset_info(repo)
            files = [f for f in api.list_repo_tree(repo, repo_type="dataset", revision=info.sha,
                                                   path_in_repo=prefix.rstrip("/") or None, recursive=True)
                     if getattr(f, "size", None) is not None and f.path.endswith(DATA_EXT)]
        except Exception as e:  # gated, 404 등은 그대로 보고한다
            out.append({"name": name, "repo": repo, "error": f"{type(e).__name__}: {e}"[:300]})
            continue
        files.sort(key=lambda f: f.path)
        total_bytes = sum(f.size for f in files)
        chosen = files if count is None else files[:count]
        out.append({
            "name": name, "repo": repo, "revision": info.sha, "purpose": purpose, "gated": info.gated,
            "files_in_prefix": len(files), "bytes_in_prefix": total_bytes,
            "chosen": [{"path": f.path, "bytes": f.size} for f in chosen],
            "chosen_bytes": sum(f.size for f in chosen),
        })
    json.dump(out, sys.stdout, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
