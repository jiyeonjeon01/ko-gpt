"""Phase 0 소량 표본 다운로드 (승인 범위: 약 3.7GB, dolly-ko 제외)

- 모든 저장소는 Revision(commit) 고정
- 큰 Parquet는 row group 일부만 원격으로 읽는다. 선택: 층화 무작위 (구간 k개마다 1개, 고정 seed)
- 필요한 열만 저장 (zstd Parquet)
- 네트워크 수신량(/proc/net/dev, lo 제외)과 디스크 저장량을 출처별로 따로 기록
- 실패는 파일당 3회까지 재시도하고, 그래도 실패하면 중단하고 보고한다 (나머지는 받지 않음)
- 이미 받은 출처(manifest에 기록됨)는 건너뛴다

    uv run --group data python scripts/download_sample.py --out data/sample
"""

import argparse
import hashlib
import json
import random
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem, hf_hub_download

SEED = 20261105
MAX_ATTEMPTS = 3
LIMIT_BYTES = 4.0e9  # 승인 범위(약 3.7GB)를 넘으면 중단하는 안전장치 (네트워크 수신량 기준)

# 라이선스·생성 방식·그룹: 설계안 v0.2 2절
SOURCES = [
    {"name": "P1_fineweb2_ko", "repo": "HuggingFaceFW/fineweb-2", "revision": "af9c13333e",
     "path": "data/kor_Hang/train/000_00000.parquet", "row_groups": 400,
     "columns": ["text", "id", "url", "date", "language_score", "minhash_cluster_size"],
     "license": "ODC-BY + Common Crawl ToU", "generator": "web (human)", "group": "A"},
    {"name": "P2_wikipedia_ko", "repo": "wikimedia/wikipedia", "revision": "b04c8d1ceb",
     "files": ["20231101.ko/train-00000-of-00003.parquet", "20231101.ko/train-00001-of-00003.parquet",
               "20231101.ko/train-00002-of-00003.parquet"],
     "license": "CC-BY-SA 3.0 / GFDL", "generator": "human", "group": "A"},
    {"name": "P3_finepdfs_ko", "repo": "HuggingFaceFW/finepdfs", "revision": "220bac3acb",
     "path": "data/kor_Hang/train/000_00000.parquet", "row_groups": 30,
     "columns": ["text", "id", "url", "token_count", "minhash_cluster_size", "duplicate_count"],
     "license": "ODC-BY + Common Crawl ToU", "generator": "web PDF (human)", "group": "A"},
    {"name": "P4_fineweb_edu", "repo": "HuggingFaceFW/fineweb-edu", "revision": "87f09149ef",
     "path": "sample/10BT/000_00000.parquet", "row_groups": 100,
     "columns": ["text", "id", "url", "score", "token_count"],
     "license": "ODC-BY + Common Crawl ToU", "generator": "web (human), Llama3 분류기로 선별만", "group": "A"},
    {"name": "P5_cosmopedia_v2", "repo": "HuggingFaceTB/smollm-corpus", "revision": "3ba9d60577",
     "path": "cosmopedia-v2/train-00000-of-00104.parquet", "row_groups": 100,
     "columns": ["text", "format", "audience", "seed_data", "token_length"],
     "license": "ODC-BY", "generator": "Mixtral-8x7B-Instruct-v0.1 (Apache-2.0); 시드 데이터 권리는 미확인",
     "group": "A"},
    {"name": "P6_wikipedia_en", "repo": "wikimedia/wikipedia", "revision": "b04c8d1ceb",
     "files": ["20231101.en/train-00000-of-00041.parquet"],
     "license": "CC-BY-SA 3.0 / GFDL", "generator": "human", "group": "A"},
    {"name": "P7_finemath_4plus", "repo": "HuggingFaceTB/finemath", "revision": "e92b25a616",
     "files": ["finemath-4plus/train-00000-of-00064.parquet"],
     "license": "ODC-BY + Common Crawl ToU", "generator": "web (human), 분류기로 선별만", "group": "A"},
    {"name": "P8a_opc_algorithmic", "repo": "OpenCoder-LLM/opc-annealing-corpus", "revision": "078e4bc1d3",
     "files": ["algorithmic_corpus/1.jsonl"],
     "license": "ODC-BY (원 코드는 The Stack v2, 파일별 라이선스 포함 여부 확인 필요)", "generator": "code (human)",
     "group": "A"},
    {"name": "S1_aya", "repo": "CohereLabs/aya_dataset", "revision": "f9ea04583f",
     "files": ["data/train-00000-of-00001.parquet", "data/test-00000-of-00001.parquet"],
     "license": "Apache-2.0", "generator": "human", "group": "A"},
    {"name": "S2_oasst2", "repo": "OpenAssistant/oasst2", "revision": "179dd21fc5",
     "files": ["2023-11-05_oasst2_ready.trees.jsonl.gz"],
     "license": "Apache-2.0", "generator": "human", "group": "A"},
    {"name": "S10_dolly", "repo": "databricks/databricks-dolly-15k", "revision": "bdd27f4d94",
     "files": ["databricks-dolly-15k.jsonl"],
     "license": "CC-BY-SA 3.0", "generator": "human", "group": "A"},
    {"name": "E_kobest", "repo": "skt/kobest_v1", "revision": "a5ea15e3ac",
     "files": [f"{t}/{s}.jsonl" for t in ("boolq", "copa", "hellaswag", "sentineg", "wic")
               for s in ("train", "validation", "test")],
     "license": "CC-BY-SA 4.0", "generator": "human", "group": "eval"},
    {"name": "E_hellaswag", "repo": "Rowan/hellaswag", "revision": "218ec52e09",
     "files": ["data/train-00000-of-00001.parquet", "data/validation-00000-of-00001.parquet",
               "data/test-00000-of-00001.parquet"],
     "license": "MIT", "generator": "human + adversarial filtering", "group": "eval"},
    {"name": "E_arc_easy", "repo": "allenai/ai2_arc", "revision": "210d026faf",
     "files": ["ARC-Easy/train-00000-of-00001.parquet", "ARC-Easy/validation-00000-of-00001.parquet",
               "ARC-Easy/test-00000-of-00001.parquet"],
     "license": "CC-BY-SA 4.0", "generator": "human", "group": "eval"},
    {"name": "E_piqa", "repo": "ybisk/piqa", "revision": "142c51238b3ca2bc61e9a075913871b8b600e8e1",
     "files": ["plain_text/train/0000.parquet", "plain_text/validation/0000.parquet",
               "plain_text/test/0000.parquet"],
     "license": "AFL-3.0 (카드 기준, 재확인 필요)", "generator": "human", "group": "eval"},
]


def rx_bytes() -> int:
    total = 0
    for line in Path("/proc/net/dev").read_text().splitlines()[2:]:
        iface, data = line.split(":", 1)
        if iface.strip() != "lo":
            total += int(data.split()[0])
    return total


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stratified_row_groups(n: int, k: int, seed: int) -> list[int]:
    """0..n-1을 k개 구간으로 나눠 구간마다 하나씩 무작위로 고른다 (특정 구간 편중 방지)"""
    rng = random.Random(seed)
    bounds = [round(i * n / k) for i in range(k + 1)]
    return [rng.randrange(bounds[i], bounds[i + 1]) for i in range(k)]


def retry(fn, what: str):
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return fn()
        except Exception as e:  # 네트워크 오류 등
            print(f"  실패 {attempt}/{MAX_ATTEMPTS}: {what}: {type(e).__name__}: {e}", flush=True)
            if attempt == MAX_ATTEMPTS:
                raise
            time.sleep(10 * attempt)


def full_revision(repo: str, prefix: str) -> str:
    """계획 조회(runs/phase0/sample_plan.json)에 기록된 전체 commit 해시. 접두어가 맞지 않으면 중단한다."""
    if len(prefix) == 40:
        return prefix
    plan = json.loads(Path("runs/phase0/sample_plan.json").read_text())
    sha = next(r["revision"] for r in plan if r["repo"] == repo and "revision" in r)
    if not sha.startswith(prefix):
        raise RuntimeError(f"{repo}: 계획 파일의 Revision({sha[:10]})이 {prefix}와 다르다")
    return sha


def download_row_groups(fs, src, revision, out_dir: Path) -> dict:
    remote = f"datasets/{src['repo']}@{revision}/{src['path']}"
    # 캐시 없이 요청한 바이트 범위만 받는다. 1MB 블록 캐시는 같은 구간을 다시 받아 수신량이 필요량의 1.66배였다
    # (P1 실측: 디스크 693MB, 수신 1,170MB). cache_type="none"은 1.05배 (3개 row group 시험)
    pf = retry(lambda: pq.ParquetFile(fs.open(remote, cache_type="none")), "metadata")
    n = pf.metadata.num_row_groups
    chosen = stratified_row_groups(n, src["row_groups"], SEED)
    out_path = out_dir / "sample.parquet"
    tmp = out_path.with_name(out_path.name + ".tmp")
    writer, rows = None, 0
    try:
        for i, rg in enumerate(chosen):
            table = retry(lambda rg=rg: pf.read_row_group(rg, columns=src["columns"]), f"row group {rg}")
            table = table.append_column("source_row_group", pa.array([rg] * table.num_rows, pa.int32()))
            if writer is None:
                writer = pq.ParquetWriter(tmp, table.schema, compression="zstd")
            writer.write_table(table)
            rows += table.num_rows
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(chosen)} row groups, {rows:,} rows", flush=True)
        writer.close()
        tmp.replace(out_path)
    except BaseException:
        if writer is not None:
            writer.close()
        tmp.unlink(missing_ok=True)
        raise
    return {"path": src["path"], "row_groups_total": n, "row_groups_chosen": chosen,
            "selection": f"stratified random, k={src['row_groups']}, seed={SEED}", "rows": rows,
            "columns": src["columns"], "local_files": [out_path]}


def download_files(src, revision, out_dir: Path) -> dict:
    local = []
    for name in src["files"]:
        path = retry(lambda name=name: hf_hub_download(src["repo"], name, repo_type="dataset", revision=revision,
                                                        local_dir=out_dir), name)
        local.append(Path(path))
    return {"files": src["files"], "local_files": local}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("data/sample"))
    args = parser.parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"sources": {}}
    manifest["seed"] = SEED
    fs = HfFileSystem()
    start_rx = rx_bytes()

    for src in SOURCES:
        if src["name"] in manifest["sources"]:
            print(f"건너뜀 (이미 받음): {src['name']}")
            continue
        print(f"== {src['name']} ({src['repo']})", flush=True)
        out_dir = args.out / src["name"]
        rx0, t0 = rx_bytes(), time.time()
        try:
            revision = full_revision(src["repo"], src["revision"])
            if "row_groups" in src:
                result = download_row_groups(fs, src, revision, out_dir.mkdir(parents=True, exist_ok=True) or out_dir)
            else:
                result = download_files(src, revision, out_dir)
        except Exception as e:
            manifest.setdefault("failures", []).append(
                {"name": src["name"], "error": f"{type(e).__name__}: {e}"[:500],
                 "time": datetime.now(timezone.utc).isoformat()})
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
            print(f"중단: {src['name']} 다운로드 실패. 남은 출처는 받지 않았다.", file=sys.stderr)
            sys.exit(1)
        local = [Path(p).resolve() for p in result.pop("local_files")]
        shutil.rmtree(out_dir / ".cache", ignore_errors=True)  # hf_hub_download의 메타데이터 폴더
        entry = {k: v for k, v in src.items() if k not in ("files", "path", "row_groups", "columns", "revision")}
        entry.update(result)
        entry.update(
            revision=revision,
            local_files=[{"path": str(p.relative_to(args.out)), "bytes": p.stat().st_size,
                          "sha256": file_sha256(p)} for p in local],
            disk_bytes=sum(p.stat().st_size for p in local),
            network_rx_bytes=rx_bytes() - rx0,
            seconds=round(time.time() - t0, 1),
            downloaded_at=datetime.now(timezone.utc).isoformat(),
        )
        manifest["sources"][src["name"]] = entry
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
        print(f"  디스크 {entry['disk_bytes'] / 1e6:.1f} MB | 네트워크 수신 {entry['network_rx_bytes'] / 1e6:.1f} MB "
              f"| {entry['seconds']}s", flush=True)
        if rx_bytes() - start_rx > LIMIT_BYTES:
            print("중단: 이번 실행의 네트워크 수신량이 안전 상한 4.0GB를 넘었다", file=sys.stderr)
            sys.exit(2)

    total_disk = sum(s["disk_bytes"] for s in manifest["sources"].values())
    total_rx = sum(s["network_rx_bytes"] for s in manifest["sources"].values())
    manifest["totals"] = {"disk_bytes": total_disk, "network_rx_bytes": total_rx}
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    print(f"완료: 디스크 {total_disk / 1e9:.3f} GB, 네트워크 수신 {total_rx / 1e9:.3f} GB")


if __name__ == "__main__":
    main()
