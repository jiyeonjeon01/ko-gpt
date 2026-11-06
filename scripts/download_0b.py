"""Phase 0b 추가 다운로드: runs/phase0b/plan.json에 적힌 파일·row group만 받는다 (승인 범위 약 7.8GB)

download_sample.py와 같은 규칙: Revision 고정, cache_type="none", 필요한 열만, 재시도 3회 후 중단,
네트워크 수신량과 디스크 저장량을 따로 기록, 이미 받은 항목은 건너뜀.

    uv run --group data python scripts/download_0b.py
"""

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem, hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parent))
from download_sample import SOURCES, file_sha256, retry, rx_bytes

OUT = Path("data/0b").resolve()
LIMIT_BYTES = 9.0e9  # 계획 7.8GB + 여유. 넘으면 중단


def main() -> None:
    plan = json.loads(Path("runs/phase0b/plan.json").read_text())
    columns = {s["name"]: s.get("columns") for s in SOURCES}
    meta = {s["name"]: {k: s[k] for k in ("license", "generator", "group")} for s in SOURCES}
    manifest_path = OUT / "manifest.json"
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"items": {}}
    fs = HfFileSystem()
    start = rx_bytes()

    for src in plan["sources"]:
        name = src["name"]
        for item in src["extra"]:
            key = f"{name}:{item['file']}"
            if key in manifest["items"]:
                continue
            print(f"== {key}", flush=True)
            rx0, t0 = rx_bytes(), time.time()
            out_dir = OUT / name
            out_dir.mkdir(parents=True, exist_ok=True)
            try:
                if "row_groups" in item:
                    remote = f"datasets/{item['repo']}@{item['revision']}/{item['file']}"
                    pf = retry(lambda: pq.ParquetFile(fs.open(remote, cache_type="none")), "metadata")
                    local = out_dir / (Path(item["file"]).stem + ".sample.parquet")
                    tmp = local.with_name(local.name + ".tmp")
                    writer, rows = None, 0
                    for rg in item["row_groups"]:
                        table = retry(lambda rg=rg: pf.read_row_group(rg, columns=columns[name]), f"row group {rg}")
                        table = table.append_column("source_row_group", pa.array([rg] * table.num_rows, pa.int32()))
                        if writer is None:
                            writer = pq.ParquetWriter(tmp, table.schema, compression="zstd")
                        writer.write_table(table)
                        rows += table.num_rows
                    writer.close()
                    tmp.replace(local)
                else:
                    local = Path(retry(lambda: hf_hub_download(item["repo"], item["file"], repo_type="dataset",
                                                               revision=item["revision"], local_dir=out_dir),
                                       item["file"])).resolve()
                    rows = None
            except Exception as e:
                manifest.setdefault("failures", []).append({"item": key, "error": f"{type(e).__name__}: {e}"[:500],
                                                            "time": datetime.now(timezone.utc).isoformat()})
                manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
                print(f"중단: {key} 실패. 남은 항목은 받지 않았다.", file=sys.stderr)
                sys.exit(1)
            manifest["items"][key] = {
                "source": name, **meta[name], "repo": item["repo"], "revision": item["revision"], "file": item["file"],
                "row_groups": item.get("row_groups"), "rows": rows, "columns": columns[name] if "row_groups" in item else None,
                "local": str(local.relative_to(OUT)), "disk_bytes": local.stat().st_size, "sha256": file_sha256(local),
                "network_rx_bytes": rx_bytes() - rx0, "seconds": round(time.time() - t0, 1),
                "downloaded_at": datetime.now(timezone.utc).isoformat(),
            }
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
            print(f"  디스크 {manifest['items'][key]['disk_bytes'] / 1e6:.1f} MB | 수신 "
                  f"{manifest['items'][key]['network_rx_bytes'] / 1e6:.1f} MB", flush=True)
            if rx_bytes() - start > LIMIT_BYTES:
                print("중단: 이번 실행의 수신량이 안전 상한 9.0GB를 넘었다", file=sys.stderr)
                sys.exit(2)
        import shutil
        shutil.rmtree(OUT / name / ".cache", ignore_errors=True)

    items = manifest["items"].values()
    manifest["totals"] = {"disk_bytes": sum(i["disk_bytes"] for i in items),
                          "network_rx_bytes": sum(i["network_rx_bytes"] for i in items)}
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    print(f"완료: 디스크 {manifest['totals']['disk_bytes'] / 1e9:.2f} GB, 수신 {manifest['totals']['network_rx_bytes'] / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
