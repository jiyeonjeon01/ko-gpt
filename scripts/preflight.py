"""학습 시작 전 자동 점검: Windows RAM, WSL 메모리, 다른 WSL 배포판, GPU VRAM, 디스크, 다른 학습 프로세스

    python3 scripts/preflight.py --out-dir runs/x -- --model main-110m --max-steps 11600 ...

통과하면 종료 코드 0, 하나라도 실패하면 8. 결과는 <out-dir>/preflight.jsonl에 덧붙인다.
실패해도 프로그램을 닫거나 설정을 바꾸지 않는다 (사람이 처리한다). 학습 중 안전 정지 기준(monitor.py)과는 별개다.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from monitor import GPU_MARGIN_MB, gpu, windows_available_mb, windows_top_processes, wsl_memory_mb  # noqa: E402

EXIT_PREFLIGHT = 8
# 시작 전 기준 (메모리_안정성_검토.md 3-A). 학습 중 WSL VM이 약 7.5GB를 쓰므로 Windows 여유는 넉넉해야 한다
MIN_WIN_AVAIL_MB = 6144
MIN_WSL_AVAIL_MB = 4096
DISK_MARGIN_GB = 2.0
CUDA_CONTEXT_MIB = 512
# 프리셋별 PyTorch reserved (측정): pilot-42m 6,626MiB (Phase 1), main-110m 8.49GB (Phase 0 Smoke)
PRESET_RESERVED_MIB = {"pilot-42m": 6626, "main-110m": 8700}
PRESET_SHAPES = {"pilot-42m": (8, 512, 1408), "main-110m": (12, 768, 2048)}  # (n_layer, n_embd, ffn_dim)
VOCAB, BLOCK = 32768, 1024


def params(preset: str) -> int:
    n_layer, d, ffn = PRESET_SHAPES[preset]
    return VOCAB * d + n_layer * (4 * d * d + 3 * d * ffn + 2 * d) + d  # 가중치 공유, bias 없음, RMSNorm


def checkpoint_bytes(preset: str) -> int:
    """fp32 가중치 + AdamW 상태 2개 ≈ 파라미터당 12바이트 (pilot-42m 실측 509,796,457 바이트와 0.02% 차)"""
    return 12 * params(preset)


def parse_train_args(train_args: list[str], out_dir: Path) -> dict:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--model")
    p.add_argument("--resume", type=Path)
    p.add_argument("--micro-batch", type=int, default=6)
    p.add_argument("--grad-accum", type=int, default=32)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--decay-steps", type=int, default=0)
    p.add_argument("--milestone-tokens", type=int, default=0)
    a, _ = p.parse_known_args(train_args)
    model, source = a.model, "인자"
    if model is None:
        cmd = out_dir / "command.txt"
        if cmd.exists():
            words = cmd.read_text(encoding="utf-8").split()
            if "--model" in words:
                model, source = words[words.index("--model") + 1], "command.txt"
    if model not in PRESET_SHAPES:
        model, source = "main-110m", "알 수 없어 가장 큰 프리셋으로 가정"
    total_tokens = a.max_steps * a.micro_batch * a.grad_accum * BLOCK
    milestones = total_tokens // a.milestone_tokens if a.milestone_tokens > 0 else 0
    return {"model": model, "model_source": source, "milestones": milestones, "pre_decay": a.decay_steps > 0,
            "resume": a.resume is not None}


def running_wsl_distros() -> list[str] | None:
    try:
        raw = subprocess.run(["wsl.exe", "-l", "--running", "-q"], capture_output=True, timeout=20).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    text = raw.decode("utf-16-le", errors="ignore") if b"\x00" in raw else raw.decode(errors="ignore")
    return [line.strip() for line in text.replace("\r", "").splitlines() if line.strip()]


def other_training_pids() -> list[int]:
    """실행 중인 train.py (python 프로세스만. 명령줄에 문자열만 들어간 셸은 제외)"""
    pids = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            comm = (proc / "comm").read_text().strip()
            argv = (proc / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        if comm.startswith("python") and any(a.endswith(b"train.py") for a in argv):
            pids.append(int(proc.name))
    return pids


def evaluate(m: dict, info: dict, limits: dict) -> list[dict]:
    """측정값 m과 학습 정보 info로 항목별 판정. 측정 실패는 실패로 본다 (모르면 시작하지 않는다)"""
    checks = []

    def add(name, ok, value, need, note=""):
        checks.append({"check": name, "ok": bool(ok), "value": value, "need": need, "note": note})

    win = m.get("win_avail_mb")
    add("Windows 사용 가능 RAM (MB)", win is not None and win >= limits["min_win_avail_mb"], win,
        f">= {limits['min_win_avail_mb']}", "" if win is not None else "측정 실패")
    wsl = m.get("wsl_avail_mb")
    add("WSL MemAvailable (MB)", wsl is not None and wsl >= limits["min_wsl_avail_mb"], wsl,
        f">= {limits['min_wsl_avail_mb']}")
    distros = m.get("wsl_distros")
    own = m.get("own_distro")
    others = None if distros is None else [d for d in distros if d != own]
    add("다른 WSL 배포판 실행 중", others == [], others, "없음",
        "Docker Desktop 등은 같은 WSL VM 메모리를 쓴다" if others else ("" if others is not None else "측정 실패"))
    used, total = m.get("gpu_used_mb"), m.get("gpu_total_mb")
    need_vram = PRESET_RESERVED_MIB[info["model"]] + CUDA_CONTEXT_MIB + GPU_MARGIN_MB
    free_vram = None if used is None or total is None else total - used
    add("GPU 여유 VRAM (MiB)", free_vram is not None and free_vram >= need_vram, free_vram, f">= {need_vram}",
        f"{info['model']} reserved {PRESET_RESERVED_MIB[info['model']]} + CUDA 문맥 {CUDA_CONTEXT_MIB} "
        f"+ 감시 여유 {GPU_MARGIN_MB}")
    ck = checkpoint_bytes(info["model"])
    n_files = 2 + info["milestones"] + (1 if info["pre_decay"] else 0)  # last.pt + 저장 중 임시 파일 + 마일스톤 + 감쇠 직전
    need_disk = n_files * ck / 1e9 + limits["disk_margin_gb"]
    disk = m.get("disk_free_gb")
    add("디스크 여유 (GB)", disk is not None and disk >= need_disk, None if disk is None else round(disk, 1),
        f">= {need_disk:.1f}", f"Checkpoint {ck / 1e9:.2f}GB × {n_files}개 + 여유 {limits['disk_margin_gb']}GB")
    pids = m.get("training_pids", [])
    add("다른 학습 프로세스", not pids, pids, "없음")
    return checks


def measure(out_dir: Path) -> dict:
    m = {"win_avail_mb": windows_available_mb(), **wsl_memory_mb(), **gpu(),
         "wsl_distros": running_wsl_distros(), "own_distro": os.environ.get("WSL_DISTRO_NAME"),
         "training_pids": other_training_pids()}
    target = out_dir if out_dir.exists() else next(p for p in [out_dir, *out_dir.parents] if p.exists())
    m["disk_free_gb"] = shutil.disk_usage(target).free / 1e9
    return m


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--min-win-avail-mb", type=int, default=MIN_WIN_AVAIL_MB)
    p.add_argument("--min-wsl-avail-mb", type=int, default=MIN_WSL_AVAIL_MB)
    p.add_argument("--disk-margin-gb", type=float, default=DISK_MARGIN_GB)
    p.add_argument("train_args", nargs=argparse.REMAINDER)
    a = p.parse_args(argv)
    train_args = a.train_args[1:] if a.train_args[:1] == ["--"] else a.train_args
    info = parse_train_args(train_args, a.out_dir)
    m = measure(a.out_dir)
    limits = {"min_win_avail_mb": a.min_win_avail_mb, "min_wsl_avail_mb": a.min_wsl_avail_mb,
              "disk_margin_gb": a.disk_margin_gb}
    checks = evaluate(m, info, limits)
    ok = all(c["ok"] for c in checks)
    print(f"[{time.strftime('%F %T')}] 시작 전 점검 ({info['model']}, 모델 판단: {info['model_source']}, "
          f"마일스톤 {info['milestones']}개)")
    for c in checks:
        print(f"  {'통과' if c['ok'] else '실패'}  {c['check']}: {c['value']} (기준 {c['need']}) {c['note']}".rstrip())
    record = {"time": time.strftime("%F %T"), "ok": ok, "info": info, "limits": limits, "checks": checks,
              "swap_used_mb": m.get("swap_used_mb")}
    if not ok:
        top = windows_top_processes()
        record["windows_top_processes"] = top
        print("  Windows 상위 프로세스 (MB):")
        for r in top or []:
            print(f"    {r['mb']:>7,}  {r['name']} ×{r['count']}")
        print("시작하지 않는다. 위 항목을 사람이 정리한 뒤 다시 실행한다.")
    a.out_dir.mkdir(parents=True, exist_ok=True)
    with open(a.out_dir / "preflight.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return 0 if ok else EXIT_PREFLIGHT


if __name__ == "__main__":
    sys.exit(main())
