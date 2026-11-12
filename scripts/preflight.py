"""학습 시작 전 자동 점검: Windows RAM, WSL 메모리, GPU VRAM, 디스크, 이 프로젝트의 다른 학습 프로세스

    python3 scripts/preflight.py --out-dir runs/x -- --model main-110m --max-steps 11600 ...

통과하면 종료 코드 0, 하나라도 실패하면 8. 결과는 <out-dir>/preflight.jsonl에 덧붙인다.
실패해도 프로그램을 닫거나 설정을 바꾸지 않는다. GPT 학습만 시작하지 않고 이유를 남긴다.
학습 중 안전 정지 기준(monitor.py)과는 별개다.

다른 프로젝트와 함께 쓰는 원칙 (2026-11-09):
- Docker Desktop(docker-desktop 배포판)과 다른 프로젝트의 프로세스는 그대로 둔다. 종료·중단하지 않고 `wsl --shutdown`도 하지 않는다.
- 다른 WSL 배포판이 실행 중이라는 사실만으로는 차단하지 않는다 (기록만 한다).
  - WSL2 배포판들은 하나의 VM을 같이 쓴다. Docker가 쓰는 메모리는 이미 WSL MemAvailable과 Windows 사용 가능 RAM에 반영되어 있다.
  - GPU를 쓰는 다른 프로그램의 몫은 nvidia-smi 사용량에 반영되어 있다.
- 그래서 지금 실행 중인 다른 프로젝트의 자원 사용을 그대로 둔 상태에서 실제 여유(RAM, WSL 메모리, VRAM, 디스크)로 판단한다.
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


ROOT = Path(__file__).resolve().parents[1]


def _in_project(proc: Path, argv: list[bytes]) -> bool:
    """이 저장소의 프로세스인가 (작업 폴더 또는 스크립트 경로가 저장소 안). 다른 프로젝트의 train.py는 세지 않는다"""
    try:
        cwd = Path(os.readlink(proc / "cwd"))
    except OSError:
        cwd = None
    if cwd is not None and (cwd == ROOT or ROOT in cwd.parents):
        return True
    return any(a.endswith(b"train.py") and (Path(a.decode(errors="ignore")).is_absolute()
                                             and ROOT in Path(a.decode(errors="ignore")).parents) for a in argv)


def other_training_pids() -> list[int]:
    """이 저장소에서 실행 중인 train.py·sft_train.py (python 프로세스만. 명령줄에 문자열만 들어간 셸은 제외).
    다른 프로젝트의 학습은 막지 않는다. 그 GPU·RAM 사용량은 VRAM·RAM 점검에 반영된다"""
    pids = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            comm = (proc / "comm").read_text().strip()
            argv = (proc / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        if comm.startswith("python") and any(a.endswith(b"train.py") for a in argv) and _in_project(proc, argv):
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
    add("다른 WSL 배포판 (기록만, 차단 안 함)", True, others, "제한 없음",
        "그대로 둔다. 쓰는 메모리는 위 RAM·WSL 점검에 이미 반영됨" if others else ("" if others is not None else "측정 실패 (기록용)"))
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
    add("이 프로젝트의 다른 학습 프로세스", not pids, pids, "없음")
    return checks


def measure(out_dir: Path) -> dict:
    m = {"win_avail_mb": windows_available_mb(), **wsl_memory_mb(), **gpu(),
         "wsl_distros": running_wsl_distros(), "own_distro": os.environ.get("WSL_DISTRO_NAME"),
         "training_pids": other_training_pids()}
    target = out_dir if out_dir.exists() else next(p for p in [out_dir, *out_dir.parents] if p.exists())
    m["disk_free_gb"] = shutil.disk_usage(target).free / 1e9
    return m


EXIT_USAGE = 2
DEFAULT_LIMITS = {"min_win_avail_mb": MIN_WIN_AVAIL_MB, "min_wsl_avail_mb": MIN_WSL_AVAIL_MB, "disk_margin_gb": DISK_MARGIN_GB}


def changed_limits(a) -> dict:
    """기본값과 다른 기준만 {이름: [기본, 지금]}"""
    return {k: [d, getattr(a, k)] for k, d in DEFAULT_LIMITS.items() if getattr(a, k) != d}


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--min-win-avail-mb", type=int, default=MIN_WIN_AVAIL_MB)
    p.add_argument("--min-wsl-avail-mb", type=int, default=MIN_WSL_AVAIL_MB)
    p.add_argument("--disk-margin-gb", type=float, default=DISK_MARGIN_GB)
    p.add_argument("--approval", default="", help="기준을 기본값과 다르게 쓸 때 필수: 사용자 승인 내용 (기록용)")
    p.add_argument("train_args", nargs=argparse.REMAINDER)
    a = p.parse_args(argv)
    # 기준 변경은 사용자가 명시적으로 승인한 경우에만 쓴다. 승인 기록이 없으면 측정도 하지 않고 거부한다
    changed = changed_limits(a)
    if changed and not a.approval.strip():
        print(f"시작 전 점검 기준을 기본값과 다르게 주었지만 승인 기록(--approval)이 없다: {changed}. 학습을 시작하지 않는다")
        return EXIT_USAGE
    train_args = a.train_args[1:] if a.train_args[:1] == ["--"] else a.train_args
    info = parse_train_args(train_args, a.out_dir)
    m = measure(a.out_dir)
    limits = {"min_win_avail_mb": a.min_win_avail_mb, "min_wsl_avail_mb": a.min_wsl_avail_mb,
              "disk_margin_gb": a.disk_margin_gb}
    checks = evaluate(m, info, limits)
    ok = all(c["ok"] for c in checks)
    if changed:
        print(f"  기준 변경 (사용자 승인): {changed} / 승인: {a.approval.strip()}")
    print(f"[{time.strftime('%F %T')}] 시작 전 점검 ({info['model']}, 모델 판단: {info['model_source']}, "
          f"마일스톤 {info['milestones']}개)")
    for c in checks:
        print(f"  {'통과' if c['ok'] else '실패'}  {c['check']}: {c['value']} (기준 {c['need']}) {c['note']}".rstrip())
    record = {"time": time.strftime("%F %T"), "ok": ok, "info": info, "limits": limits, "checks": checks,
              "swap_used_mb": m.get("swap_used_mb")}
    if changed:
        record["limits_changed"] = changed
        record["approval"] = a.approval.strip()
    top = windows_top_processes()  # 읽기만 한다 (통과해도 기록: 다른 프로젝트가 쓰던 자원을 남긴다)
    record["windows_top_processes"] = top
    record["wsl_distros"] = m.get("wsl_distros")
    if not ok:
        record["refused"] = [c["check"] for c in checks if not c["ok"]]
        print("  Windows 상위 프로세스 (MB, 기록용):")
        for r in top or []:
            print(f"    {r['mb']:>7,}  {r['name']} ×{r['count']}")
        print("GPT 학습을 시작하지 않는다. 이유: " + ", ".join(record["refused"]))
        print("다른 프로그램(Docker Desktop, 다른 프로젝트, 다른 WSL 배포판)은 종료하거나 바꾸지 않는다.")
    a.out_dir.mkdir(parents=True, exist_ok=True)
    with open(a.out_dir / "preflight.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return 0 if ok else EXIT_PREFLIGHT


if __name__ == "__main__":
    sys.exit(main())
