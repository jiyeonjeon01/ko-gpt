"""학습 중 자원 감시와 안전 정지 요청 (Phase 1 사전 점검)

30초마다 기록 (<out-dir>/monitor.csv):
    Windows 사용 가능 RAM, WSL 사용·캐시·swap, GPU 메모리·온도·전력·사용률, 학습 Step·처리량
위험이 **지속될 때만** <out-dir>/STOP 파일에 사유를 쓴다. train.py가 다음 Step 경계에서 last.pt를 저장하고 멈춘다.
학습 설정을 바꾸거나 데이터를 건너뛰지 않는다. 기록하고 멈추게 할 뿐이다.

중단 조건 (한 번 튄 값으로는 멈추지 않는다):
    - Windows 사용 가능 RAM < 1,024MB가 연속 4회 (약 2분)
    - WSL swap 사용량이 최근 10회(약 5분) 동안 1,024MB 넘게 증가, 또는 4,096MB 초과가 연속 2회
    - GPU 사용량 > 물리 용량 - 300MB가 연속 4회
    - GPU 온도 ≥ 90°C가 연속 4회
경고만 (중단 안 함): Windows 사용 가능 RAM < 2,048MB, GPU 온도 ≥ 85°C

    python3 scripts/monitor.py --out-dir runs/pilot --pid <train.py PID>
"""

import argparse
import csv
import os
import subprocess
import time
from collections import deque
from pathlib import Path

WIN_AVAIL_STOP_MB, WIN_AVAIL_WARN_MB = 1024, 2048
SWAP_GROWTH_MB, SWAP_ABS_MB = 1024, 4096
GPU_MARGIN_MB = 300
GPU_TEMP_STOP, GPU_TEMP_WARN = 90, 85
PERSIST = 4  # 연속 표본 수


def windows_available_mb() -> int | None:
    try:
        out = subprocess.run(
            ["powershell.exe", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_PerfFormattedData_PerfOS_Memory).AvailableMBytes"],
            capture_output=True, text=True, timeout=20).stdout.strip()
        return int(out)
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None


def wsl_memory_mb() -> dict:
    info = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        info[key] = int(value.split()[0]) // 1024
    return {"wsl_used_mb": info["MemTotal"] - info["MemAvailable"], "wsl_cache_mb": info["Cached"],
            "wsl_avail_mb": info["MemAvailable"], "swap_used_mb": info["SwapTotal"] - info["SwapFree"]}


def gpu() -> dict:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total,temperature.gpu,power.draw,"
                              "utilization.gpu", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20).stdout.strip()
        used, total, temp, power, util = (float(x) for x in out.split(","))
        return {"gpu_used_mb": int(used), "gpu_total_mb": int(total), "gpu_temp_c": int(temp),
                "gpu_power_w": round(power, 1), "gpu_util_pct": int(util)}
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return {}


def train_progress(out_dir: Path) -> dict:
    path = out_dir / "steps.csv"
    if not path.exists():
        return {}
    try:
        with open(path, newline="", encoding="utf-8") as f:
            last = None
            for last in csv.DictReader(f):
                pass
        return {"step": last["step"], "tokens_per_s": last["tokens_per_s"], "loss": last["loss"]} if last else {}
    except (OSError, KeyError, csv.Error):
        return {}


class Judge:
    """표본을 받아 지속되는 위험이면 중단 사유를, 아니면 None을 돌려준다."""

    def __init__(self):
        self.low_ram = self.gpu_full = self.hot = self.swap_high = 0
        self.swap_hist: deque[int] = deque(maxlen=10)

    def update(self, s: dict) -> tuple[str | None, list[str]]:
        warnings = []
        avail = s.get("win_avail_mb")
        self.low_ram = self.low_ram + 1 if avail is not None and avail < WIN_AVAIL_STOP_MB else 0
        if avail is not None and avail < WIN_AVAIL_WARN_MB:
            warnings.append(f"Windows 사용 가능 RAM {avail}MB")
        swap = s["swap_used_mb"]
        self.swap_hist.append(swap)
        self.swap_high = self.swap_high + 1 if swap > SWAP_ABS_MB else 0
        growth = swap - min(self.swap_hist)
        if "gpu_used_mb" in s:
            self.gpu_full = self.gpu_full + 1 if s["gpu_used_mb"] > s["gpu_total_mb"] - GPU_MARGIN_MB else 0
            self.hot = self.hot + 1 if s["gpu_temp_c"] >= GPU_TEMP_STOP else 0
            if s["gpu_temp_c"] >= GPU_TEMP_WARN:
                warnings.append(f"GPU 온도 {s['gpu_temp_c']}°C")
        if self.low_ram >= PERSIST:
            return f"Windows 사용 가능 RAM < {WIN_AVAIL_STOP_MB}MB가 {self.low_ram}회 연속 (최근 {avail}MB)", warnings
        if len(self.swap_hist) == self.swap_hist.maxlen and growth > SWAP_GROWTH_MB:
            return f"WSL swap이 최근 {len(self.swap_hist)}회 동안 {growth}MB 증가 (현재 {swap}MB)", warnings
        if self.swap_high >= 2:
            return f"WSL swap {swap}MB > {SWAP_ABS_MB}MB가 {self.swap_high}회 연속", warnings
        if self.gpu_full >= PERSIST:
            return f"GPU 메모리 {s['gpu_used_mb']}/{s['gpu_total_mb']}MB (여유 {GPU_MARGIN_MB}MB 미만)가 {self.gpu_full}회 연속", warnings
        if self.hot >= PERSIST:
            return f"GPU 온도 ≥ {GPU_TEMP_STOP}°C가 {self.hot}회 연속 ({s['gpu_temp_c']}°C)", warnings
        return None, warnings


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--pid", type=int, required=True, help="감시할 train.py 프로세스 (끝나면 감시도 끝남)")
    p.add_argument("--interval", type=float, default=30.0)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    log = args.out_dir / "monitor.csv"
    judge, requested = Judge(), False
    fields = ["time", "win_avail_mb", "wsl_used_mb", "wsl_cache_mb", "wsl_avail_mb", "swap_used_mb", "gpu_used_mb",
              "gpu_total_mb", "gpu_temp_c", "gpu_power_w", "gpu_util_pct", "step", "tokens_per_s", "loss",
              "warnings", "stop_reason"]
    new = not log.exists()
    with open(log, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if new:
            w.writeheader()
        while alive(args.pid):
            s = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "win_avail_mb": windows_available_mb(),
                 **wsl_memory_mb(), **gpu(), **train_progress(args.out_dir)}
            reason, warnings = judge.update(s)
            s["warnings"] = "; ".join(warnings)
            if reason and not requested:
                (args.out_dir / "STOP").write_text(f"[monitor {s['time']}] {reason}\n", encoding="utf-8")
                s["stop_reason"] = reason
                requested = True
                print(f"안전 정지 요청: {reason}", flush=True)
            w.writerow(s)
            f.flush()
            time.sleep(args.interval)
    print("감시 종료 (학습 프로세스 종료)", flush=True)


if __name__ == "__main__":
    main()
