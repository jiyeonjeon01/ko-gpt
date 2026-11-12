"""시작 전 점검: 기준 판정, 학습 인자 해석, Checkpoint 크기 추정"""

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import preflight as pf

GOOD = {"win_avail_mb": 8000, "wsl_avail_mb": 8000, "wsl_distros": ["Ubuntu-24.04"], "own_distro": "Ubuntu-24.04",
        "gpu_used_mb": 1200, "gpu_total_mb": 12288, "disk_free_gb": 900.0, "training_pids": []}
LIMITS = {"min_win_avail_mb": 6144, "min_wsl_avail_mb": 4096, "disk_margin_gb": 2.0}
INFO = {"model": "main-110m", "model_source": "인자", "milestones": 14, "pre_decay": True, "resume": False}


def failed(m, info=INFO):
    return [c["check"] for c in pf.evaluate(m, info, LIMITS) if not c["ok"]]


def test_all_pass():
    assert failed(GOOD) == []


def test_each_failure():
    assert failed({**GOOD, "win_avail_mb": 5000}) == ["Windows 사용 가능 RAM (MB)"]
    assert failed({**GOOD, "win_avail_mb": None}) == ["Windows 사용 가능 RAM (MB)"]  # 측정 실패도 실패
    assert failed({**GOOD, "wsl_avail_mb": 3000}) == ["WSL MemAvailable (MB)"]
    assert failed({**GOOD, "gpu_used_mb": 3000}) == ["GPU 여유 VRAM (MiB)"]  # 110M: 9,512 필요
    assert failed({**GOOD, "gpu_used_mb": 3000}, {**INFO, "model": "pilot-42m"}) == []
    assert failed({**GOOD, "disk_free_gb": 20.0}) == ["디스크 여유 (GB)"]
    assert failed({**GOOD, "training_pids": [123]}) == ["이 프로젝트의 다른 학습 프로세스"]


def test_checkpoint_size_estimate_matches_measured_pilot():
    measured = 509_796_457  # runs/phase1/pilot42m/last.pt
    assert abs(pf.checkpoint_bytes("pilot-42m") - measured) / measured < 0.001
    assert 110_000_000 < pf.params("main-110m") < 110_200_000


def test_parse_train_args(tmp_path):
    a = pf.parse_train_args(["--model", "main-110m", "--max-steps", "100", "--micro-batch", "6", "--grad-accum", "42",
                             "--milestone-tokens", "10000000", "--decay-steps", "20"], tmp_path)
    assert a["model"] == "main-110m" and a["milestones"] == 100 * 6 * 42 * 1024 // 10_000_000 and a["pre_decay"]
    (tmp_path / "command.txt").write_text("scripts/run_monitored.sh x --model pilot-42m --data d\n", encoding="utf-8")
    b = pf.parse_train_args(["--resume", str(tmp_path / "last.pt")], tmp_path)
    assert b["model"] == "pilot-42m" and b["model_source"] == "command.txt" and b["resume"]
    c = pf.parse_train_args(["--resume", "x.pt"], tmp_path / "none")
    assert c["model"] == "main-110m" and "가장 큰" in c["model_source"]


def test_running_wsl_distros_decodes_utf16(monkeypatch):
    raw = "Ubuntu-24.04\r\ndocker-desktop\r\n".encode("utf-16-le")
    monkeypatch.setattr(pf.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout=raw))
    assert pf.running_wsl_distros() == ["Ubuntu-24.04", "docker-desktop"]


def test_main_writes_record_and_exit_code(tmp_path, monkeypatch):
    monkeypatch.setattr(pf, "measure", lambda out: {**GOOD, "win_avail_mb": 5000})
    monkeypatch.setattr(pf, "windows_top_processes", lambda n=15: [{"name": "chrome", "mb": 1000, "count": 3}])
    assert pf.main(["--out-dir", str(tmp_path), "--", "--model", "pilot-42m"]) == pf.EXIT_PREFLIGHT
    monkeypatch.setattr(pf, "measure", lambda out: GOOD)
    assert pf.main(["--out-dir", str(tmp_path), "--", "--model", "pilot-42m"]) == 0
    lines = (tmp_path / "preflight.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2 and '"ok": false' in lines[0] and "chrome" in lines[0] and '"ok": true' in lines[1]


def test_docker_desktop_alone_does_not_block():
    """docker-desktop 등 다른 배포판이 실행 중이어도 실제 여유가 충분하면 통과 (기록만)"""
    m = {**GOOD, "wsl_distros": ["Ubuntu-24.04", "docker-desktop"]}
    assert failed(m) == []
    row = next(c for c in pf.evaluate(m, INFO, LIMITS) if c["check"].startswith("다른 WSL 배포판"))
    assert row["ok"] and row["value"] == ["docker-desktop"]
    assert failed({**GOOD, "wsl_distros": None}) == []  # 배포판 목록 측정 실패도 차단 사유가 아니다


def test_docker_memory_use_still_counts_through_real_availability():
    """Docker가 메모리를 많이 써서 실제 여유가 부족하면 거부한다 (기준은 그대로)"""
    m = {**GOOD, "wsl_distros": ["Ubuntu-24.04", "docker-desktop"], "win_avail_mb": 6143, "wsl_avail_mb": 4095}
    assert failed(m) == ["Windows 사용 가능 RAM (MB)", "WSL MemAvailable (MB)"]
    assert failed({**GOOD, "gpu_used_mb": None}) == ["GPU 여유 VRAM (MiB)"]  # 측정 실패는 여전히 거부
    assert failed({**GOOD, "disk_free_gb": None}) == ["디스크 여유 (GB)"]


def test_thresholds_not_relaxed():
    assert pf.MIN_WIN_AVAIL_MB == 6144 and pf.MIN_WSL_AVAIL_MB == 4096 and pf.DISK_MARGIN_GB == 2.0
    import monitor
    assert monitor.WIN_AVAIL_STOP_MB == 1024 and monitor.WIN_AVAIL_WARN_MB == 2048 and monitor.PERSIST == 4


def test_refusal_never_kills_or_changes_other_programs(tmp_path, monkeypatch):
    """거부할 때 실행하는 외부 명령은 읽기 전용 측정뿐이다 (wsl --shutdown, docker, taskkill 등 없음)"""
    calls = []
    real_run = subprocess.run

    def spy(cmd, *a, **k):
        calls.append(cmd)
        return real_run(["true"], capture_output=True)
    monkeypatch.setattr(pf.subprocess, "run", spy)
    monkeypatch.setattr(pf, "measure", lambda out: {**GOOD, "wsl_distros": ["Ubuntu-24.04", "docker-desktop"], "win_avail_mb": 3000})
    monkeypatch.setattr(pf, "windows_top_processes", lambda n=15: [])
    assert pf.main(["--out-dir", str(tmp_path), "--", "--model", "pilot-42m"]) == pf.EXIT_PREFLIGHT
    flat = " ".join(" ".join(map(str, c)) if isinstance(c, (list, tuple)) else str(c) for c in calls).lower()
    for bad in ("shutdown", "--terminate", "taskkill", "stop-process", "docker", "kill"):
        assert bad not in flat
    rec = (tmp_path / "preflight.jsonl").read_text(encoding="utf-8")
    assert "Windows 사용 가능 RAM" in rec and '"refused"' in rec
    src = Path(pf.__file__).read_text(encoding="utf-8")
    for bad in ('"--shutdown"', '"--terminate"', "taskkill", "Stop-Process", "os.kill"):
        assert bad not in src


def test_other_project_train_py_is_not_counted(tmp_path):
    assert not pf._in_project(tmp_path, [b"python", b"/home/x/other-project/train.py"])
    assert pf._in_project(tmp_path, [b"python", str(pf.ROOT / "sft_train.py").encode()])


def test_default_win_ram_limit_stays_6144():
    # 기본 기준은 낮아지지 않는다 (2026-11-12 사용자 지시)
    assert pf.MIN_WIN_AVAIL_MB == 6144 and pf.DEFAULT_LIMITS["min_win_avail_mb"] == 6144


def test_changed_limit_without_approval_is_refused(tmp_path, monkeypatch):
    def no_measure(out):
        raise AssertionError("승인 없는 기준 변경은 측정 전에 거부해야 한다")
    monkeypatch.setattr(pf, "measure", no_measure)
    assert pf.main(["--out-dir", str(tmp_path), "--min-win-avail-mb", "4608", "--", "--model", "pilot-42m"]) == pf.EXIT_USAGE
    assert pf.main(["--out-dir", str(tmp_path), "--min-win-avail-mb", "4608", "--approval", "  ",
                    "--", "--model", "pilot-42m"]) == pf.EXIT_USAGE
    assert not (tmp_path / "preflight.jsonl").exists()


def test_changed_limit_with_approval_is_recorded(tmp_path, monkeypatch):
    import json
    monkeypatch.setattr(pf, "measure", lambda out: {**GOOD, "win_avail_mb": 5000})
    monkeypatch.setattr(pf, "windows_top_processes", lambda n=15: [])
    assert pf.main(["--out-dir", str(tmp_path), "--min-win-avail-mb", "4608", "--approval", "사용자 승인 (시험)",
                    "--", "--model", "pilot-42m"]) == 0
    rec = json.loads((tmp_path / "preflight.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    assert rec["limits_changed"] == {"min_win_avail_mb": [6144, 4608]} and rec["approval"] == "사용자 승인 (시험)"
    # 기본 기준이면 승인 없이도 되고 limits_changed가 없다
    assert pf.main(["--out-dir", str(tmp_path), "--", "--model", "pilot-42m"]) == pf.EXIT_PREFLIGHT
    rec = json.loads((tmp_path / "preflight.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    assert "limits_changed" not in rec and rec["limits"]["min_win_avail_mb"] == 6144


def test_run_monitored_refuses_changed_limit_without_approval(tmp_path):
    root = Path(__file__).resolve().parent.parent
    env = {"PATH": "/usr/bin:/bin", "PREFLIGHT_ARGS": "--min-win-avail-mb 4608"}
    r = subprocess.run(["bash", str(root / "scripts/run_monitored.sh"), str(tmp_path / "out"), "--model", "pilot-42m"],
                       env=env, capture_output=True, text=True, timeout=60)
    assert r.returncode == 2 and "승인 기록" in r.stdout
    assert "PREFLIGHT_APPROVAL" in (tmp_path / "out" / "command.txt").read_text(encoding="utf-8")
