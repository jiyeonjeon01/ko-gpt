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
    assert failed({**GOOD, "wsl_distros": ["Ubuntu-24.04", "docker-desktop"]}) == ["다른 WSL 배포판 실행 중"]
    assert failed({**GOOD, "wsl_distros": None}) == ["다른 WSL 배포판 실행 중"]
    assert failed({**GOOD, "gpu_used_mb": 3000}) == ["GPU 여유 VRAM (MiB)"]  # 110M: 9,512 필요
    assert failed({**GOOD, "gpu_used_mb": 3000}, {**INFO, "model": "pilot-42m"}) == []
    assert failed({**GOOD, "disk_free_gb": 20.0}) == ["디스크 여유 (GB)"]
    assert failed({**GOOD, "training_pids": [123]}) == ["다른 학습 프로세스"]


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
    monkeypatch.setattr(pf, "measure", lambda out: {**GOOD, "wsl_distros": ["Ubuntu-24.04", "docker-desktop"]})
    monkeypatch.setattr(pf, "windows_top_processes", lambda n=15: [{"name": "chrome", "mb": 1000, "count": 3}])
    assert pf.main(["--out-dir", str(tmp_path), "--", "--model", "pilot-42m"]) == pf.EXIT_PREFLIGHT
    monkeypatch.setattr(pf, "measure", lambda out: GOOD)
    assert pf.main(["--out-dir", str(tmp_path), "--", "--model", "pilot-42m"]) == 0
    lines = (tmp_path / "preflight.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2 and '"ok": false' in lines[0] and "chrome" in lines[0] and '"ok": true' in lines[1]
