"""그래프 스크립트: 축이 데이터를 덮고, 재개·측정 실패 기록을 견디고, 올바른 SVG를 만든다"""

import csv
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from monitor import WIN_AVAIL_STOP_MB
from plot_run import fmt, latest_per_step, line_chart, nice_ticks, plot_run, plot_sweep

SVG = "{http://www.w3.org/2000/svg}"
SOURCES = ["P1_fineweb2_ko", "P2_wikipedia_ko", "P3_finepdfs_ko", "P4_fineweb_edu"]


@pytest.mark.parametrize("lo,hi", [(0, 1), (1.13, 2.33), (0.28, 1.01), (3.1, 10.5), (0, 1e-3), (-2.5, 3.7), (5, 5)])
def test_ticks_cover_range(lo, hi):
    t = nice_ticks(lo, hi)
    assert t[0] <= lo and t[-1] >= hi
    assert len(t) >= 2
    steps = {round(b - a, 9) for a, b in zip(t, t[1:])}
    assert len(steps) == 1  # 등간격


def test_fmt():
    assert fmt(12288) == "12,288"
    assert fmt(0) == "0"
    assert fmt(1.5) == "1.5"
    assert fmt(2.5e-4) == "2.5e-04"


def test_latest_per_step_keeps_rerun_after_resume():
    rows = [{"step": "1", "loss": "9"}, {"step": "2", "loss": "8"}, {"step": "3", "loss": "7"},
            {"step": "2", "loss": "8.5"}, {"step": "3", "loss": "7.5"}, {"step": "4", "loss": "7"}]
    out = latest_per_step(rows)
    assert [r["step"] for r in out] == ["1", "2", "3", "4"]
    assert [r["loss"] for r in out] == ["9", "8.5", "7.5", "7"]


def _points_inside(svg: Path) -> None:
    """모든 선 꼭짓점이 축 눈금 범위(격자선 사이) 안에 있어야 한다"""
    root = ET.parse(svg).getroot()
    grid_y = [float(e.get("y1")) for e in root.iter(f"{SVG}line") if e.get("stroke") == "#e1e0d9"]
    top, bottom = min(grid_y), max(grid_y)
    for path in root.iter(f"{SVG}path"):
        for seg in path.get("d").replace("M", " ").replace("L", " ").split():
            y = float(seg.split(",")[1])
            assert top - 0.11 <= y <= bottom + 0.11, (svg.name, y, top, bottom)


def test_line_chart_valid_svg_and_escaped(tmp_path):
    out = tmp_path / "a.svg"
    line_chart(out, "제목 <&>", "부제", "x<", "y", [("a&b", [(0, 1.0), (1, 2.0)]), ("c", [(0, 0.5), (1, 0.7)])],
               markers=True, hline=(1.5, "기준"))
    root = ET.parse(out).getroot()  # 이스케이프가 틀리면 여기서 실패
    titles = [e.text for e in root.iter(f"{SVG}title")]
    assert any("a&b" in t and "x<" in t for t in titles)
    _points_inside(out)


def test_line_chart_negative_with_y_zero(tmp_path):
    out = tmp_path / "neg.svg"
    line_chart(out, "t", "s", "x", "y", [("v", [(0, -3.0), (1, 2.0)])], y_zero=True)
    _points_inside(out)


def test_line_chart_drops_empty_series_and_rejects_all_empty(tmp_path):
    line_chart(tmp_path / "ok.svg", "t", "s", "x", "y", [("빈", []), ("v", [(0, 1.0), (1, 2.0)])])
    with pytest.raises(ValueError):
        line_chart(tmp_path / "bad.svg", "t", "s", "x", "y", [("빈", [])])


def _write(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def _fake_run(run: Path, steps: int = 40, resume_at: int | None = None, blank_monitor: bool = False) -> None:
    run.mkdir(parents=True)
    srows = [{"step": s, "loss": 10 - s * 0.1, "grad_norm": 1.5 if s < 3 else 0.4, "lr": min(s, 5) * 2e-4,
              "tokens_seen": s * 1000, "tokens_per_s": 40000, "resume": 0} for s in range(1, steps + 1)]
    if resume_at:
        srows += [{**r, "loss": r["loss"] - 0.01, "resume": 1} for r in srows if r["step"] > resume_at]
    _write(run / "steps.csv", srows)
    erows = [{"step": s, "tokens_seen": s * 1000, **{f"{src}_bpb": 2 - s * 0.01 * (i + 1) for i, src in enumerate(SOURCES)}}
             for s in range(10, steps + 1, 10)]
    _write(run / "eval.csv", erows)
    mrows = [{"time": f"2026-11-06 13:{m:02d}:00", "win_avail_mb": "" if blank_monitor and m == 3 else 3000 - m * 10,
              "swap_used_mb": 193, "gpu_used_mb": 8000, "gpu_total_mb": 12288, "gpu_temp_c": 58} for m in range(10)]
    _write(run / "monitor.csv", mrows)


def test_plot_run_end_to_end(tmp_path):
    run = tmp_path / "pilot"
    _fake_run(run, resume_at=25, blank_monitor=True)
    made = plot_run(run, tmp_path / "plots")
    assert set(made) == {"train_loss.svg", "val_bpb_korean.svg", "val_bpb_other.svg", "lr.svg", "grad_norm.svg",
                         "tokens_per_s.svg", "windows_ram.svg", "gpu_memory.svg", "gpu_temp.svg", "wsl_memory.svg"}
    for name in made:
        _points_inside(tmp_path / "plots" / name)
    ram = (tmp_path / "plots" / "windows_ram.svg").read_text(encoding="utf-8")
    assert f"{WIN_AVAIL_STOP_MB:,}MB" in ram  # 정지 기준은 monitor.py 값을 그대로 쓴다
    # 재개 후 중복 step은 한 번만 그린다: Step별 Loss 선의 꼭짓점 수 = 고유 step 수
    root = ET.parse(tmp_path / "plots" / "train_loss.svg").getroot()
    first_path = next(root.iter(f"{SVG}path")).get("d")
    assert first_path.count("L") + 1 == 40


def test_plot_sweep(tmp_path):
    runs = []
    for lr in ("5e-4", "1e-3"):
        _fake_run(tmp_path / f"lr_{lr}")
        runs.append(tmp_path / f"lr_{lr}")
    made = plot_sweep(runs, tmp_path / "plots")
    for name in made:
        _points_inside(tmp_path / "plots" / name)
