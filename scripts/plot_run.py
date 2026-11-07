"""학습 기록(steps.csv, eval.csv, monitor.csv)을 의존성 없는 SVG 그래프로 그린다.

사용:
    python3 scripts/plot_run.py runs/phase1/pilot42m --out runs/phase1/pilot42m/plots
    python3 scripts/plot_run.py --sweep runs/phase1/lr_5e-4 runs/phase1/lr_1e-3 runs/phase1/lr_2e-3 \
        --out runs/phase1/plots

각 점에는 <title>이 있어 브라우저에서 마우스를 올리면 값이 보인다.
"""
import argparse
import csv
import math
from datetime import datetime
from html import escape
from pathlib import Path

from monitor import GPU_TEMP_STOP, PERSIST, SWAP_ABS_MB, SWAP_GROWTH_MB, WIN_AVAIL_STOP_MB

# 색상: 범주형 고정 순서 (dataviz 기본 팔레트, 밝은 배경용)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE, TEXT1, TEXT2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
FONT = "'Malgun Gothic','Apple SD Gothic Neo','Noto Sans KR',sans-serif"

KO_SOURCES = ["P1_fineweb2_ko", "P2_wikipedia_ko", "P3_finepdfs_ko"]
OTHER_SOURCES = ["P4_fineweb_edu", "P5_cosmopedia_v2", "P6_wikipedia_en", "P8a_opc_algorithmic", "P7_finemath_4plus"]
LABELS = {
    "P1_fineweb2_ko": "한국어 웹", "P2_wikipedia_ko": "한국어 위키", "P3_finepdfs_ko": "한국어 PDF",
    "P4_fineweb_edu": "영어 edu", "P5_cosmopedia_v2": "영어 합성교재", "P6_wikipedia_en": "영어 위키",
    "P8a_opc_algorithmic": "코드", "P7_finemath_4plus": "수학",
}


def nice_ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    if hi <= lo:
        hi = lo + 1
    raw = (hi - lo) / n
    mag = 10 ** math.floor(math.log10(raw))
    step = min((m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw), default=10 * mag)
    # 데이터 범위를 모두 덮도록 아래·위 모두 바깥쪽 눈금까지 넓힌다
    first = math.floor(lo / step + 1e-9)
    last = math.ceil(hi / step - 1e-9)
    return [round(k * step, 10) for k in range(first, last + 1)]


def fmt(v: float) -> str:
    if abs(v) >= 1000:
        return f"{v:,.0f}"
    if abs(v) >= 10:
        return f"{v:.0f}"
    if abs(v) >= 1:
        return f"{v:.2f}".rstrip("0").rstrip(".")
    if v == 0:
        return "0"
    if abs(v) < 0.01:
        return f"{v:.1e}"
    return f"{v:.3f}".rstrip("0").rstrip(".")


def line_chart(path: Path, title: str, subtitle: str, xlabel: str, ylabel: str,
               series: list[tuple[str, list[tuple[float, float]]]], *, markers: bool = False,
               y_zero: bool = False, hline: tuple[float, str] | None = None,
               colors: list[str] | None = None, value_fmt=fmt) -> None:
    W, H = 760, 420
    L, R, T, B = 72, 150 if len(series) > 1 else 40, 70, 56
    pw, ph = W - L - R, H - T - B
    colors = colors or SERIES
    series = [(name, pts) for name, pts in series if pts]
    if not series:
        raise ValueError(f"{path.name}: 그릴 값이 없다")
    xs = [x for _, pts in series for x, _ in pts]
    ys = [y for _, pts in series for _, y in pts]
    if hline:
        ys.append(hline[0])
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    pad = (y1 - y0) * 0.05 or 1
    # y_zero: 0을 축에 포함한다 (음수가 있으면 음수 쪽도 덮는다)
    lo = min(0.0, y0 - pad if y0 < 0 else 0.0) if y_zero else y0 - pad
    yt = nice_ticks(lo, y1 + pad)
    y0, y1 = yt[0], yt[-1]
    xt = [t for t in nice_ticks(x0, x1) if x0 <= t <= x1] or [x0, x1]

    def sx(x):
        return L + (x - x0) / ((x1 - x0) or 1) * pw

    def sy(y):
        return T + ph - (y - y0) / ((y1 - y0) or 1) * ph

    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
         f'font-family="{FONT}" role="img" aria-label="{escape(title)}">',
         f'<rect width="{W}" height="{H}" fill="{SURFACE}"/>',
         f'<text x="{L}" y="26" font-size="16" font-weight="600" fill="{TEXT1}">{escape(title)}</text>',
         f'<text x="{L}" y="46" font-size="12" fill="{TEXT2}">{escape(subtitle)}</text>']
    for t in yt:
        o.append(f'<line x1="{L}" x2="{L + pw}" y1="{sy(t):.1f}" y2="{sy(t):.1f}" stroke="{GRID}" stroke-width="1"/>')
        o.append(f'<text x="{L - 8}" y="{sy(t) + 4:.1f}" font-size="11" fill="{MUTED}" text-anchor="end">{fmt(t)}</text>')
    o.append(f'<line x1="{L}" x2="{L + pw}" y1="{T + ph}" y2="{T + ph}" stroke="{AXIS}" stroke-width="1"/>')
    for t in xt:
        o.append(f'<text x="{sx(t):.1f}" y="{T + ph + 18}" font-size="11" fill="{MUTED}" text-anchor="middle">{fmt(t)}</text>')
    o.append(f'<text x="{L + pw / 2}" y="{H - 12}" font-size="12" fill="{TEXT2}" text-anchor="middle">{escape(xlabel)}</text>')
    o.append(f'<text transform="translate(16,{T + ph / 2}) rotate(-90)" font-size="12" fill="{TEXT2}" '
             f'text-anchor="middle">{escape(ylabel)}</text>')
    if hline:
        hy = sy(hline[0])
        o.append(f'<line x1="{L}" x2="{L + pw}" y1="{hy:.1f}" y2="{hy:.1f}" stroke="{TEXT2}" stroke-width="1"/>')
        o.append(f'<text x="{L + pw - 4}" y="{hy - 6:.1f}" font-size="11" fill="{TEXT2}" text-anchor="end">'
                 f'{escape(hline[1])}</text>')

    ends = []
    for i, (name, pts) in enumerate(series):
        c = colors[i % len(colors)]
        d = " ".join(f"{'M' if j == 0 else 'L'}{sx(x):.1f},{sy(y):.1f}" for j, (x, y) in enumerate(pts))
        o.append(f'<path d="{d}" fill="none" stroke="{c}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>')
        # 호버용 점: 많으면 솎아서 최대 400개
        stride = max(1, len(pts) // 400)
        for j, (x, y) in enumerate(pts):
            if j % stride and j != len(pts) - 1:
                continue
            r = 4 if markers else 6
            vis = f'fill="{c}" stroke="{SURFACE}" stroke-width="2"' if markers else 'fill="transparent"'
            o.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="{r}" {vis}>'
                     f'<title>{escape(name)} · {escape(xlabel)} {fmt(x)} · {escape(value_fmt(y))}</title></circle>')
        ends.append((sy(pts[-1][1]), name, pts[-1][1], c))

    if len(series) > 1:
        # 범례(오른쪽, 선 끝 값 포함). 4개 이하면 선 끝에 가깝게 정렬
        ends_sorted = sorted(ends) if len(series) <= 4 else ends
        gap = 18
        ypos = []
        for k, (yy, *_r) in enumerate(ends_sorted):
            want = yy if len(series) <= 4 else T + 10 + k * gap
            ypos.append(max(want, (ypos[-1] + gap) if ypos else T))
        over = ypos[-1] - (T + ph) if ypos else 0
        if over > 0:
            ypos = [y - over for y in ypos]
        for (yy, name, val, c), ly in zip(ends_sorted, ypos):
            lx = L + pw + 10
            o.append(f'<line x1="{lx}" x2="{lx + 14}" y1="{ly:.1f}" y2="{ly:.1f}" stroke="{c}" stroke-width="3" stroke-linecap="round"/>')
            o.append(f'<text x="{lx + 20}" y="{ly + 4:.1f}" font-size="11" fill="{TEXT1}">{escape(name)} '
                     f'<tspan fill="{TEXT2}">{value_fmt(val)}</tspan></text>')
    o.append("</svg>")
    path.write_text("\n".join(o), encoding="utf-8")


def read_csv(p: Path) -> list[dict]:
    with p.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def latest_per_step(rows: list[dict]) -> list[dict]:
    """재개하면 Checkpoint 이후 step이 다시 기록된다 (resume 열로 구분). 같은 step은 마지막 기록만 남긴다."""
    by_step = {}
    for r in rows:
        by_step[int(r["step"])] = r
    return [by_step[k] for k in sorted(by_step)]


def num(v: str | None) -> float | None:
    """감시 값이 비어 있으면(측정 실패) None"""
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def points(xs: list[float], rows: list[dict], key: str) -> list[tuple[float, float]]:
    return [(x, y) for x, r in zip(xs, rows) if (y := num(r.get(key))) is not None]


def moving_avg(vals: list[float], k: int) -> list[float]:
    out, s = [], 0.0
    for i, v in enumerate(vals):
        s += v
        if i >= k:
            s -= vals[i - k]
        out.append(s / min(i + 1, k))
    return out


def plot_run(run: Path, out: Path) -> list[str]:
    out.mkdir(parents=True, exist_ok=True)
    name = run.name
    steps = latest_per_step(read_csv(run / "steps.csv"))
    ev = latest_per_step(read_csv(run / "eval.csv"))
    mon = read_csv(run / "monitor.csv") if (run / "monitor.csv").exists() else []
    st = [int(r["step"]) for r in steps]
    made = []

    loss = [float(r["loss"]) for r in steps]
    line_chart(out / "train_loss.svg", f"{name} · Train Loss", "Step마다 기록한 학습 Loss와 50 Step 이동 평균",
               "Step", "Loss (nats/token)",
               [("Step별", list(zip(st, loss))), ("50 Step 평균", list(zip(st, moving_avg(loss, 50))))],
               colors=["#c3c2b7", SERIES[0]])
    made.append("train_loss.svg")

    es = [int(r["step"]) for r in ev]
    for fname, srcs, label in [("val_bpb_korean.svg", KO_SOURCES, "한국어 출처"),
                               ("val_bpb_other.svg", OTHER_SOURCES, "영어·코드·수학 출처")]:
        ser = [(LABELS[s], points(es, ev, f"{s}_bpb")) for s in srcs if f"{s}_bpb" in ev[0]]
        if ser:
            line_chart(out / fname, f"{name} · Val BPB ({label})", "검증 데이터 bytes당 bit, 낮을수록 좋음",
                       "Step", "BPB", ser, markers=True)
            made.append(fname)

    line_chart(out / "lr.svg", f"{name} · 학습률", "Step별 학습률", "Step", "lr",
               [("lr", points(st, steps, "lr"))], y_zero=True, value_fmt=lambda v: f"{v:.1e}")
    made.append("lr.svg")

    line_chart(out / "grad_norm.svg", f"{name} · Gradient Norm", "clip 전 전체 Gradient Norm", "Step", "Gradient Norm",
               [("grad_norm", points(st, steps, "grad_norm"))], y_zero=True, hline=(1.0, "clip 1.0"))
    made.append("grad_norm.svg")

    line_chart(out / "tokens_per_s.svg", f"{name} · 처리량", "Step마다 잰 tokens/s (데이터 로딩 포함)", "Step", "tokens/s",
               [("tokens/s", points(st, steps, "tokens_per_s"))], y_zero=True)
    made.append("tokens_per_s.svg")

    if mon:
        t0 = datetime.fromisoformat(mon[0]["time"])
        tm = [(datetime.fromisoformat(r["time"]) - t0).total_seconds() / 3600 for r in mon]
        charts = [
            ("windows_ram.svg", "Windows 사용 가능 RAM",
             f"{WIN_AVAIL_STOP_MB:,}MB 미만 {PERSIST}회 연속이면 안전 정지", "win_avail_mb",
             (WIN_AVAIL_STOP_MB, f"정지 기준 {WIN_AVAIL_STOP_MB:,}MB")),
            ("gpu_memory.svg", "GPU 메모리", "nvidia-smi 기준 GPU 전체 사용량 (화면 포함)", "gpu_used_mb", None),
            ("gpu_temp.svg", "GPU 온도", f"{GPU_TEMP_STOP}°C 이상 {PERSIST}회면 안전 정지", "gpu_temp_c", None),
            ("wsl_memory.svg", "WSL swap 사용량",
             f"5분간 {SWAP_GROWTH_MB:,}MB 넘게 늘거나 {SWAP_ABS_MB:,}MB를 넘으면 안전 정지", "swap_used_mb", None),
        ]
        total = next((v for r in mon if (v := num(r.get("gpu_total_mb"))) is not None), None)
        for fname, title, sub, key, hline in charts:
            if fname == "gpu_memory.svg" and total:
                hline = (total, f"GPU 전체 {total:,.0f}MB")
            pts = points(tm, mon, key)
            if pts:
                line_chart(out / fname, f"{name} · {title}", sub, "경과 시간 (시간)", "°C" if key == "gpu_temp_c" else "MB",
                           [(title, pts)], y_zero=True, hline=hline)
                made.append(fname)
    return made


def plot_sweep(runs: list[Path], out: Path) -> list[str]:
    out.mkdir(parents=True, exist_ok=True)
    ser_bpb, ser_loss, last = [], [], 0
    for run in runs:
        ev = latest_per_step(read_csv(run / "eval.csv"))
        steps = latest_per_step(read_csv(run / "steps.csv"))
        lbl = run.name.replace("lr_", "lr ")
        ser_bpb.append((lbl, [(int(r["step"]), sum(float(r[f"{s}_bpb"]) for s in KO_SOURCES) / len(KO_SOURCES))
                              for r in ev]))
        loss = [float(r["loss"]) for r in steps]
        ser_loss.append((lbl, list(zip([int(r["step"]) for r in steps], moving_avg(loss, 20)))))
        last = max(last, int(steps[-1]["step"]))
    sub = f"{len(runs)}개 실행, 각 최대 {last:,} Step"
    line_chart(out / "lr_sweep_korean_bpb.svg", "학습률 비교 · 한국어 Val BPB 평균", sub,
               "Step", "한국어 BPB 평균", ser_bpb, markers=True)
    line_chart(out / "lr_sweep_train_loss.svg", "학습률 비교 · Train Loss", "20 Step 이동 평균", "Step", "Loss", ser_loss)
    return ["lr_sweep_korean_bpb.svg", "lr_sweep_train_loss.svg"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("run", nargs="?", type=Path)
    p.add_argument("--sweep", nargs="+", type=Path)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    made = plot_sweep(a.sweep, a.out) if a.sweep else plot_run(a.run, a.out)
    for m in made:
        print(a.out / m)


if __name__ == "__main__":
    main()
