"""학습 실험 요약: 출처별 Val BPB, Train Loss, Gradient 안정성, NaN, 처리량, 자원 추세 (Phase 1 학습률 비교 보고용)

    python3 scripts/summarize_runs.py runs/phase1/lr_5e-4 runs/phase1/lr_1e-3 runs/phase1/lr_2e-3 > runs/phase1/summary.json
"""

import csv
import json
import sys
from pathlib import Path

KOREAN = ("P1_fineweb2_ko", "P2_wikipedia_ko", "P3_finepdfs_ko")


def rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def stats(values: list[float]) -> dict:
    return {"min": round(min(values), 1), "max": round(max(values), 1), "first": round(values[0], 1),
            "last": round(values[-1], 1)} if values else {}


def summarize(run: Path) -> dict:
    steps, evals, mon = rows(run / "steps.csv"), rows(run / "eval.csv"), rows(run / "monitor.csv")
    out = {"run": str(run), "steps": len(steps)}
    if steps:
        loss = [float(r["loss"]) for r in steps]
        grad = [float(r["grad_norm"]) for r in steps]
        out["train_loss_last50_mean"] = round(sum(loss[-50:]) / len(loss[-50:]), 4)
        out["grad_norm_max"] = round(max(grad), 3)
        out["grad_norm_max_after_warmup"] = round(max(grad[30:] or [0]), 3)
        out["clipped_fraction"] = round(sum(g > 1.0 for g in grad) / len(grad), 3)
        out["clipped_fraction_last100"] = round(sum(g > 1.0 for g in grad[-100:]) / len(grad[-100:]), 3)
        out["tokens_per_s_median"] = sorted(int(r["tokens_per_s"]) for r in steps)[len(steps) // 2]
    out["spikes"] = len(rows(run / "spikes.csv"))
    out["nonfinite_files"] = [p.name for p in run.glob("nonfinite_step*.json")]
    out["stops"] = rows(run / "stops.csv")
    if evals:
        last = evals[-1]
        bpb = {k[:-4]: float(v) for k, v in last.items() if k.endswith("_bpb")}
        out["final_eval_step"] = int(last["step"])
        out["val_bpb"] = bpb
        out["val_bpb_korean_mean"] = round(sum(bpb[k] for k in KOREAN) / len(KOREAN), 4)
        out["val_bpb_all_mean"] = round(sum(bpb.values()) / len(bpb), 4)
        out["val_bpb_curve_korean_mean"] = [
            (int(r["step"]), round(sum(float(r[f"{k}_bpb"]) for k in KOREAN) / len(KOREAN), 4)) for r in evals]
    if mon:
        out["monitor_samples"] = len(mon)
        for key in ("win_avail_mb", "wsl_used_mb", "wsl_cache_mb", "swap_used_mb", "gpu_used_mb", "gpu_temp_c",
                    "gpu_power_w"):
            out[key] = stats([float(r[key]) for r in mon if r.get(key) not in ("", None)])
        out["monitor_warnings"] = sorted({r["warnings"] for r in mon if r["warnings"]})
    return out


if __name__ == "__main__":
    print(json.dumps([summarize(Path(p)) for p in sys.argv[1:]], ensure_ascii=False, indent=1))
