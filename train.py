"""사전학습 루프 (설계안 v0.2 6-3, 8절)

    uv run python train.py --model pilot-42m --data runs/phase0/data.json --micro-batch 8 --grad-accum 32 \\
        --max-steps 200 --out-dir runs/smoke/pilot --eval-interval 50
    uv run python train.py --resume runs/smoke/pilot/last.pt --max-steps 400

원칙 (8절): 원인을 모르는 자동 수정은 하지 않는다.
- Loss 급등: spikes.csv에 기록만 한다 (배치 출처·샤드 정보 포함)
- NaN·Inf: 업데이트하지 않고 진단 파일을 남긴 뒤 종료 코드 3으로 멈춘다 (재개 판단은 감시 스크립트·사람)
- VRAM: 첫 업데이트 뒤 PyTorch reserved와 GPU 전체 사용량을 확인해 상한을 넘으면 멈춘다
  (Windows는 12GB를 넘으면 OOM 대신 시스템 메모리로 넘어가 조용히 느려진다)
"""

import argparse
import csv
import dataclasses
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from config import GPTConfig, model_config
from model import GPT
from optim import NonFiniteError, OptimConfig, build_optimizer, train_step, wsd_lr
from sampler import DataExhausted, MixtureSampler, Source
from tokenizer import file_sha256, load_tokenizer, token_byte_lengths

CHECKPOINT_VERSION = 1
SPIKE_LOSS_FACTOR = 1.5  # 직전 50 Step 평균보다 이만큼 크면 급등으로 기록
SPIKE_GRAD_FACTOR = 10.0  # Clipping 전 Gradient Norm이 grad_clip의 이 배수를 넘으면 기록
EXIT_NONFINITE, EXIT_VRAM, EXIT_EXHAUSTED, EXIT_STOPPED = 3, 4, 5, 6


# ---- 데이터 설정 ----


def load_data_config(path: Path) -> tuple[dict, str]:
    """반환: 설정, 설정 파일과 모든 샤드 manifest를 묶은 해시 (재개 검사용)"""
    text = Path(path).read_text(encoding="utf-8")
    cfg = json.loads(text)
    h = hashlib.sha256(text.encode())
    for src in cfg["sources"]:
        for split in ("train", "val"):
            for shard in src[split]:
                h.update(f"{shard['path']}:{shard['sha256']}:{shard['tokens']}".encode())
    return cfg, h.hexdigest()


def build_sampler(data_cfg: dict, split: str, block_size: int) -> MixtureSampler:
    sources = []
    for src in data_cfg["sources"]:
        shards = [Path(s["path"]) for s in src[split]]
        max_repeat = src["max_repeat"] if split == "train" else 1e12  # 평가는 상한 없이 고정 seed로 뽑는다
        sources.append(Source(src["name"], shards, src["weight"], max_repeat))
    return MixtureSampler(sources, block_size)


# ---- 평가 ----


@torch.no_grad()
def evaluate(model: GPT, data_cfg: dict, byte_lengths: torch.Tensor, batches: int, batch_size: int,
             seed: int, device: str) -> dict:
    """출처별 Val Loss(토큰당 nat)와 BPB(bytes당 bit). 매번 같은 seed로 같은 창을 본다."""
    was_training = model.training
    model.eval()
    out = {}
    for src in data_cfg["sources"]:
        sampler = MixtureSampler([Source(src["name"], [Path(s["path"]) for s in src["val"]], 1.0, 1e12)],
                                 model.config.block_size)
        gen = torch.Generator().manual_seed(seed)
        nll, tokens, nbytes = 0.0, 0, 0
        for _ in range(batches):
            x, y, _ = sampler.sample(batch_size, gen)
            nbytes += int(byte_lengths[y].sum())  # 바이트 길이 표와 y가 모두 CPU에 있을 때 센다
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
                logits, _ = model(x)
            nll += torch.nn.functional.cross_entropy(
                logits.float().view(-1, logits.shape[-1]), y.view(-1), reduction="sum").item()
            tokens += y.numel()
        out[src["name"]] = {"loss": nll / tokens, "bpb": nll / math.log(2) / max(nbytes, 1)}
    model.train(was_training)
    return out


# ---- 기록 ----


def append_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists() or path.stat().st_size == 0
    fields = list(row) if new else next(csv.reader(open(path, newline="", encoding="utf-8")))
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)


def last_logged_step(path: Path) -> int | None:
    if not path.exists():
        return None
    with open(path, newline="", encoding="utf-8") as f:
        steps = [int(r["step"]) for r in csv.DictReader(f)]
    return max(steps) if steps else None


def git_state() -> dict:
    def run(*args):
        return subprocess.run(["git", *args], capture_output=True, text=True, cwd=Path(__file__).parent).stdout.strip()
    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain", "--untracked-files=no"))}


def gpu_memory() -> dict:
    free, total = torch.cuda.mem_get_info()
    return {"reserved_mib": round(torch.cuda.memory_reserved() / 2**20),
            "max_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20),
            "max_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20),
            "device_used_mib": round((total - free) / 2**20), "device_total_mib": round(total / 2**20)}


# ---- Checkpoint ----


def save_checkpoint(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        torch.save(state, tmp)
        os.replace(tmp, path)
        # 방금 쓴 Checkpoint(110M 약 1.3GB)가 WSL 파일 캐시로 남아 호스트 메모리를 차지하지 않게 버린다
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


# ---- 학습 ----


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="사전학습")
    p.add_argument("--model", help="MODEL_PRESETS 이름 (새 학습)")
    p.add_argument("--data", type=Path, help="데이터 설정 JSON (새 학습)")
    p.add_argument("--resume", type=Path)
    p.add_argument("--out-dir", type=Path)
    p.add_argument("--micro-batch", type=int, default=6)
    p.add_argument("--grad-accum", type=int, default=32)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--max-lr", type=float, default=6e-4)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--decay-steps", type=int, default=0)
    p.add_argument("--min-lr-ratio", type=float, default=0.0)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--eval-interval", type=int, default=250)
    p.add_argument("--eval-batches", type=int, default=8)
    p.add_argument("--save-minutes", type=float, default=30.0)
    p.add_argument("--vram-limit-mib", type=int, default=10500,
                   help="PyTorch reserved 상한. GPU 전체 사용량은 물리 용량 - 512MiB를 넘으면 멈춘다")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 학습 경로를 바꾸는 설정: 재개 시 Checkpoint와 같아야 한다 (max_steps·decay_steps 연장은 허용)
    resume_keys = ("micro_batch", "grad_accum", "max_lr", "warmup_steps", "weight_decay", "grad_clip", "seed")

    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=True)
        out_dir = args.resume.parent if args.out_dir is None else args.out_dir
        if out_dir.resolve() != args.resume.parent.resolve():
            raise ValueError("재개는 Checkpoint가 있는 폴더에서만 한다 (기존 결과 보호)")
        run = ckpt["run"]
        diffs = [k for k in resume_keys if run[k] != getattr(args, k)]
        if diffs:
            raise ValueError(f"학습 설정이 Checkpoint와 다르다: {diffs}")
        model_cfg = GPTConfig(**ckpt["model_config"])
        data_path = Path(run["data"])
    else:
        if not (args.model and args.data and args.out_dir):
            raise ValueError("새 학습은 --model, --data, --out-dir가 필요하다")
        out_dir = args.out_dir
        if (out_dir / "last.pt").exists() or (out_dir / "steps.csv").exists() or (out_dir / "STOP").exists():
            raise FileExistsError(f"{out_dir}에 이전 결과가 있다. 다른 --out-dir을 쓰거나 --resume")
        data_path = args.data

    data_cfg, data_hash = load_data_config(data_path)
    tok_path = Path(data_cfg["tokenizer"])
    tok_sha = file_sha256(tok_path)
    tok = load_tokenizer(tok_path)
    byte_lengths = torch.tensor(token_byte_lengths(tok), dtype=torch.long)

    if args.resume:
        if ckpt["data_hash"] != data_hash or ckpt["tokenizer_sha256"] != tok_sha:
            raise ValueError("데이터 또는 Tokenizer가 Checkpoint와 다르다")
        logged = last_logged_step(out_dir / "steps.csv")
        if logged is not None and logged > ckpt["step"]:
            print(f"참고: steps.csv에 Checkpoint(step {ckpt['step']}) 이후 기록(~{logged})이 있다. "
                  f"재개 후 같은 step이 다시 기록되며 resume 열로 구분한다")
    else:
        model_cfg = model_config(args.model, vocab_size=tok.get_vocab_size(), block_size=data_cfg["block_size"])

    optim_cfg = OptimConfig(max_lr=args.max_lr, min_lr_ratio=args.min_lr_ratio, warmup_steps=args.warmup_steps,
                            decay_steps=args.decay_steps, max_steps=args.max_steps,
                            weight_decay=args.weight_decay, grad_clip=args.grad_clip)
    torch.manual_seed(args.seed)
    model = GPT(model_cfg).to(device)
    optimizer = build_optimizer(model, optim_cfg)
    sampler = build_sampler(data_cfg, "train", model_cfg.block_size)
    gen = torch.Generator().manual_seed(args.seed)
    step, tokens_seen, resume_count = 0, 0, 0
    if args.resume:
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        sampler.load_state_dict(ckpt["sampler"])
        gen.set_state(ckpt["rng"]["train_gen"])
        torch.set_rng_state(ckpt["rng"]["torch_cpu"])
        if ckpt["rng"]["torch_cuda"] is not None and device == "cuda":
            torch.cuda.set_rng_state_all(ckpt["rng"]["torch_cuda"])
        step, tokens_seen, resume_count = ckpt["step"], ckpt["tokens_seen"], ckpt["resume_count"] + 1
        del ckpt

    run_info = {**{k: getattr(args, k) for k in resume_keys}, "data": str(data_path), "model": args.model}
    tokens_per_step = args.micro_batch * args.grad_accum * model_cfg.block_size
    print(f"device {device} | 파라미터 {model.num_parameters():,} | {tokens_per_step:,} 토큰/Step "
          f"(micro {args.micro_batch} × accum {args.grad_accum} × {model_cfg.block_size}) | "
          f"step {step} → {args.max_steps} | 재개 {resume_count}회 | 출력 {out_dir}", flush=True)

    def state() -> dict:
        return {
            "format_version": CHECKPOINT_VERSION, "step": step, "tokens_seen": tokens_seen,
            "resume_count": resume_count, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "model_config": dataclasses.asdict(model_cfg), "optim_config": dataclasses.asdict(optim_cfg),
            "run": run_info, "sampler": sampler.state_dict(), "data_hash": data_hash,
            "tokenizer_sha256": tok_sha, "git": git_state(), "torch_version": str(torch.__version__),
            "rng": {"train_gen": gen.get_state(), "torch_cpu": torch.get_rng_state(),
                    "torch_cuda": torch.cuda.get_rng_state_all() if device == "cuda" else None},
        }

    recent: list[float] = []
    last_save = time.monotonic()
    vram_checked = False
    model.train()
    while step < args.max_steps:
        stop_file = out_dir / "STOP"
        if stop_file.exists():  # 감시 스크립트(scripts/monitor.py)나 사람이 요청한 안전 정지
            reason = stop_file.read_text(encoding="utf-8").strip() or "사유 없음"
            save_checkpoint(out_dir / "last.pt", state())
            append_csv(out_dir / "stops.csv", {"step": step, "tokens_seen": tokens_seen, "reason": reason,
                                               "time": time.strftime("%Y-%m-%d %H:%M:%S")})
            print(f"중단 요청으로 정지 (step {step}): {reason}. last.pt 저장 완료. "
                  f"원인 확인 후 STOP 파일을 지우고 --resume으로 재개", flush=True)
            return EXIT_STOPPED
        lr = wsd_lr(step, optim_cfg)
        t_data = time.perf_counter()
        try:
            draws = [sampler.sample(args.micro_batch, gen) for _ in range(args.grad_accum)]
        except DataExhausted as e:
            print(f"중단: 모든 출처가 노출 상한에 도달했다 (step {step})\n{e}", flush=True)
            save_checkpoint(out_dir / "last.pt", state())
            return EXIT_EXHAUSTED
        data_seconds = time.perf_counter() - t_data
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            loss, grad_norm = train_step(model, optimizer, [(x, y) for x, y, _ in draws], lr, args.grad_clip, device)
        except NonFiniteError as e:
            diag = {"step": step, "error": str(e), "lr": lr, "recent_losses": recent[-100:],
                    "batch_sources": [n for _, _, names in draws for n in names],
                    "sampler": sampler.state_dict(), "checkpoint_step": None}
            (out_dir / f"nonfinite_step{step}.json").write_text(json.dumps(diag, ensure_ascii=False, indent=1))
            print(f"중단: step {step}에서 {e}. 진단 파일을 남겼다. 마지막 Checkpoint에서 1회만 재개를 시도할 것",
                  flush=True)
            return EXIT_NONFINITE
        if device == "cuda":
            torch.cuda.synchronize()
        step_seconds = time.perf_counter() - t0
        step += 1
        tokens_seen += tokens_per_step

        mem = gpu_memory() if device == "cuda" else {}
        if device == "cuda" and not vram_checked:
            vram_checked = True
            print(f"VRAM (첫 업데이트 뒤): {mem}", flush=True)
            if mem["max_reserved_mib"] > args.vram_limit_mib or mem["device_used_mib"] > mem["device_total_mib"] - 512:
                print(f"중단: VRAM 상한 초과 (reserved 상한 {args.vram_limit_mib} MiB, 전체 여유 512 MiB 미만)",
                      flush=True)
                return EXIT_VRAM

        mean_recent = sum(recent[-50:]) / len(recent[-50:]) if recent else None
        spike = (mean_recent is not None and len(recent) >= 50 and loss > SPIKE_LOSS_FACTOR * mean_recent) \
            or grad_norm > SPIKE_GRAD_FACTOR * args.grad_clip
        recent.append(loss)
        append_csv(out_dir / "steps.csv", {
            "step": step, "loss": f"{loss:.4f}", "grad_norm": f"{grad_norm:.4f}", "lr": f"{lr:.3e}",
            "tokens_seen": tokens_seen, "step_s": f"{step_seconds:.3f}", "data_s": f"{data_seconds:.3f}",
            "tokens_per_s": round(tokens_per_step / (step_seconds + data_seconds)),
            "reserved_mib": mem.get("reserved_mib", ""), "device_used_mib": mem.get("device_used_mib", ""),
            "resume": resume_count,
        })
        if spike:
            sources = [n for _, _, names in draws for n in names]
            append_csv(out_dir / "spikes.csv", {
                "step": step, "loss": f"{loss:.4f}", "recent_mean": "" if mean_recent is None else f"{mean_recent:.4f}",
                "grad_norm": f"{grad_norm:.4f}", "lr": f"{lr:.3e}",
                "sources": ";".join(f"{s}={sources.count(s)}" for s in sorted(set(sources))),
            })

        is_eval = step % args.eval_interval == 0 or step == args.max_steps
        if is_eval:
            ev = evaluate(model, data_cfg, byte_lengths, args.eval_batches, args.micro_batch, args.seed + 1, device)
            row = {"step": step, "tokens_seen": tokens_seen, "train_loss_recent": f"{mean_recent or loss:.4f}"}
            for name, v in ev.items():
                row[f"{name}_loss"] = f"{v['loss']:.4f}"
                row[f"{name}_bpb"] = f"{v['bpb']:.4f}"
            row.update({f"seen_{k}": v for k, v in sampler.state.seen_tokens.items()})
            append_csv(out_dir / "eval.csv", row)
            print(f"step {step} | loss {loss:.4f} | " + " | ".join(f"{k} {v['bpb']:.3f}" for k, v in ev.items())
                  + f" bpb | {tokens_per_step / step_seconds:,.0f} tok/s (연산), VRAM {mem.get('max_reserved_mib')} MiB",
                  flush=True)
        if is_eval or time.monotonic() - last_save > args.save_minutes * 60:
            save_checkpoint(out_dir / "last.pt", state())
            last_save = time.monotonic()

    print(sampler.report(), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
