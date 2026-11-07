"""SFT 학습 (Phase 2). 사전학습 Checkpoint에서 시작해 sft_data.py가 만든 데이터로 학습한다.

    scripts/run_monitored.sh runs/phase2/sft42m  (TRAIN_SCRIPT=sft_train.py)  \
        --model pilot-42m --init runs/phase1/pilot42m/last.pt --data-dir data/sft/builds/phase2_a2_noS11_v1 \
        --max-lr 1e-4 --epochs 1 --deterministic

Loss: Step 안 모든 micro-batch의 응답 토큰(마스크 1) 전체에 대한 평균 cross-entropy.
      micro-batch마다 합계를 내고 Step 전체 응답 토큰 수로 나눈다 (micro-batch 평균의 평균이 아니다).
배치: Step마다 conv_per_step개 대화(epoch마다 seed로 섞음) → 길이순 정렬 → 패딩 포함 micro_tokens 이하로 묶음.
      오른쪽 패딩 (x=PAD, y=IGNORE). Causal이라 패딩은 앞 토큰의 값에 영향을 주지 않는다.
안전장치: train.py와 같음 (STOP 파일 안전 정지, NaN 정지, 저장 전 디스크 점검, 저장 중 메모리 기록, VRAM 상한).
"""

import argparse
import dataclasses
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

import train as T
from config import GPTConfig
from model import GPT, IGNORE_INDEX
from optim import NonFiniteError, OptimConfig, build_optimizer, wsd_lr
from text import PAD_ID
from tokenizer import file_sha256

SFT_CHECKPOINT_VERSION = 1
RESUME_KEYS = ("data_dir", "data_train_sha256", "init_sha256", "conv_per_step", "micro_tokens", "epochs", "max_lr",
               "warmup_frac", "min_lr_ratio", "weight_decay", "grad_clip", "seed", "deterministic")


# ---- 데이터 ----


def load_split(path: Path) -> list[dict]:
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    for r in rows:
        if len(r["ids"]) != len(r["mask"]) or len(r["ids"]) < 2 or not any(r["mask"][1:]):
            raise ValueError(f"{r.get('id')}: ids·mask 길이가 다르거나 응답 토큰이 없다")
    return rows


def step_plan(n: int, conv_per_step: int, epochs: int, seed: int) -> list[list[int]]:
    """전체 학습의 Step별 대화 번호. epoch마다 seed+epoch로 섞는다 (재개 시 같은 계획을 다시 만든다)"""
    plan = []
    for ep in range(epochs):
        order = torch.randperm(n, generator=torch.Generator().manual_seed(seed + ep)).tolist()
        plan += [order[i:i + conv_per_step] for i in range(0, n, conv_per_step)]
    return plan


def micro_batches(rows: list[dict], idx: list[int], micro_tokens: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """길이순으로 묶어 (x, y) 목록. x는 PAD, y는 IGNORE_INDEX로 오른쪽 패딩. 패딩 포함 토큰 수 ≤ micro_tokens"""
    items = sorted(idx, key=lambda i: (len(rows[i]["ids"]), i), reverse=True)
    out, cur = [], []
    for i in items:
        # 길이 내림차순이라 묶음의 첫 대화가 폭(가장 긴 길이)을 정한다
        if cur and (len(cur) + 1) * (len(rows[cur[0]]["ids"]) - 1) > micro_tokens:
            out.append(cur)
            cur = []
        cur.append(i)
    if cur:
        out.append(cur)
    batches = []
    for group in out:
        width = max(len(rows[i]["ids"]) for i in group) - 1
        x = torch.full((len(group), width), PAD_ID, dtype=torch.long)
        y = torch.full((len(group), width), IGNORE_INDEX, dtype=torch.long)
        for b, i in enumerate(group):
            ids, mask = rows[i]["ids"], rows[i]["mask"]
            n = len(ids) - 1
            x[b, :n] = torch.tensor(ids[:-1])
            yy = torch.tensor(ids[1:])
            yy[~torch.tensor(mask[1:], dtype=torch.bool)] = IGNORE_INDEX
            y[b, :n] = yy
        batches.append((x, y))
    return batches


def summed_loss(model: GPT, x: torch.Tensor, y: torch.Tensor, device: str) -> tuple[torch.Tensor, int]:
    """응답 토큰 cross-entropy 합과 응답 토큰 수"""
    x, y = x.to(device), y.to(device)
    with torch.autocast(device_type="cuda" if device.startswith("cuda") else "cpu", dtype=torch.bfloat16,
                        enabled=device.startswith("cuda")):
        logits, _ = model(x)
    loss = F.cross_entropy(logits.float().view(-1, logits.shape[-1]), y.view(-1), ignore_index=IGNORE_INDEX,
                           reduction="sum")
    return loss, int((y != IGNORE_INDEX).sum())


def sft_step(model: GPT, optimizer, batches, lr: float, grad_clip: float, device: str) -> tuple[float, float, int]:
    """응답 토큰 가중 평균 Loss로 한 번 업데이트. 반환: Loss, Clipping 전 Gradient Norm, 응답 토큰 수"""
    for g in optimizer.param_groups:
        g["lr"] = lr
    optimizer.zero_grad(set_to_none=True)
    total_tokens = sum(int((y != IGNORE_INDEX).sum()) for _, y in batches)
    if total_tokens == 0:
        raise ValueError("이 Step에 응답 토큰이 없다")
    total = 0.0
    for x, y in batches:
        loss_sum, _ = summed_loss(model, x, y, device)
        (loss_sum / total_tokens).backward()
        total += loss_sum.detach().float().item()
    loss = total / total_tokens
    if not math.isfinite(loss):
        optimizer.zero_grad(set_to_none=True)
        raise NonFiniteError(f"loss가 {loss}")
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip).item()
    if not math.isfinite(grad_norm):
        optimizer.zero_grad(set_to_none=True)
        raise NonFiniteError(f"gradient norm이 {grad_norm}")
    optimizer.step()
    return loss, grad_norm, total_tokens


@torch.no_grad()
def evaluate(model: GPT, rows: list[dict], micro_tokens: int, device: str) -> dict:
    """Val 응답 토큰 평균 Loss (전체, 언어별)"""
    was = model.training
    model.eval()
    out = {}
    for lang in ("all", "ko", "en"):
        idx = [i for i, r in enumerate(rows) if lang == "all" or r["lang"] == lang]
        if not idx:
            continue
        s, n = 0.0, 0
        for x, y in micro_batches(rows, idx, micro_tokens):
            ls, k = summed_loss(model, x, y, device)
            s += float(ls)
            n += k
        out[lang] = s / n
    model.train(was)
    return out


# ---- 실행 ----


def set_deterministic(on: bool) -> None:
    if on:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # cuBLAS 결정론 (CUDA 사용 전에 설정)
        torch.use_deterministic_algorithms(True)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SFT")
    p.add_argument("--model", help="프리셋 이름 (시작 전 점검용, --init Checkpoint의 모델과 같아야 함)")
    p.add_argument("--init", type=Path, help="시작할 사전학습 Checkpoint (새 학습)")
    p.add_argument("--data-dir", type=Path, help="sft_data.py 출력 폴더 (새 학습)")
    p.add_argument("--resume", type=Path)
    p.add_argument("--out-dir", type=Path)
    p.add_argument("--conv-per-step", type=int, default=64)
    p.add_argument("--micro-tokens", type=int, default=8192, help="micro-batch 패딩 포함 토큰 상한")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max-lr", type=float, default=1e-4)
    p.add_argument("--warmup-frac", type=float, default=0.05)
    p.add_argument("--min-lr-ratio", type=float, default=0.0)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--eval-interval", type=int, default=25)
    p.add_argument("--save-minutes", type=float, default=30.0)
    p.add_argument("--vram-limit-mib", type=int, default=10500)
    return p.parse_args(argv)


def main(argv=None) -> int:
    try:
        return run(argv)
    except T.DiskSpaceError as e:
        print(f"중단: {e}. 디스크를 정리한 뒤 --resume으로 재개", flush=True)
        return T.EXIT_DISK


def run(argv=None) -> int:
    args = parse_args(argv)
    set_deterministic(args.deterministic)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=True)
        out_dir = args.out_dir or args.resume.parent
        if out_dir.resolve() != args.resume.parent.resolve():
            raise ValueError("재개는 Checkpoint가 있는 폴더에 이어 쓴다")
        info = ck["sft"]
        args.data_dir, args.init = Path(info["data_dir"]), Path(info["init"])
    else:
        if not (args.init and args.data_dir and args.out_dir):
            raise ValueError("새 학습은 --init, --data-dir, --out-dir가 필요하다")
        out_dir = args.out_dir
        if (out_dir / "last.pt").exists() or (out_dir / "steps.csv").exists() or (out_dir / "STOP").exists():
            raise FileExistsError(f"{out_dir}에 이전 결과가 있다. 다른 --out-dir을 쓰거나 --resume")
    manifest = json.loads((args.data_dir / "manifest.json").read_text(encoding="utf-8"))
    train_rows = load_split(args.data_dir / "train.jsonl")
    val_rows = load_split(args.data_dir / "val.jsonl")
    if file_sha256(args.data_dir / "train.jsonl") != manifest["train_sha256"]:
        raise ValueError("train.jsonl이 manifest와 다르다")

    init = torch.load(args.init, map_location="cpu", weights_only=True)
    if args.model and init["run"].get("model") != args.model:
        raise ValueError(f"--model {args.model} ≠ Checkpoint 모델 {init['run'].get('model')}")
    model_cfg = GPTConfig(**init["model_config"])
    plan = step_plan(len(train_rows), args.conv_per_step, args.epochs, args.seed)
    total_steps = len(plan)
    warmup = max(1, round(args.warmup_frac * total_steps))
    optim_cfg = OptimConfig(max_lr=args.max_lr, min_lr_ratio=args.min_lr_ratio, warmup_steps=warmup,
                            decay_steps=total_steps - warmup, max_steps=total_steps,
                            weight_decay=args.weight_decay, grad_clip=args.grad_clip)
    torch.manual_seed(args.seed)
    model = GPT(model_cfg)
    model.load_state_dict(init["model"])
    model.to(device)
    optimizer = build_optimizer(model, optim_cfg)
    run_info = {"data_dir": str(args.data_dir), "data_train_sha256": manifest["train_sha256"], "init": str(args.init),
                "init_sha256": file_sha256(args.init), "init_step": init["step"], "model": init["run"].get("model"),
                **{k: getattr(args, k) for k in RESUME_KEYS if hasattr(args, k) and k != "data_dir"}}
    step = 0
    if args.resume:
        diffs = [k for k in RESUME_KEYS if ck["sft"].get(k) != run_info.get(k)]
        if diffs:
            raise ValueError(f"학습 설정이 Checkpoint와 다르다: {diffs}")
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        torch.set_rng_state(ck["rng"]["torch_cpu"])
        step = ck["step"]
        resume_count = ck["resume_count"] + 1
        del ck
    else:
        resume_count = 0
    del init
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_log = out_dir / "checkpoints.csv"
    print(f"device {device} | 파라미터 {model.num_parameters():,} | 학습 대화 {len(train_rows):,} · Val {len(val_rows):,} | "
          f"{args.conv_per_step}대화/Step × {total_steps} Step (warmup {warmup}) | 결정론 {args.deterministic} | "
          f"step {step} 시작 | 재개 {resume_count}회 | 출력 {out_dir}", flush=True)

    def state() -> dict:
        return {"format_version": SFT_CHECKPOINT_VERSION, "kind": "sft", "step": step, "total_steps": total_steps,
                "resume_count": resume_count, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "model_config": dataclasses.asdict(model_cfg), "optim_config": dataclasses.asdict(optim_cfg),
                "run": {"model": run_info["model"], "tokenizer": manifest["tokenizer"]}, "sft": run_info, "git": T.git_state(),
                "torch_version": str(torch.__version__), "rng": {"torch_cpu": torch.get_rng_state()}}

    if not args.resume:
        ev = evaluate(model, val_rows, args.micro_tokens, device)
        T.append_csv(out_dir / "val.csv", {"step": 0, **{f"val_loss_{k}": f"{v:.4f}" for k, v in ev.items()}})
        print("step 0 (SFT 전) | " + " | ".join(f"val {k} {v:.4f}" for k, v in ev.items()), flush=True)
    last_save = time.monotonic()
    vram_checked = False
    model.train()
    while step < total_steps:
        if (out_dir / "STOP").exists():
            reason = (out_dir / "STOP").read_text(encoding="utf-8").strip() or "사유 없음"
            T.save_checkpoint(out_dir / "last.pt", state(), ckpt_log, step)
            T.append_csv(out_dir / "stops.csv", {"step": step, "reason": reason, "time": time.strftime("%F %T")})
            print(f"중단 요청으로 정지 (step {step}): {reason}. last.pt 저장 완료", flush=True)
            return T.EXIT_STOPPED
        lr = wsd_lr(step, optim_cfg)
        batches = micro_batches(train_rows, plan[step], args.micro_tokens)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            loss, gn, n_tok = sft_step(model, optimizer, batches, lr, args.grad_clip, device)
        except NonFiniteError as e:
            (out_dir / f"nonfinite_step{step}.json").write_text(json.dumps(
                {"step": step, "error": str(e), "lr": lr, "conversations": [train_rows[i]["id"] for i in plan[step]]},
                ensure_ascii=False, indent=1))
            print(f"중단: step {step}에서 {e}. 진단 파일을 남겼다", flush=True)
            return T.EXIT_NONFINITE
        if device == "cuda":
            torch.cuda.synchronize()
        secs = time.perf_counter() - t0
        step += 1
        mem = T.gpu_memory() if device == "cuda" else {}
        if device == "cuda" and not vram_checked:
            vram_checked = True
            print(f"VRAM (첫 업데이트 뒤): {mem}", flush=True)
            if mem["max_reserved_mib"] > args.vram_limit_mib or mem["device_used_mib"] > mem["device_total_mib"] - 512:
                print("중단: VRAM 상한 초과", flush=True)
                return T.EXIT_VRAM
        padded = sum(x.numel() for x, _ in batches)
        T.append_csv(out_dir / "steps.csv", {
            "step": step, "loss": f"{loss:.4f}", "grad_norm": f"{gn:.4f}", "lr": f"{lr:.3e}",
            "conversations": len(plan[step - 1]), "loss_tokens": n_tok, "padded_tokens": padded,
            "micro_batches": len(batches), "step_s": f"{secs:.3f}", "tokens_per_s": round(padded / secs),
            "reserved_mib": mem.get("reserved_mib", ""), "device_used_mib": mem.get("device_used_mib", ""),
            "resume": resume_count})
        if step % args.eval_interval == 0 or step == total_steps:
            ev = evaluate(model, val_rows, args.micro_tokens, device)
            T.append_csv(out_dir / "val.csv", {"step": step, **{f"val_loss_{k}": f"{v:.4f}" for k, v in ev.items()}})
            print(f"step {step}/{total_steps} | loss {loss:.4f} | " + " | ".join(f"val {k} {v:.4f}" for k, v in ev.items())
                  + f" | {padded / secs:,.0f} tok/s | VRAM {mem.get('max_reserved_mib')} MiB", flush=True)
        if step == total_steps or time.monotonic() - last_save > args.save_minutes * 60:
            T.save_checkpoint(out_dir / "last.pt", state(), ckpt_log, step)
            last_save = time.monotonic()
    return 0


if __name__ == "__main__":
    sys.exit(main())
