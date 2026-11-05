"""처리량·VRAM 측정 (Phase 0). 무작위 토큰으로 실제 학습 Step(BF16, accumulation, AdamW)을 돌린다.

Loss 값은 의미가 없다 (무작위 정답). 보는 것: Forward·Backward 정상 동작, 유한한 Loss·Gradient,
토큰/초, 최대 VRAM. 측정은 별도 프로세스에서 하나씩 실행한다 (VRAM 최대값이 섞이지 않게).

    uv run python bench.py --model main-110m --micro-batch 8 --grad-accum 4 --steps 20
"""

import argparse
import json
import math
import time

import torch

from config import model_config
from model import GPT
from optim import OptimConfig, build_optimizer, train_step


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--micro-batch", type=int, required=True)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--vocab-size", type=int, default=32768)
    parser.add_argument("--steps", type=int, default=20, help="측정에 쓰는 업데이트 수 (워밍업 3회 별도)")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--json", help="결과를 덧붙일 JSONL 파일")
    args = parser.parse_args()

    device = "cuda"
    torch.manual_seed(0)
    cfg = model_config(args.model, vocab_size=args.vocab_size, block_size=args.block_size)
    model = GPT(cfg).to(device)
    optimizer = build_optimizer(model, OptimConfig())
    step_model = torch.compile(model) if args.compile else model
    gen = torch.Generator().manual_seed(0)

    def batches():
        return [(torch.randint(args.vocab_size, (args.micro_batch, args.block_size), generator=gen),
                 torch.randint(args.vocab_size, (args.micro_batch, args.block_size), generator=gen))
                for _ in range(args.grad_accum)]

    result = {"model": args.model, "params": model.num_parameters(), "micro_batch": args.micro_batch,
              "grad_accum": args.grad_accum, "block_size": args.block_size, "vocab_size": args.vocab_size,
              "compile": args.compile}
    try:
        losses, norms = [], []
        for _ in range(args.warmup):
            train_step(step_model, optimizer, batches(), 1e-4, 1.0, device)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        elapsed = 0.0
        for _ in range(args.steps):
            data = batches()  # 배치 생성 시간은 측정에서 뺀다
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            loss, norm = train_step(step_model, optimizer, data, 1e-4, 1.0, device)
            torch.cuda.synchronize()
            elapsed += time.perf_counter() - t0
            losses.append(loss)
            norms.append(norm)
        tokens = args.steps * args.grad_accum * args.micro_batch * args.block_size
        result.update(
            ok=True,
            tokens_per_sec=round(tokens / elapsed),
            ms_per_update=round(1000 * elapsed / args.steps, 1),
            tokens_per_update=args.grad_accum * args.micro_batch * args.block_size,
            peak_allocated_mib=round(torch.cuda.max_memory_allocated() / 2**20),
            peak_reserved_mib=round(torch.cuda.max_memory_reserved() / 2**20),
            loss_first=round(losses[0], 4), loss_last=round(losses[-1], 4),
            grad_norm_max=round(max(norms), 4),
            all_finite=all(math.isfinite(v) for v in losses + norms),
            ln_vocab=round(math.log(args.vocab_size), 4),
        )
    except torch.OutOfMemoryError as e:
        result.update(ok=False, error="OOM", detail=str(e).splitlines()[0])
    print(json.dumps(result, ensure_ascii=False))
    if args.json:
        with open(args.json, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
