"""사전학습 모델과 터미널에서 묻고 답하기 (추론 전용, SFT 전 모델 확인용)

    uv run python ask.py                      # 기본: runs/phase3/main110m/last.pt
    uv run python ask.py --temperature 0.9 --max-new-tokens 200

- 프롬프트: 학습 문서처럼 <|endoftext|>로 시작하고 `질문: {입력}\\n답변:` 형식으로 넣는다 (SFT 전이라 대화 템플릿 대신).
- 모델은 시작할 때 한 번만 GPU에 올리고 질문마다 재사용한다. exit(또는 quit, Ctrl-D)로 끝낸다.
- 추론만 한다: torch.inference_mode, model.eval(). Optimizer를 만들지 않고 아무 파일도 쓰지 않는다.
- Checkpoint는 mmap으로 읽어 가중치만 GPU로 옮긴다 (Optimizer 상태는 메모리에 올리지 않음). 읽은 뒤 파일 캐시를 돌려준다.
- 시작 전에 Windows RAM·WSL·GPU 여유를 확인하고, 모자라면 모델을 읽지 않고 끝낸다 (종료 코드 8).
  기준은 추론용이다 (학습 시작 기준 6,144MB와 별개): 110M 가중치 0.44GB + CUDA 문맥, 감시 경고선 2,048MB 위로 1GB 여유.
- 모델이 답변 뒤에 다음 "질문:"이나 "답변:" 줄을 지어내 이어 쓰는 경우가 많아, 화면에는 그 앞까지만 보여 준다 (--raw면 전부).
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))
from chat_generate import generate  # noqa: E402  (정지 토큰 처리까지 같은 생성 함수)
from config import GPTConfig  # noqa: E402
from filecache import drop_cache  # noqa: E402
from model import GPT  # noqa: E402
from monitor import gpu, windows_available_mb, wsl_memory_mb  # noqa: E402
from preflight import other_training_pids  # noqa: E402
from text import EOT_ID  # noqa: E402
from tokenizer import decode, encode, load_tokenizer  # noqa: E402

DEFAULT_CKPT = ROOT / "runs/phase3/main110m/last.pt"
PROMPT = "질문: {q}\n답변:"
MIN_WIN_AVAIL_MB = 3072
MIN_WSL_AVAIL_MB = 2048
MIN_GPU_FREE_MIB = 2048
EXIT_RESOURCES = 8
QUIT = {"exit", "quit", "종료"}


def resource_check() -> list[str]:
    """모자란 항목의 설명 목록 (비어 있으면 통과). 측정 실패도 모자란 것으로 본다.
    측정 중에는 표준 입력을 /dev/null로 바꿔 둔다 (powershell.exe가 미리 입력한 질문을 읽어 가지 않게)"""
    saved = os.dup(0)
    null = os.open(os.devnull, os.O_RDONLY)
    os.dup2(null, 0)
    try:
        return _resource_check()
    finally:
        os.dup2(saved, 0)
        os.close(saved)
        os.close(null)


def _resource_check() -> list[str]:
    bad = []
    win = windows_available_mb()
    print(f"  Windows 사용 가능 RAM: {win} MB (기준 >= {MIN_WIN_AVAIL_MB})")
    if win is None or win < MIN_WIN_AVAIL_MB:
        bad.append(f"Windows 사용 가능 RAM {win} MB < {MIN_WIN_AVAIL_MB}")
    wsl = wsl_memory_mb()["wsl_avail_mb"]
    print(f"  WSL MemAvailable: {wsl} MB (기준 >= {MIN_WSL_AVAIL_MB})")
    if wsl < MIN_WSL_AVAIL_MB:
        bad.append(f"WSL MemAvailable {wsl} MB < {MIN_WSL_AVAIL_MB}")
    g = gpu()
    free = g["gpu_total_mb"] - g["gpu_used_mb"] if g else None
    print(f"  GPU 여유: {free} MiB (기준 >= {MIN_GPU_FREE_MIB})")
    if free is None or free < MIN_GPU_FREE_MIB:
        bad.append(f"GPU 여유 {free} MiB < {MIN_GPU_FREE_MIB}")
    pids = other_training_pids()
    if pids:
        bad.append(f"이 프로젝트의 학습이 실행 중 (pid {pids})")
    return bad


def load_model(path: Path, device: str):
    """가중치만 읽어 GPU에 올린다. Checkpoint 파일은 읽기만 한다"""
    ck = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    model = GPT(GPTConfig(**ck["model_config"]))
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    tok_path = ck["run"]["tokenizer"] if ck.get("kind") == "sft" else \
        json.loads((ROOT / ck["run"]["data"]).read_text(encoding="utf-8"))["tokenizer"]
    info = {"step": ck.get("step"), "tokens_seen": ck.get("tokens_seen")}
    del ck
    drop_cache(path)  # filecache.py: 읽은 Checkpoint를 WSL 파일 캐시에서 뺀다
    return model, load_tokenizer(tok_path), info


def cut_answer(text: str) -> str:
    """모델이 지어낸 다음 질문·답변 줄부터는 잘라낸다"""
    cuts = [i for m in ("\n질문:", "\n질문 :", "질문:", "\n답변:") if (i := text.find(m)) > 0]
    return text[:min(cuts)] if cuts else text


def ask(model, tok, question: str, settings: dict, seed: int, device: str) -> tuple[str, str, int]:
    """반환: 생성 전체 문자열, 정지 이유, 생성 토큰 수"""
    ids = [EOT_ID] + encode(tok, PROMPT.format(q=question))
    gen = torch.Generator().manual_seed(seed)
    with torch.inference_mode():
        out, reason = generate(model, ids, settings["max_new_tokens"], settings["temperature"], settings["top_k"],
                               gen, device)
    return decode(tok, out), reason, len(out)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", nargs="?", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--seed", type=int, default=1234, help="질문마다 seed + 질문 번호")
    p.add_argument("--raw", action="store_true", help="다음 '질문:' 이후도 자르지 않고 모두 보여 준다")
    a = p.parse_args(argv)
    if not torch.cuda.is_available():
        print("CUDA를 쓸 수 없다. 실행하지 않는다")
        return EXIT_RESOURCES
    print("자원 확인:")
    bad = resource_check()
    if bad:
        print("자원이 모자라 모델을 읽지 않고 끝낸다: " + "; ".join(bad))
        print("다른 프로그램(Docker Desktop, 다른 프로젝트)은 종료하거나 바꾸지 않는다.")
        return EXIT_RESOURCES
    device = "cuda"
    t0 = time.perf_counter()
    model, tok, info = load_model(a.checkpoint, device)
    settings = {"temperature": a.temperature, "top_k": a.top_k, "max_new_tokens": a.max_new_tokens}
    print(f"모델: {a.checkpoint} (step {info['step']}, {info['tokens_seen'] or 0:,} 토큰), "
          f"{time.perf_counter() - t0:.1f}초, GPU 할당 {torch.cuda.memory_allocated() / 2**20:,.0f} MiB")
    print(f"설정: {settings}, 프롬프트 {PROMPT!r}")
    print("SFT 전 모델이라 대화가 아니라 '이어 쓰기'다. 사실이 틀릴 수 있다. exit 입력 시 종료\n")
    n = 0
    while True:
        try:
            q = input("질문> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        if q.lower() in QUIT:
            break
        t = time.perf_counter()
        text, reason, ntok = ask(model, tok, q, settings, a.seed + n, device)
        n += 1
        shown = text if a.raw else cut_answer(text)
        cut = "" if a.raw or shown == text else " · 뒤에 지어낸 질문·답변은 생략"
        print(f"답변>{shown.rstrip()}")
        print(f"      ({ntok} 토큰, {time.perf_counter() - t:.1f}초, 정지: {reason}{cut})\n")
    print("종료")
    return 0


if __name__ == "__main__":
    try:
        import readline  # noqa: F401  (한글 입력 편집·방향키)
    except ImportError:
        pass
    sys.exit(main())
