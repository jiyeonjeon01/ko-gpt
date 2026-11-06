"""고정 프롬프트 이어 쓰기 (사전학습 모델 확인용, 대화 형식 아님)

Checkpoint마다 같은 프롬프트·seed·샘플링 설정으로 생성해 학습 전후를 비교한다.
문서는 학습 때처럼 <|endoftext|>로 시작하고, <|endoftext|>가 나오면 멈춘다.

    uv run python generate.py runs/phase1/pilot42m/init.pt runs/phase1/pilot42m/last.pt --out runs/phase1/pilot42m/samples.md
"""

import argparse
import json
from pathlib import Path

import torch

from config import GPTConfig
from model import GPT
from text import EOT_ID
from tokenizer import decode, encode, load_tokenizer

PROMPTS = [
    "대한민국의 수도는",
    "오늘 날씨가 좋아서",
    "광합성은 식물이",
    "인공지능이란",
    "안녕하세요. 저는",
    "조선 시대에는",
    "The capital of France is",
    "Once upon a time,",
    "def fibonacci(n):",
]
SETTINGS = {"max_new_tokens": 60, "temperature": 0.8, "top_k": 50, "seed": 1234}


@torch.no_grad()
def generate(model: GPT, ids: list[int], max_new_tokens: int, temperature: float, top_k: int,
             gen: torch.Generator) -> list[int]:
    idx = torch.tensor([ids], dtype=torch.long)
    out = []
    for _ in range(max_new_tokens):
        logits, _ = model(idx[:, -model.config.block_size:])
        logits = logits[0, -1].float() / temperature
        values, indices = torch.topk(logits, top_k)
        probs = torch.softmax(values, dim=-1)
        token = int(indices[torch.multinomial(probs, 1, generator=gen)])
        if token == EOT_ID:
            break
        out.append(token)
        idx = torch.cat([idx, torch.tensor([[token]])], dim=1)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("checkpoints", nargs="+", type=Path)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    results = []
    for path in args.checkpoints:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        data = json.loads(Path(ckpt["run"]["data"]).read_text(encoding="utf-8"))
        tok = load_tokenizer(data["tokenizer"])
        model = GPT(GPTConfig(**ckpt["model_config"]))
        model.load_state_dict(ckpt["model"])
        model.eval()
        for prompt in PROMPTS:
            gen = torch.Generator().manual_seed(SETTINGS["seed"])
            ids = generate(model, [EOT_ID] + encode(tok, prompt), SETTINGS["max_new_tokens"], SETTINGS["temperature"],
                           SETTINGS["top_k"], gen)
            results.append({"checkpoint": str(path), "step": ckpt["step"], "tokens_seen": ckpt["tokens_seen"],
                            "prompt": prompt, "completion": decode(tok, ids)})
    lines = [f"# 고정 프롬프트 생성 결과\n\n설정: {json.dumps(SETTINGS)}, 학습 문서처럼 `<|endoftext|>`로 시작\n"]
    for prompt in PROMPTS:
        lines.append(f"\n## {prompt}\n")
        for r in results:
            if r["prompt"] == prompt:
                text = r["completion"].replace("\n", " ⏎ ")
                lines.append(f"- **step {r['step']}** ({r['tokens_seen'] / 1e6:,.0f}M 토큰): {prompt}**{text}**")
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    args.out.with_suffix(".json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
