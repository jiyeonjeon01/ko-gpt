"""대화 평가 세트 응답 생성 (설계안 4-3 템플릿, SFT 전후 같은 설정)

    uv run python chat_generate.py runs/phase2/sft42m/last.pt eval/chat120/frozen_v1.json --out runs/phase2/sft42m/gen_post

- 입력: <|endoftext|> + 대화(역할 머리말·<|im_end|>) + "<|im_start|>assistant\\n" (sft.prompt_ids, 학습 때와 같은 토큰 경계)
- 정지: <|im_end|> (정상), <|endoftext|> 또는 <|im_start|> (템플릿 이탈), max_new_tokens (멈추지 않음)
- 사전학습 Checkpoint에도 같은 템플릿을 붙여 생성한다 (설계안 9-2 "SFT 전후 비교")
- 고정되지 않은 평가 세트(status != "frozen")는 거부한다. 시험용 작은 세트는 --allow-unfrozen (chat120 초안은 허용하지 않음)
- 출력 폴더가 이미 있으면 멈춘다
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

from filecache import drop_cache

from config import GPTConfig
from model import GPT
from sft import prompt_ids
from text import EOT_ID, IM_END_ID, IM_START_ID
from tokenizer import decode, encode, file_sha256, load_tokenizer

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))
from freeze_eval_set import verify  # noqa: E402

SETTINGS = {"temperature": 0.7, "top_k": 50, "seed": 1234, "max_new_tokens": 256}
STOP_TOKENS = {IM_END_ID: "im_end", EOT_ID: "eot", IM_START_ID: "im_start"}


@torch.no_grad()
def generate(model, ids: list[int], max_new_tokens: int, temperature: float, top_k: int, gen: torch.Generator,
             device: str) -> tuple[list[int], str]:
    """반환: 생성 토큰(정지 토큰 제외), 정지 이유"""
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out = []
    block = model.config.block_size
    for _ in range(max_new_tokens):
        logits, _ = model(idx[:, -block:])
        logits = logits[0, -1].float().cpu() / temperature
        values, indices = torch.topk(logits, min(top_k, logits.numel()))
        token = int(indices[torch.multinomial(torch.softmax(values, -1), 1, generator=gen)])
        if token in STOP_TOKENS:
            return out, STOP_TOKENS[token]
        out.append(token)
        idx = torch.cat([idx, torch.tensor([[token]], device=device)], dim=1)
    return out, "max_tokens"


def load_model(path: Path, device: str):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    drop_cache(path)  # 읽은 Checkpoint를 WSL 파일 캐시에서 뺀다 (filecache.py)
    if ck.get("kind") == "sft":
        tok_path = ck["run"]["tokenizer"]
    else:
        tok_path = json.loads(Path(ck["run"]["data"]).read_text(encoding="utf-8"))["tokenizer"]
    model = GPT(GPTConfig(**ck["model_config"]))
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    return model, load_tokenizer(tok_path), ck


def check_eval_set(path: Path, allow_unfrozen: bool) -> dict:
    doc = json.loads(path.read_text(encoding="utf-8"))
    if doc.get("status") == "frozen":
        if not verify(path):
            raise ValueError(f"{path}: 고정 이후 내용이 바뀌었다 (해시 불일치)")
        return doc
    if doc.get("name") == "chat120":
        raise PermissionError("chat120은 고정본(frozen_v*.json)으로만 생성한다 (eval/chat120/README.md 규칙)")
    if not allow_unfrozen:
        raise PermissionError(f"{path}는 고정본이 아니다. 시험용이면 --allow-unfrozen")
    return doc


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", type=Path)
    p.add_argument("eval_set", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--allow-unfrozen", action="store_true", help="시험용 평가 세트 (chat120 초안은 불가)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args(argv)
    if a.out.exists():
        raise FileExistsError(f"{a.out}가 이미 있다. 기존 결과를 덮어쓰지 않는다")
    doc = check_eval_set(a.eval_set, a.allow_unfrozen)
    model, tok, ck = load_model(a.checkpoint, a.device)
    a.out.mkdir(parents=True)
    t0 = time.time()
    with open(a.out / "generations.jsonl", "w", encoding="utf-8") as f:
        for it in doc["items"]:
            ids = prompt_ids(it["messages"], lambda t: encode(tok, t))
            gen = torch.Generator().manual_seed(SETTINGS["seed"])
            out, why = generate(model, ids, SETTINGS["max_new_tokens"], SETTINGS["temperature"], SETTINGS["top_k"],
                                gen, a.device)
            f.write(json.dumps({"id": it["id"], "category": it.get("category"), "stop_reason": why,
                                "n_tokens": len(out), "completion": decode(tok, out), "completion_ids": out},
                               ensure_ascii=False) + "\n")
    meta = {"checkpoint": str(a.checkpoint), "checkpoint_sha256": file_sha256(a.checkpoint),
            "checkpoint_kind": ck.get("kind", "pretrain"), "step": ck["step"], "eval_set": str(a.eval_set),
            "eval_set_sha256": file_sha256(a.eval_set), "eval_status": doc.get("status"),
            "items_sha256": doc.get("items_sha256"), "settings": SETTINGS, "device": a.device,
            "seconds": round(time.time() - t0, 1), "created": time.strftime("%F %T")}
    (a.out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
