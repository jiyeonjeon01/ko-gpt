"""SFT 학습 데이터 생성 (Phase 2, Phase2_SFT계획_승인요청.md 2·3절)

    uv run python sft_data.py configs/sft_phase2_a2_noS11.json

순서:
  1. 출처별 대화 읽기 (A 그룹만): aya ko/en, oasst2 ko/en(1위 답변 경로), dolly-ko, S11(직접 작성, 선택)
  2. 같은 출처 안의 완전히 같은 대화 제거
  3. 평가 문항(chat120·ko20)과 비슷한 사용자 발화가 있는 대화 제외 (scripts/eval_overlap.EvalGuard)
  4. 설계안 4-3 템플릿으로 토큰화, 1,024 토큰 초과 제외 (자르지 않는다: 설계안 4-2)
  5. 출처별 Val 분리 (대화 id 해시 < val_frac, S11은 Val로 떼지 않음)
  6. 영어: 한국어 학습 응답 토큰(S11 반복 포함) × ratio를 넘지 않게, 지정 출처·조건에서 seed 고정 무작위 추출
  7. S11: {MODEL_NAME} 치환, epoch마다 repeat번 반복
  8. train.jsonl·val.jsonl(토큰 ID와 마스크), manifest.json(설정, 출처별 수·토큰, 제외 목록, 파일 SHA-256)

출력 폴더가 이미 있으면 멈춘다. 같은 설정·같은 원본이면 같은 결과가 나온다 (seed 고정, 해시 기반 분리).
"""

import argparse
import gzip
import hashlib
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))
from eval_overlap import LEAK_PATTERNS, EvalGuard, user_text  # noqa: E402
from sft import render_conversation  # noqa: E402
from tokenizer import encode, file_sha256, load_tokenizer  # noqa: E402

SAMPLE = ROOT / "data/sample"
SOURCE_FILES = {
    "S1_aya": [SAMPLE / "S1_aya/data/train-00000-of-00001.parquet", SAMPLE / "S1_aya/data/test-00000-of-00001.parquet"],
    "S2_oasst2": [SAMPLE / "S2_oasst2/2023-11-05_oasst2_ready.trees.jsonl.gz"],
    "S4_dolly_ko": [ROOT / "data/sft/S4_dolly_ko/databricks-dolly-15k-ko.jsonl"],
    "S10_dolly": [SAMPLE / "S10_dolly/databricks-dolly-15k.jsonl"],
}
S11_DIR = ROOT / "data/sft/S11"


# ---- 출처 읽기: (lang, messages) ----


def load_aya():
    for f in SOURCE_FILES["S1_aya"]:
        for r in pq.read_table(f).to_pylist():
            lang = {"kor": "ko", "eng": "en"}.get(r["language_code"])
            if lang:
                yield lang, [{"role": "user", "content": r["inputs"]}, {"role": "assistant", "content": r["targets"]}]


def load_oasst2():
    """트리마다 경로 하나: rank 1위 답, rank가 없으면 첫 답 (sft_stats.py와 같은 규칙)"""
    with gzip.open(SOURCE_FILES["S2_oasst2"][0], "rt", encoding="utf-8") as f:
        for line in f:
            tree = json.loads(line)
            lang = tree["prompt"].get("lang")
            if lang not in ("ko", "en"):
                continue
            node, messages = tree["prompt"], []
            while node is not None:
                messages.append({"role": "user" if node["role"] == "prompter" else "assistant", "content": node["text"]})
                replies = [r for r in node.get("replies", []) if not r.get("deleted")]
                ranked = [r for r in replies if r.get("rank") is not None]
                node = min(ranked, key=lambda r: r["rank"]) if ranked else (replies[0] if replies else None)
            while messages and messages[-1]["role"] != "assistant":
                messages.pop()
            if messages:
                yield lang, messages


def _dolly_rows(path: Path, lang: str):
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        ctx = r.get("context") or ""
        user = r["instruction"] + (f"\n\n{ctx}" if ctx else "")
        yield lang, [{"role": "user", "content": user}, {"role": "assistant", "content": r["response"]}]


def load_dolly_ko():
    path = SOURCE_FILES["S4_dolly_ko"][0]
    if not path.exists():
        raise FileNotFoundError(f"{path}가 없다. scripts/download_dolly_ko.py를 먼저 실행한다")
    yield from _dolly_rows(path, "ko")


def load_dolly_en():
    yield from _dolly_rows(SOURCE_FILES["S10_dolly"][0], "en")


def load_s11():
    files = sorted(S11_DIR.glob("*.jsonl")) if S11_DIR.exists() else []
    if not files:
        raise FileNotFoundError(f"{S11_DIR}/*.jsonl이 없다 (scripts/s11_convert.py로 만든다)")
    for f in files:
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            yield "ko", r["messages"]


LOADERS = {"S1_aya": load_aya, "S2_oasst2": load_oasst2, "S4_dolly_ko": load_dolly_ko, "S10_dolly": load_dolly_en,
           "S11_direct": load_s11}


# ---- 생성 ----


def conv_id(source: str, messages: list[dict]) -> str:
    blob = json.dumps(messages, ensure_ascii=False, sort_keys=True)
    return f"{source}:{hashlib.sha1(blob.encode('utf-8')).hexdigest()[:16]}"


def is_val(cid: str, frac: float) -> bool:
    return int(hashlib.sha256(cid.encode()).hexdigest()[:8], 16) / 2**32 < frac


def n_turns(messages: list[dict]) -> int:
    return sum(m["role"] == "assistant" for m in messages)


def substitute(messages: list[dict], model_name: str) -> list[dict]:
    return [{**m, "content": m["content"].replace("{MODEL_NAME}", model_name)} for m in messages]


def build(cfg: dict, tok_encode, eval_prompts: list[tuple[str, str]], loaders=None) -> dict:
    """반환: {"train": [...], "val": [...], "stats": ..., "excluded": ...}. 각 예: id, source, lang, turns, ids, mask"""
    loaders = loaders or LOADERS
    guard = EvalGuard(eval_prompts)
    max_len, val_frac = cfg["max_len"], cfg["val_frac"]
    stats = defaultdict(lambda: defaultdict(int))
    excluded = defaultdict(list)
    pools = defaultdict(lambda: {"train": [], "val": []})  # (group) → 예

    def take(group: str, source: str, lang: str, messages: list[dict], no_val: bool = False):
        key = f"{source}/{lang}"
        cid = conv_id(source, messages)
        st = stats[key]
        st["read"] += 1
        if cid in seen:
            st["excluded_duplicate"] += 1
            return
        seen.add(cid)
        hit = guard.hit(messages)
        if hit:
            st["excluded_eval_overlap"] += 1
            excluded["eval_overlap"].append({"id": cid, "eval_item": hit, "user": user_text({"messages": messages})[:120]})
            return
        try:
            ids, mask = render_conversation(messages, tok_encode)
        except ValueError as e:
            st["excluded_invalid"] += 1
            excluded["invalid"].append({"id": cid, "error": str(e)})
            return
        if len(ids) > max_len:
            st["excluded_over_max_len"] += 1
            return
        split = "train" if no_val or not is_val(cid, val_frac) else "val"
        ex = {"id": cid, "source": source, "lang": lang, "turns": n_turns(messages), "ids": ids, "mask": mask}
        pools[group][split].append(ex)

    seen: set[str] = set()
    for spec in cfg["korean"]:
        for lang, messages in loaders[spec["source"]]():
            if lang == "ko":
                take("korean", spec["source"], "ko", messages)
    s11 = cfg.get("s11", {})
    if s11.get("enabled"):
        for lang, messages in loaders["S11_direct"]():
            take("s11", "S11_direct", "ko", substitute(messages, cfg["model_name"]), no_val=True)
    eng = cfg.get("english")
    if eng:
        for lang, messages in loaders[eng["source"]]():
            if lang == "en" and (not eng.get("multi_turn_only") or n_turns(messages) >= 2):
                take("english_pool", eng["source"], "en", messages)

    loss = lambda exs: sum(sum(e["mask"]) for e in exs)  # noqa: E731
    repeat = int(s11.get("repeat", 1)) if s11.get("enabled") else 0
    korean_train = pools["korean"]["train"]
    s11_train = pools["s11"]["train"] * repeat
    korean_loss = loss(korean_train) + loss(s11_train)

    english_train, english_val = [], []
    english_target = 0
    if eng:
        english_target = int(eng["ratio_to_korean_loss_tokens"] * korean_loss)
        pool = list(pools["english_pool"]["train"])
        random.Random(f"{cfg['seed']}:english").shuffle(pool)
        acc = 0
        for ex in pool:
            if acc + sum(ex["mask"]) > english_target:
                continue  # 넘치는 대화는 건너뛰고 더 짧은 대화로 채운다 (목표를 넘지 않게)
            english_train.append(ex)
            acc += sum(ex["mask"])
        # Val도 같은 비율로 (영어 Val 풀에서 Train 선택 비율만큼)
        frac = len(english_train) / max(len(pool), 1)
        vpool = list(pools["english_pool"]["val"])
        random.Random(f"{cfg['seed']}:english_val").shuffle(vpool)
        english_val = vpool[:round(frac * len(vpool))]

    train = korean_train + s11_train + english_train
    val = pools["korean"]["val"] + english_val
    summary = {
        "korean_train_loss_tokens": korean_loss, "english_target_loss_tokens": english_target,
        "english_train_loss_tokens": loss(english_train),
        "english_pool_train_conversations": len(pools["english_pool"]["train"]),
        "s11_unique_conversations": len(pools["s11"]["train"]), "s11_repeat": repeat,
    }
    comp = defaultdict(lambda: defaultdict(int))
    for split, exs in (("train", train), ("val", val)):
        for e in exs:
            c = comp[f"{split}:{e['source']}/{e['lang']}"]
            c["conversations"] += 1
            c["multi_turn"] += e["turns"] > 1
            c["tokens"] += len(e["ids"])
            c["loss_tokens"] += sum(e["mask"])
    return {"train": train, "val": val, "summary": summary,
            "source_stats": {k: dict(v) for k, v in sorted(stats.items())},
            "composition": {k: dict(v) for k, v in sorted(comp.items())},
            "excluded": {k: v for k, v in excluded.items()}}


def write_jsonl(path: Path, rows: list[dict]) -> str:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({k: r[k] for k in ("id", "source", "lang", "turns", "ids", "mask")}) + "\n")
    return file_sha256(path)


def load_eval_prompts() -> tuple[list[tuple[str, str]], dict]:
    """chat120(초안·고정본 모두)과 ko20. 반환: 프롬프트 목록, 사용한 파일의 SHA-256"""
    prompts, files = [], {}
    for f in sorted((ROOT / "eval/chat120").glob("*.json")):
        files[str(f.relative_to(ROOT))] = file_sha256(f)
        prompts += [(f"chat120:{it['id']}", user_text(it)) for it in json.loads(f.read_text(encoding="utf-8"))["items"]]
    ko20 = ROOT / "eval/prompts_ko20.json"
    files[str(ko20.relative_to(ROOT))] = file_sha256(ko20)
    if LEAK_PATTERNS.exists():  # EvalGuard 3단계가 쓰는 바꿔 쓴 질문 정규식
        files[str(LEAK_PATTERNS.relative_to(ROOT))] = file_sha256(LEAK_PATTERNS)
    prompts += [(f"ko20:{p['id']}", p["text"]) for p in json.loads(ko20.read_text(encoding="utf-8"))["prompts"]]
    return list(dict.fromkeys(prompts)), files


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("config", type=Path)
    p.add_argument("--out-root", type=Path, default=ROOT / "data/sft/builds")
    a = p.parse_args(argv)
    cfg = json.loads(a.config.read_text(encoding="utf-8"))
    out = a.out_root / cfg["name"]
    if out.exists():
        raise FileExistsError(f"{out}가 이미 있다. 이름을 바꾸거나 기존 결과를 확인한다")
    data_cfg = json.loads((ROOT / "runs/phase0b/data_full.json").read_text(encoding="utf-8"))
    tok_path = ROOT / data_cfg["tokenizer"]
    tok = load_tokenizer(tok_path)
    prompts, eval_files = load_eval_prompts()
    t0 = time.time()
    res = build(cfg, lambda t: encode(tok, t), prompts)
    out.mkdir(parents=True)
    manifest = {
        "config": cfg, "config_file": str(a.config), "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tokenizer": str(tok_path), "tokenizer_sha256": file_sha256(tok_path),
        "source_files": {s: [{"path": str(f.relative_to(ROOT)), "sha256": file_sha256(f)} for f in fs if f.exists()]
                         for s, fs in SOURCE_FILES.items()},
        "eval_files_excluded_against": eval_files,
        "eval_guard_version": EvalGuard.VERSION,
        "train_sha256": write_jsonl(out / "train.jsonl", res["train"]),
        "val_sha256": write_jsonl(out / "val.jsonl", res["val"]),
        "summary": res["summary"], "composition": res["composition"], "source_stats": res["source_stats"],
        "excluded_counts": {k: len(v) for k, v in res["excluded"].items()}, "seconds": round(time.time() - t0, 1),
    }
    (out / "excluded.json").write_text(json.dumps(res["excluded"], ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("summary", "composition", "excluded_counts")}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
