"""KoBEST 5종·영어 HellaSwag zero-shot 선택지 로그우도 평가 (lm-evaluation-harness 방식 재현)

참조: EleutherAI/lm-evaluation-harness main d6de81643928d653435c431bae19945d41d32520 (2026-11-07 확인)
  - lm_eval/tasks/kobest/*.yaml, utils.py   프롬프트·선택지·지표
  - lm_eval/tasks/hellaswag/utils.py        영어 HellaSwag 전처리
  - lm_eval/api/task.py  construct_requests  (context, target_delimiter + choice), target_delimiter 기본 " "
  - lm_eval/api/task.py  process_results     acc = argmax(ll), acc_norm = argmax(ll / len(choice))  (문자 수)
  - lm_eval/api/model.py _encode_pair        문맥 끝 공백을 선택지 쪽으로 옮기고,
                                             whole = encode(ctx + cont), ctx_enc = encode(ctx), cont_enc = whole[len(ctx_enc):]
  - lm_eval/models/huggingface.py            입력 = (ctx_enc + cont_enc)[-(max_len + 1):][:-1], 선택지 토큰 log_softmax 합
  - lm_eval/api/metrics.py                   mean_stderr = 표본표준편차 / sqrt(n)
  - kobest utils.macro_f1_score              sklearn f1_score(average="macro")

lm-eval과 다른 점 (기록용):
  - 문맥 앞에 아무 토큰도 붙이지 않는 것이 lm-eval 기본(add_bos_token=False)과 같다. 우리 모델은 학습 때 문서 앞에
    <|endoftext|>를 두므로 --prefix-eot로 붙인 결과도 민감도 확인용으로 따로 낼 수 있다.
  - 인코딩은 이 프로젝트의 유일한 입구 tokenizer.encode()(preprocess 포함)를 쓴다.

    uv run python eval_bench.py runs/phase1/pilot42m/last.pt --out runs/phase1/pilot42m/eval_bench
"""

import argparse
import hashlib
import json
import math
import random
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

from checkpoint_io import checkpoint_sha256, load_weights, meta_of, release
from config import GPTConfig
from model import GPT
from text import EOT_ID
from tokenizer import encode, file_sha256, load_tokenizer

LM_EVAL_REF = "EleutherAI/lm-evaluation-harness@d6de81643928d653435c431bae19945d41d32520"
TARGET_DELIMITER = " "
DATA = Path("data/sample")


@dataclass
class Doc:
    context: str
    choices: list[str]
    gold: int


# ---- 과제 정의 (lm-eval 템플릿 그대로) ----


def kobest_boolq(r: dict) -> Doc:
    return Doc(f"{r['paragraph']} 질문: {r['question']} 답변: ", ["아니오", "예"], int(r["label"]))


def kobest_copa(r: dict) -> Doc:
    connector = {"원인": " 왜냐하면", "결과": " 그래서"}[r["question"].strip()]
    return Doc(f"{r['premise']} {connector}", [r["alternative_1"], r["alternative_2"]], int(r["label"]))


def kobest_wic(r: dict) -> Doc:
    return Doc(f"문장1: {r['context_1']} 문장2: {r['context_2']} 두 문장에서 {r['word']}가 같은 뜻으로 쓰였나?",
               ["아니오", "예"], int(r["label"]))


def kobest_hellaswag(r: dict) -> Doc:
    return Doc(f"문장: {r['context']}", [r["ending_1"], r["ending_2"], r["ending_3"], r["ending_4"]], int(r["label"]))


def kobest_sentineg(r: dict) -> Doc:
    return Doc(f"문장: {r['sentence']} 긍부정:", ["부정", "긍정"], int(r["label"]))


def hellaswag_preprocess(text: str) -> str:
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    return text.replace("  ", " ")


def hellaswag(r: dict) -> Doc:
    ctx = r["ctx_a"] + " " + r["ctx_b"].capitalize()
    return Doc(hellaswag_preprocess(r["activity_label"] + ": " + ctx),
               [hellaswag_preprocess(e) for e in r["endings"]], int(r["label"]))


TASKS = {
    # 이름: (파일, 변환, 지표)
    "kobest_boolq": (DATA / "E_kobest/boolq/test.jsonl", kobest_boolq, ("acc", "f1")),
    "kobest_copa": (DATA / "E_kobest/copa/test.jsonl", kobest_copa, ("acc", "f1")),
    "kobest_hellaswag": (DATA / "E_kobest/hellaswag/test.jsonl", kobest_hellaswag, ("acc", "acc_norm", "f1")),
    "kobest_sentineg": (DATA / "E_kobest/sentineg/test.jsonl", kobest_sentineg, ("acc", "f1")),
    "kobest_wic": (DATA / "E_kobest/wic/test.jsonl", kobest_wic, ("acc", "f1")),
    "hellaswag": (DATA / "E_hellaswag/data/validation-00000-of-00001.parquet", hellaswag, ("acc", "acc_norm")),
}


def load_docs(task: str) -> list[Doc]:
    path, fn, _ = TASKS[task]
    if path.suffix == ".parquet":
        rows = pq.read_table(path).to_pylist()
    else:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [fn(r) for r in rows]


# ---- 토큰화와 점수 ----


def encode_pair(tok, context: str, continuation: str) -> tuple[list[int], list[int], bool]:
    """lm-eval _encode_pair. 세 번째 값: ctx_enc가 whole_enc의 앞부분과 정확히 같은지 (경계 병합 진단)"""
    n_spaces = len(context) - len(context.rstrip())
    if n_spaces > 0:
        continuation = context[-n_spaces:] + continuation
        context = context[:-n_spaces]
    whole = encode(tok, context + continuation)
    ctx = encode(tok, context)
    return ctx, whole[len(ctx):], whole[:len(ctx)] == ctx


@torch.no_grad()
def score(model, requests: list[tuple[list[int], list[int]]], max_len: int, device: str,
          batch_tokens: int = 16384) -> list[tuple[float, bool]]:
    """각 (ctx_enc, cont_enc)의 선택지 로그우도 합과 greedy 일치 여부. 오른쪽 패딩 (causal이라 앞 위치에 영향 없음)"""
    items = []
    for i, (ctx, cont) in enumerate(requests):
        if not cont:
            raise ValueError(f"요청 {i}: 선택지 토큰이 없다")
        if not ctx:
            raise ValueError(f"요청 {i}: 문맥 토큰이 없다 (lm-eval은 이 경우 prefix 토큰을 쓴다)")
        if len(cont) > max_len:
            raise ValueError(f"요청 {i}: 선택지가 문맥 길이 {max_len}보다 길다")
        inp = (ctx + cont)[-(max_len + 1):][:-1]
        items.append((i, inp, cont))
    out: list[tuple[float, bool] | None] = [None] * len(items)
    items.sort(key=lambda t: -len(t[1]))  # 길이순으로 묶어 패딩 낭비를 줄인다
    pos = 0
    while pos < len(items):
        width = len(items[pos][1])
        n = max(1, batch_tokens // width)
        chunk = items[pos:pos + n]
        pos += n
        x = torch.zeros((len(chunk), width), dtype=torch.long)
        for b, (_, inp, _) in enumerate(chunk):
            x[b, :len(inp)] = torch.tensor(inp)
        logits, _ = model(x.to(device))
        logp = F.log_softmax(logits.float(), dim=-1)
        for b, (i, inp, cont) in enumerate(chunk):
            n_cont = len(cont)
            sl = logp[b, len(inp) - n_cont:len(inp)]  # 선택지 각 토큰을 예측하는 위치
            target = torch.tensor(cont, device=sl.device)
            ll = float(sl.gather(1, target[:, None]).sum())
            greedy = bool((sl.argmax(-1) == target).all())
            out[i] = (ll, greedy)
    return out  # type: ignore[return-value]


# ---- 지표 ----


def macro_f1(golds: list[int], preds: list[int]) -> float:
    """sklearn f1_score(average="macro")와 같다: 정답·예측에 나온 클래스의 F1 평균, 정의되지 않으면 0"""
    labels = sorted(set(golds) | set(preds))
    f1s = []
    for c in labels:
        tp = sum(1 for g, p in zip(golds, preds) if g == c and p == c)
        fp = sum(1 for g, p in zip(golds, preds) if g != c and p == c)
        fn = sum(1 for g, p in zip(golds, preds) if g == c and p != c)
        f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return sum(f1s) / len(f1s)


def mean_stderr(xs: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return float("nan")
    mu = sum(xs) / n
    return math.sqrt(sum((x - mu) ** 2 for x in xs) / (n - 1)) / math.sqrt(n)


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def binom_two_sided_p(k: int, n: int, p: float) -> float:
    """정확 이항검정 (양측, 관측값 이하 확률의 합)"""
    def logpmf(i):
        return math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * math.log(p) + (n - i) * math.log1p(-p)
    obs = logpmf(k)
    total = sum(math.exp(lp) for i in range(n + 1) if (lp := logpmf(i)) <= obs + 1e-9)
    return min(1.0, total)


def bootstrap_f1(golds: list[int], preds: list[int], reps: int = 1000, seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(golds)
    vals = []
    for _ in range(reps):
        idx = [rng.randrange(n) for _ in range(n)]
        vals.append(macro_f1([golds[i] for i in idx], [preds[i] for i in idx]))
    vals.sort()
    return vals[int(0.025 * reps)], vals[int(0.975 * reps) - 1]


def summarize(task: str, docs: list[Doc], lls: list[list[float]]) -> tuple[dict, list[dict]]:
    metrics = TASKS[task][2]
    k = len(docs[0].choices)
    golds = [d.gold for d in docs]
    preds = [max(range(k), key=lambda j: ll[j]) for ll in lls]  # np.argmax와 같이 동점이면 앞쪽
    preds_norm = [max(range(k), key=lambda j: ll[j] / len(d.choices[j])) for d, ll in zip(docs, lls)]
    n = len(docs)
    res = {"n": n, "num_choices": k, "chance": 1 / k}
    gold_counts = [golds.count(c) for c in range(k)]
    res["majority_baseline"] = max(gold_counts) / n
    res["gold_distribution"] = gold_counts
    res["pred_distribution"] = [preds.count(c) for c in range(k)]
    for name, pr in [("acc", preds), ("acc_norm", preds_norm)]:
        if name not in metrics:
            continue
        hits = [1.0 if p == g else 0.0 for p, g in zip(pr, golds)]
        c = int(sum(hits))
        lo, hi = wilson(c, n)
        res[name] = {"value": c / n, "correct": c, "stderr": mean_stderr(hits), "wilson95": [lo, hi],
                     "minus_chance": c / n - 1 / k, "binom_p_vs_chance": binom_two_sided_p(c, n, 1 / k)}
        if name == "acc_norm":
            res[name]["pred_distribution"] = [pr.count(c_) for c_ in range(k)]
    if "f1" in metrics:
        lo, hi = bootstrap_f1(golds, preds)
        res["f1"] = {"value": macro_f1(golds, preds), "bootstrap95": [lo, hi]}
    items = [{"i": i, "gold": g, "lls": ll, "pred": p, "pred_norm": pn,
              "choice_chars": [len(c) for c in d.choices]}
             for i, (d, g, ll, p, pn) in enumerate(zip(docs, golds, lls, preds, preds_norm))]
    return res, items


def git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True,
                               text=True, timeout=10).stdout.strip()
        return out + ("-dirty" if dirty else "")
    except OSError:
        return "unknown"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS))
    p.add_argument("--prefix-eot", action="store_true", help="문맥 앞에 <|endoftext|> (lm-eval 기본은 붙이지 않음)")
    p.add_argument("--limit", type=int, help="과제마다 앞 N문항만 (시험용)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError(f"{a.out}가 이미 있다. 기존 결과를 덮어쓰지 않는다")

    w = load_weights(a.checkpoint)  # 모델 가중치만 mmap으로 (checkpoint_io.py)
    data_cfg = json.loads(Path(w["run"]["data"]).read_text(encoding="utf-8"))
    tok = load_tokenizer(data_cfg["tokenizer"])
    model = GPT(GPTConfig(**w["model_config"]))
    model.load_state_dict(w["model"])
    ckpt = meta_of(w)
    del w
    release(a.checkpoint)  # filecache.py
    model.to(a.device).eval()  # fp32 그대로 (lm-eval dtype="auto"는 체크포인트 dtype)
    max_len = model.config.block_size

    a.out.mkdir(parents=True)
    meta = {"checkpoint": str(a.checkpoint), "checkpoint_sha256": checkpoint_sha256(a.checkpoint), "step": ckpt["step"],
            "tokens_seen": ckpt["tokens_seen"], "code_commit": git_commit(), "reference": LM_EVAL_REF,
            "target_delimiter": TARGET_DELIMITER, "prefix_eot": a.prefix_eot, "dtype": "float32",
            "device": a.device, "limit": a.limit,
            "data_files": {t: {"path": str(TASKS[t][0]), "sha256": file_sha256(TASKS[t][0])} for t in a.tasks}}
    summary = {}
    for task in a.tasks:
        t0 = time.time()
        docs = load_docs(task)[:a.limit] if a.limit else load_docs(task)
        reqs, boundary_ok = [], 0
        for d in docs:
            for c in d.choices:
                ctx, cont, ok = encode_pair(tok, d.context, TARGET_DELIMITER + c)
                boundary_ok += ok
                reqs.append(([EOT_ID] + ctx if a.prefix_eot else ctx, cont))
        scored = score(model, reqs, max_len, a.device)
        k = len(docs[0].choices)
        lls = [[scored[i * k + j][0] for j in range(k)] for i in range(len(docs))]
        res, items = summarize(task, docs, lls)
        res["requests"] = len(reqs)
        res["boundary_prefix_ok"] = boundary_ok
        res["seconds"] = round(time.time() - t0, 1)
        for it, i in zip(items, range(len(docs))):
            it["greedy"] = [scored[i * k + j][1] for j in range(k)]
        with open(a.out / f"{task}.jsonl", "w", encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
        summary[task] = res
        line = f"{task}: n={res['n']}"
        for m in ("acc", "acc_norm"):
            if m in res:
                line += f" {m}={res[m]['value']:.4f}±{res[m]['stderr']:.4f} (p={res[m]['binom_p_vs_chance']:.3g})"
        if "f1" in res:
            line += f" f1={res['f1']['value']:.4f}"
        print(line + f" | 예측 분포 {res['pred_distribution']} | {res['seconds']}s", flush=True)
    (a.out / "summary.json").write_text(json.dumps({"meta": meta, "results": summary}, ensure_ascii=False, indent=1),
                                        encoding="utf-8")


if __name__ == "__main__":
    main()
