"""SFT 빌드와 평가 세트의 오염 추가 점검 (학습 전)

    uv run python scripts/eval_contamination.py data/sft/builds/phase2_a2_noS11_v2 eval/chat120/draft_v0.2.json \
        --out runs/phase2/prep/eval_contamination_v2.json

1. 문장 단위 질문 겹침: 빌드의 모든 대화(train·val)에서 user·assistant 발화와 참고 글을 문장으로 나눈다.
   각 문장을 평가 문항의 사용자 발화와 비교한다 (eval_overlap의 같은 규칙: 3-gram Jaccard ≥ 0.5 또는 포함률 ≥ 0.8, 길이 비 ≤ 4).
   → 참고 글이나 응답 안에 평가 질문이 들어 있는 경우를 찾는다.
2. 사실 노출: 사실형 문항(상식 사실, 영어 사실)마다 질문 핵심어와 정답이 한 문장 안에 함께 나오는 대화를 센다.
   평가 질문이 아니라 같은 지식이 학습 데이터에 있다는 뜻이다. 제외 대상은 아니고 해석용으로 기록한다.
빌드 파일에는 토큰만 있으므로 원본 출처를 다시 읽어 conv_id로 대화 본문을 찾는다.
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import sft_data as D  # noqa: E402
from eval_overlap import flagged, grams, normalize, scores, user_text  # noqa: E402

# 사실 노출: 한 문장 안에 모든 정규식이 맞으면 노출 (소문자 원문)
FACTS = {
    "f01": [r"수도", r"서울", r"대한민국|한국"],
    "f02": [r"한글|훈민정음", r"세종", r"만들|창제|발명"],
    "f03": [r"가장 큰 (바다|대양)|최대의 (바다|대양)", r"태평양"],
    "f04": [r"1년|일 ?년|한 ?해", r"12 ?개월|열두 ?달"],
    "f05": [r"끓", r"100 ?(도|℃|°c)"],
    "f06": [r"가장 큰 행성", r"목성"],
    "f07": [r"일본", r"수도", r"도쿄"],
    "f08": [r"무지개", r"(일곱|7) ?가지"],
    "f09": [r"가장 높은 산", r"한라산"],
    "f10": [r"심장", r"(4|네) ?개의 (방|공간)|2 ?심방|심방.*심실"],
    "f11": [r"광복절", r"8월 ?15일"],
    "f12": [r"하루", r"24 ?시간"],
    "f13": [r"얼음", r"녹", r"(?<![가-힣])물"],
    "f14": [r"독도", r"동해"],
    "f15": [r"알파벳", r"26|스물여섯"],
    "g03": [r"capital of france|france's capital", r"paris"],
    "g05": [r"days (in|of) a week|days a week|week has", r"\b(7|seven)\b"],
    "g10": [r"sky", r"blue", r"clear|why is the sky"],
    "g13": [r"romeo and juliet", r"shakespeare"],
}

SENT_SPLIT = re.compile(r"(?<=[.?!。])\s+|\n+")


def sentences(text: str) -> list[str]:
    return [s.strip() for s in SENT_SPLIT.split(text) if len(s.strip()) >= 4]


def load_build_messages(build: Path) -> dict[str, tuple[str, list[dict]]]:
    need = {}
    for split in ("train", "val"):
        for line in (build / f"{split}.jsonl").read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            need[r["id"]] = split
    sources = {cid.split(":")[0] for cid in need}
    out = {}
    for src in sorted(sources):
        for _lang, m in D.LOADERS[src]():
            cid = D.conv_id(src, m)
            if cid in need and cid not in out:
                out[cid] = (need[cid], m)
    missing = set(need) - set(out)
    if missing:
        raise RuntimeError(f"원본에서 찾지 못한 대화 {len(missing)}개 (예: {sorted(missing)[:3]})")
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("build", type=Path)
    p.add_argument("eval_set", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--examples", type=int, default=3)
    a = p.parse_args(argv)
    if a.out.exists():
        raise FileExistsError(f"{a.out}가 이미 있다")
    items = json.loads(a.eval_set.read_text(encoding="utf-8"))["items"]
    convs = load_build_messages(a.build)

    # 1. 문장 단위 질문 겹침 (3-gram 역색인으로 후보를 좁힌다)
    ev = []
    index = defaultdict(set)
    for k, it in enumerate(items):
        q = user_text(it)
        for part in [q] + [m["content"] for m in it["messages"] if m["role"] == "user"]:
            g = grams(part)
            ev.append((it["id"], g, normalize(part)))
            for x in g:
                index[x].add(len(ev) - 1)
    question_hits = defaultdict(list)
    n_sent = 0
    for cid, (split, msgs) in convs.items():
        for m in msgs:
            for s in sentences(m["content"]):
                n_sent += 1
                sg, sn = grams(s), normalize(s)
                cand = set()
                for x in sg:
                    cand |= index.get(x, set())
                for e in cand:
                    eid, g, en = ev[e]
                    j, c = scores(g, sg)
                    if en == sn or flagged(j, c, len(g), len(sg)):
                        if not any(h["id"] == cid for h in question_hits[eid]):
                            question_hits[eid].append({"id": cid, "split": split, "role": m["role"], "sentence": s[:160],
                                                       "jaccard": round(j, 3), "contain": round(c, 3)})

    # 2. 사실 노출
    fact = {}
    for eid, pats in FACTS.items():
        rx = [re.compile(x) for x in pats]
        hits = []
        for cid, (split, msgs) in convs.items():
            for m in msgs:
                found = next((s for s in sentences(m["content"]) if all(r.search(s.lower()) for r in rx)), None)
                if found:
                    hits.append({"id": cid, "split": split, "role": m["role"], "sentence": found[:200]})
                    break
        fact[eid] = {"patterns": pats, "conversations": len(hits), "examples": hits[:a.examples]}

    res = {"build": str(a.build), "eval_set": str(a.eval_set), "conversations": len(convs), "sentences": n_sent,
           "question_overlap": dict(question_hits), "fact_exposure": fact}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"대화 {len(convs)} | 문장 {n_sent} | 문장 단위 질문 겹침 문항 {sorted(question_hits)}")
    for eid, v in fact.items():
        print(f"  {eid}: 사실 노출 {v['conversations']}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    finally:
        # 읽은 원본·쓴 결과가 WSL 파일 캐시로 남아 Windows 메모리를 차지하지 않게 뺀다 (이 저장소의 data/만, filecache.py)
        _r = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(_r))
        from filecache import drop_cache_under
        drop_cache_under(_r / "data" / "sample", _r / "data" / "sft")
