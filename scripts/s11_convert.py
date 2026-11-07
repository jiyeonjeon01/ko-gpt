"""S11 직접 작성 대화: 텍스트 형식 → JSONL 변환과 검사 (docs/S11_작성가이드.md)

    uv run python scripts/s11_convert.py data/sft/S11/batch01.md --out data/sft/S11/batch01.jsonl

오류가 하나라도 있으면 JSONL을 쓰지 않고 종료 코드 1. 경고는 출력만 한다.
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_overlap import is_near_duplicate, user_text  # noqa: E402

CATEGORIES = {"인사", "일상 대화", "감정", "정체성", "모름·한계", "기타"}
MAX_REPLY_CHARS = 400
HEADER = re.compile(r"^###\s*(s11-\d{4})\s*\|\s*(.+?)\s*$")
TURN = re.compile(r"^(U|A):\s?(.*)$")
EMOJI = re.compile("[\U0001F300-\U0001FAFF\u2600-\u27BF]")
BRACES = re.compile(r"\{[^}]*\}")
# 해요체 끝맺음 (문장부호·따옴표 앞). 완벽한 판정이 아니라 경고용
POLITE_END = re.compile(r"(요|죠|니다|니까|세요|예요|에요|이에요|어요|아요|해요|군요|네요|까요)[.!?…\"')\s]*$")
GUIDE_EXAMPLES = {
    "s11-0001": "오랜만이에요! 그동안 잘 지내셨어요?",
    "s11-0002": "저는 {MODEL_NAME}이라는 한국어 대화 모델이에요.",
    "s11-0003": "고생 많으셨어요! 발표는 어떻게 됐어요?",
}


def parse(text: str) -> tuple[list[dict], list[str]]:
    convs, errors, cur, last = [], [], None, None
    for n, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        h = HEADER.match(line)
        if h:
            cur = {"id": h.group(1), "category": h.group(2), "messages": [], "line": n}
            convs.append(cur)
            last = None
            continue
        if line.startswith("###"):
            errors.append(f"{n}행: 머리줄 형식이 틀림 (### s11-0000 | 범주): {line[:40]}")
            cur = None
            continue
        if cur is None:
            errors.append(f"{n}행: 대화 머리줄(###) 앞의 내용")
            continue
        t = TURN.match(line)
        if t:
            role = "user" if t.group(1) == "U" else "assistant"
            last = {"role": role, "content": t.group(2).strip()}
            cur["messages"].append(last)
        elif last is not None:
            last["content"] = (last["content"] + "\n" + line.strip()).strip()
        else:
            errors.append(f"{n}행: U:/A: 없이 시작한 줄")
    return convs, errors


def check(convs: list[dict], eval_prompts: list[tuple[str, str]]) -> tuple[list[str], list[str]]:
    errors, warnings, seen = [], [], set()
    for c in convs:
        cid = c["id"]
        if cid in seen:
            errors.append(f"{cid}: id 중복")
        seen.add(cid)
        if c["category"] not in CATEGORIES:
            errors.append(f"{cid}: 범주 '{c['category']}' (가능: {', '.join(sorted(CATEGORIES))})")
        roles = [m["role"] for m in c["messages"]]
        if not roles or roles[-1] != "assistant" or any(r != ("user" if i % 2 == 0 else "assistant")
                                                         for i, r in enumerate(roles)):
            errors.append(f"{cid}: U로 시작해 번갈아 A로 끝나야 함 (현재 {''.join('U' if r == 'user' else 'A' for r in roles)})")
        for m in c["messages"]:
            if not m["content"]:
                errors.append(f"{cid}: 빈 발화")
            if EMOJI.search(m["content"]):
                warnings.append(f"{cid}: 이모지")
            for b in BRACES.findall(m["content"]):
                if b != "{MODEL_NAME}":
                    warnings.append(f"{cid}: 알 수 없는 자리표시 {b}")
            if m["role"] == "assistant":
                if len(m["content"]) > MAX_REPLY_CHARS:
                    warnings.append(f"{cid}: 응답 {len(m['content'])}자 (권장 {MAX_REPLY_CHARS}자 미만)")
                if m["content"] and not POLITE_END.search(m["content"]):
                    warnings.append(f"{cid}: 해요체로 끝나지 않음: …{m['content'][-15:]}")
        if cid in GUIDE_EXAMPLES and any(GUIDE_EXAMPLES[cid] in m["content"] for m in c["messages"]):
            warnings.append(f"{cid}: 가이드 형식 예시를 그대로 사용 (예시는 학습에 넣지 않는다)")
        text = user_text(c)
        for eid, prompt in eval_prompts:
            if is_near_duplicate(prompt, text) or any(is_near_duplicate(prompt, m["content"])
                                                      for m in c["messages"] if m["role"] == "user"):
                errors.append(f"{cid}: 평가 문항 {eid}와 비슷함 ('{prompt[:30]}')")
                break
    return errors, warnings


def load_eval_prompts(root: Path) -> list[tuple[str, str]]:
    out = []
    for name in sorted((root / "eval" / "chat120").glob("*.json")):
        for it in json.loads(name.read_text(encoding="utf-8"))["items"]:
            out.append((f"chat120:{it['id']}", user_text(it)))
    ko20 = root / "eval" / "prompts_ko20.json"
    if ko20.exists():
        out += [(f"ko20:{p['id']}", p["text"]) for p in json.loads(ko20.read_text(encoding="utf-8"))["prompts"]]
    return list(dict.fromkeys(out))


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("src", type=Path)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    convs, errors = parse(a.src.read_text(encoding="utf-8"))
    e2, warnings = check(convs, load_eval_prompts(Path(__file__).resolve().parent.parent))
    errors += e2
    for w in warnings:
        print("경고:", w)
    for e in errors:
        print("오류:", e)
    multi = sum(1 for c in convs if len(c["messages"]) > 2)
    print(f"대화 {len(convs)}개 (다중 턴 {multi}개), 오류 {len(errors)}, 경고 {len(warnings)}")
    if errors:
        print("오류가 있어 JSONL을 쓰지 않는다")
        return 1
    if a.out.exists():
        print(f"{a.out}가 이미 있다. 덮어쓰지 않는다")
        return 1
    with open(a.out, "w", encoding="utf-8") as f:
        for c in convs:
            f.write(json.dumps({"id": c["id"], "category": c["category"], "messages": c["messages"],
                                "author": "user", "source_file": a.src.name}, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
