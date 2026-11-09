"""평가 세트를 사람이 검토하기 쉬운 Markdown 표로 만든다 (문항·정답 기준·자동 판정 방식)

    python3 scripts/eval_set_table.py eval/chat120/draft_v0.2.json --out eval/chat120/draft_v0.2_table.md
"""

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

FOCUS = ["f05", "f09", "m05", "f13", "k09", "k02", "k04"]
# 판정 예 (tests/test_chat_eval.py와 같은 예, chat_metrics.py로 실제 판정해 표에 넣는다)
EXAMPLES = {
    "f05": ["100℃에서 끓어요", "100°C", "섭씨 100도", "212°F", "90도", "기압에 따라 달라요"],
    "f09": ["한라산이에요", "백두산이에요", "남한에서는 한라산, 한반도에서는 백두산"],
    "m05": ["오후 8시예요", "저녁 8시입니다", "20:00", "8시예요", "오전 8시", "오후 18시"],
    "f13": ["물이 돼요.", "물.", "물로 변해요", "동물이 됩니다", "물질이 돼요", "물체예요"],
    "k02": ["len(my_list)을 써요", "len 함수를 써요", "리스트.length", "The length is 3"],
    "k04": ["sorted(lst)", "lst.sort()", "a.sort(reverse=False)", "sort 함수"],
    "k09": ["if key in d:", "`key in my_dict`로 확인", "d.keys()", "d.get('a')", "in 연산자를 써요", "print(string)", "파이썬에서는 in Python."],
}


def cell(s: str) -> str:
    return str(s).replace("|", "\\|").replace("\n", "<br>")


def code(s: str) -> str:
    s = str(s).replace("|", "\\|")
    return f"`` {s} ``" if "`" in s else f"`{s}`"


def convo(item: dict) -> str:
    label = {"user": "U", "assistant": "A"}
    return "<br>".join(f"**{label[m['role']]}:** {cell(m['content'])}" for m in item["messages"])


def criteria(item: dict) -> str:
    ref = item["reference"]
    parts = []
    if ref["kind"] == "exact":
        parts.append("허용 답: " + ", ".join(code(a) for a in ref["accept"]))
        if ref.get("accept_regex"):
            parts.append("자동 판정 정규식: " + " / ".join(code(r) for r in ref["accept_regex"]))
    elif ref["kind"] == "key_points":
        parts.append("핵심: " + "; ".join(cell(p) for p in ref["points"]))
    else:
        parts.append("정답 기준 없음 (틀린 사실이 없으면 사실 정확성 2점)")
    if ref.get("note"):
        parts.append("note: " + cell(ref["note"]))
    if item.get("safety"):
        parts.append("**안전:** " + cell(item["safety"]))
    return "<br>".join(parts)


def auto_rule(item: dict) -> str:
    ref = item["reference"]
    if ref["kind"] != "exact":
        return "정답 포함률 대상 아님"
    if ref.get("accept_regex"):
        return "정규식 (공백 유지, 소문자)"
    if any(a and (a[0].isdigit() or a[-1].isdigit()) for a in ref["accept"]):
        return "부분 문자열 (공백 무시, 소문자) + 숫자 경계"
    return "부분 문자열 (공백 무시, 소문자)"


def judge(item: dict, text: str) -> bool:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import chat_metrics as cm
    ref = item["reference"]
    return cm.matches_regex(text, ref["accept_regex"]) if ref.get("accept_regex") else cm.contains_answer(text, ref["accept"])


def render(doc: dict, sha: str, src: str) -> str:
    items = doc["items"]
    multi = [it for it in items if len(it["messages"]) > 1]
    cats = Counter(it["category"] for it in items)
    kinds = Counter(it["reference"]["kind"] for it in items)
    out = [f"# chat120 {doc['version']} 검토표", "",
           f"- 원본: `{src}` (SHA-256 `{sha}`)",
           f"- 상태: **{doc['status']}** (고정 전). 이 표는 원본에서 자동으로 만든 것이다. 고칠 때는 원본 JSON을 고친다.",
           f"- 문항 {len(items)}개: " + ", ".join(f"{c} {n}" for c, n in cats.items()),
           f"- 정답 기준 종류: exact {kinds['exact']}, key_points {kinds['key_points']}, none {kinds['none']} / 다중 턴 {len(multi)}개",
           "- 표 안의 `\\|`는 Markdown 표 때문에 넣은 표시다. 정규식에서는 `|`(또는)이다.",
           "- 채점 기준 전체는 `rubric.md`. 요약: 사람 채점 5개 축(형식, 자연스러움, 관련성, 사실 정확성, 안전성) × 0~2점.",
           "  - 사실 정확성은 exact = 허용 답과 같은 답, key_points = 핵심과 맞음, none = 틀린 사실 주장 없음으로 판단한다.",
           "  - 자동 정답 포함률은 exact 문항에만 계산하는 보조 지표다.", ""]

    out += ["## 1. 집중 검토 문항: f05·f09·m05·f13·k09 (+ v0.3에서 고친 k02·k04)", "",
            "| id | 질문 | 허용 답 (사람 채점) | 자동 판정 | note |", "|---|---|---|---|---|"]
    by_id = {it["id"]: it for it in items}
    for i in FOCUS:
        it, ref = by_id[i], by_id[i]["reference"]
        rule = ("정규식: " + " / ".join(code(r) for r in ref["accept_regex"])) if ref.get("accept_regex") \
            else auto_rule(it) + ": " + ", ".join(code(a) for a in ref["accept"])
        out.append(f"| {i} | {cell(it['messages'][-1]['content'])} | {', '.join(code(a) for a in ref['accept'])} | {rule} | {cell(ref.get('note', ''))} |")
    out += ["", "**자동 판정 예** (`chat_metrics.py`로 실제 판정한 결과):", "",
            "| id | 정답으로 셈 | 정답으로 세지 않음 |", "|---|---|---|"]
    for i in FOCUS:
        ok, no = [], []
        for t in EXAMPLES[i]:
            (ok if judge(by_id[i], t) else no).append(code(t))
        out.append(f"| {i} | {', '.join(ok)} | {', '.join(no)} |")
    out += ["", "- 부정 표현(\"백두산이 아니라 한라산\", \"물이 아니라 수증기\")은 자동 지표가 구별하지 못한다. 최종 판정은 사람 채점이다.", ""]

    out += ["## 2. 다중 턴 7개: 전체 대화와 정답 기준", ""]
    for it in multi:
        out += [f"### {it['id']} · {it['category']}", "", "| 순서 | 역할 | 발화 |", "|---|---|---|"]
        for k, m in enumerate(it["messages"], 1):
            out.append(f"| {k} | {'사용자' if m['role'] == 'user' else 'assistant (평가 입력)'} | {cell(m['content'])} |")
        out += ["", f"- 기대 언어: {it['expected_language']}", f"- 기준: {criteria(it)}", ""]

    out += ["## 3. 전체 120문항", ""]
    for cat in cats:
        out += [f"### {cat}", "", "| id | 대화 | 언어 | 정답 기준 | 자동 정답 포함 판정 |", "|---|---|---|---|---|"]
        for it in items:
            if it["category"] == cat:
                out.append(f"| {it['id']} | {convo(it)} | {it['expected_language']} | {criteria(it)} | {auto_rule(it)} |")
        out.append("")
    return "\n".join(out) + "\n"


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("eval_set", type=Path)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    raw = a.eval_set.read_bytes()
    a.out.write_text(render(json.loads(raw), hashlib.sha256(raw).hexdigest(), str(a.eval_set)), encoding="utf-8")
    print(a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
