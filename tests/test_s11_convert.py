"""S11 변환: 형식 해석, 검사, 평가 문항 유사도 차단"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import s11_convert as sc

GOOD = """### s11-0101 | 인사
U: 오랜만이야!
A: 오랜만이에요! 그동안 잘 지내셨어요?

### s11-0102 | 감정
U: 오늘 칭찬받았어.
A: 정말 좋으셨겠어요! 어떤 일로 칭찬받으셨어요?
U: 보고서를 잘 썼대.
그래서 기분이 좋아.
A: 노력한 만큼 인정받으셨네요. 축하드려요.
"""


def test_parse_and_check_good():
    convs, errors = sc.parse(GOOD)
    assert errors == []
    assert [c["id"] for c in convs] == ["s11-0101", "s11-0102"]
    assert convs[1]["messages"][2]["content"] == "보고서를 잘 썼대.\n그래서 기분이 좋아."
    e, w = sc.check(convs, sc.load_eval_prompts(ROOT))
    assert e == [] and w == []


def test_errors_and_warnings():
    bad = """### s11-0201 | 잡담
U: 안녕
U: 또 안녕
### s11-0201 | 인사
U: 머해
A: 몰라 😀 {NAME}
"""
    convs, errors = sc.parse(bad)
    e, w = sc.check(convs, [])
    assert any("범주" in x for x in e) and any("번갈아" in x for x in e) and any("id 중복" in x for x in e)
    assert any("이모지" in x for x in w) and any("자리표시" in x for x in w) and any("해요체" in x for x in w)


def test_blocks_eval_prompt_and_guide_example():
    text = """### s11-0001 | 인사
U: 오랜만이야!
A: 오랜만이에요! 그동안 잘 지내셨어요?

### s11-0301 | 기타
U: 잠이 안 올 때 뭐 하면 좋아?
A: 따뜻한 물을 마셔 보세요.
"""
    convs, _ = sc.parse(text)
    e, w = sc.check(convs, sc.load_eval_prompts(ROOT))
    assert any("s11-0301" in x and "평가 문항" in x for x in e)
    assert any("형식 예시" in x for x in w)


def test_main_writes_jsonl_and_refuses_overwrite(tmp_path):
    src = tmp_path / "b.md"
    src.write_text(GOOD, encoding="utf-8")
    out = tmp_path / "b.jsonl"
    assert sc.main([str(src), "--out", str(out)]) == 0
    rows = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["author"] == "user" and rows[1]["messages"][-1]["role"] == "assistant"
    assert sc.main([str(src), "--out", str(out)]) == 1
