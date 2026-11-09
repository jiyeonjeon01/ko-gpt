"""대화 평가 세트: 초안 형식, 중복 검사 기준, 고정 절차"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import eval_overlap as eo
import freeze_eval_set as fz

DRAFT = ROOT / "eval" / "chat120" / "draft_v0.1.json"


def test_draft_is_valid():
    doc = json.loads(DRAFT.read_text(encoding="utf-8"))
    assert fz.validate(doc) == []
    assert doc["status"] == "draft"


def test_validate_catches_errors():
    doc = json.loads(DRAFT.read_text(encoding="utf-8"))
    bad = {**doc, "items": doc["items"][:-1] + [{**doc["items"][0]}]}  # 영어 1개 빼고 c01 중복
    errs = fz.validate(bad)
    assert any("영어" in e for e in errs) and any("id 중복" in e for e in errs)
    it = {**doc["items"][0], "messages": [{"role": "assistant", "content": "x"}]}
    assert any("역할 순서" in e for e in fz.validate({"items": [it]}))


def test_freeze_once_and_verify(tmp_path):
    draft = tmp_path / "draft.json"
    draft.write_text(DRAFT.read_text(encoding="utf-8"), encoding="utf-8")
    rubric = tmp_path / "rubric.md"
    rubric.write_text("기준", encoding="utf-8")
    out = fz.freeze(draft, 1, rubric)
    assert fz.verify(out)
    with pytest.raises(FileExistsError):
        fz.freeze(draft, 1, rubric)
    d = json.loads(out.read_text(encoding="utf-8"))
    d["items"][0]["reference"] = {"kind": "none"}  # 고정 뒤 수정 → 검증 실패
    out.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    assert not fz.verify(out)


def test_near_duplicate_rule():
    assert eo.is_near_duplicate("What is your name?", "what is your name")
    assert eo.is_near_duplicate("대한민국의 수도는?", "대한민국의 수도는 어디인가요")
    assert eo.is_near_duplicate("잠이 안 올 때 뭐 하면 좋아?", "요즘 잠이 안 올 때 뭐 하면 좋아요?")
    assert not eo.is_near_duplicate("대한민국의 수도는?", "조선 시대 수도였던 한양의 역사와 궁궐, 4대문, 그리고 도성 안 사람들의 생활에 대해 자세히 설명해 주세요")
    assert not eo.is_near_duplicate("7 곱하기 8은?", "광합성은 식물이 빛을 이용해 양분을 만드는 과정이다")


def test_user_text_joins_turns():
    it = {"messages": [{"role": "user", "content": "A"}, {"role": "assistant", "content": "B"},
                       {"role": "user", "content": "C"}]}
    assert eo.user_text(it) == "A\nC"


def test_guard_checks_paragraphs_of_long_messages():
    """dolly처럼 '질문\n\n긴 참고 글'이면 발화 전체로는 길이 비율에 걸려 놓치므로 문단별로도 본다"""
    g = eo.EvalGuard([("f06", "태양계에서 가장 큰 행성은?")])
    ctx = "목성은 태양에서 다섯 번째 행성이다. " * 20
    assert g.hit([{"role": "user", "content": "태양계에서 가장 큰 행성은 무엇인가요?\n\n" + ctx}]) == "f06"
    assert g.hit([{"role": "user", "content": "오늘 날씨 어때?\n\n" + ctx}]) is None
    assert eo.segments("한 문단") == ["한 문단"]


def test_draft_v02_valid_and_regex_compiles():
    doc = json.loads((ROOT / "eval/chat120/draft_v0.2.json").read_text(encoding="utf-8"))
    assert fz.validate(doc) == []
    assert doc["status"] == "draft"
