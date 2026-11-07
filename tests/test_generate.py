"""고정 프롬프트 파일: 기본 목록 유지, 파일 형식 검증"""

import json
from pathlib import Path

import pytest

from generate import PROMPTS, load_prompts

ROOT = Path(__file__).resolve().parent.parent


def test_default_prompts_unchanged():
    texts, spec = load_prompts(None)
    assert texts == PROMPTS and spec is None
    assert len(PROMPTS) == 9  # Phase 1 samples_init.md·samples.md와 같은 목록


def test_ko20_file():
    texts, spec = load_prompts(ROOT / "eval" / "prompts_ko20.json")
    assert len(texts) == 20
    assert len({p["id"] for p in spec["prompts"]}) == 20
    assert texts[:6] == PROMPTS[:6]  # 기존 한국어 6개를 그대로 포함해 결과를 대조할 수 있게
    assert all(any("가" <= ch <= "힣" for ch in t) for t in texts)


@pytest.mark.parametrize("prompts", [[], [{"text": "가"}, {"text": "가"}]])
def test_rejects_empty_or_duplicate(tmp_path, prompts):
    f = tmp_path / "p.json"
    f.write_text(json.dumps({"prompts": prompts}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_prompts(f)
