"""ask.py: 답변 자르기, 자원 부족 시 모델을 읽지 않음"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import ask


def test_cut_answer_stops_at_invented_next_question():
    assert ask.cut_answer(" 서울입니다.\n질문: 부산은?\n답변: 항구") == " 서울입니다."
    assert ask.cut_answer(" 서울입니다.") == " 서울입니다."
    assert ask.cut_answer(" 6개월입니다.\n답변: 12개월") == " 6개월입니다."
    assert ask.cut_answer("질문: 처음부터") == "질문: 처음부터"  # 맨 앞이면 자르지 않는다


def test_prompt_format():
    assert ask.PROMPT.format(q="1년은 몇 개월인가요?") == "질문: 1년은 몇 개월인가요?\n답변:"


def test_refuses_without_loading_when_resources_short(monkeypatch):
    monkeypatch.setattr(ask.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(ask, "windows_available_mb", lambda: 1500)
    monkeypatch.setattr(ask, "gpu", lambda: {"gpu_total_mb": 12288, "gpu_used_mb": 1000})
    monkeypatch.setattr(ask, "other_training_pids", lambda: [])

    def no_load(*a, **k):
        raise AssertionError("자원이 모자라면 모델을 읽지 않아야 한다")
    monkeypatch.setattr(ask, "load_model", no_load)
    assert ask.main([]) == ask.EXIT_RESOURCES


def test_refuses_when_training_running(monkeypatch):
    monkeypatch.setattr(ask.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(ask, "windows_available_mb", lambda: 8000)
    monkeypatch.setattr(ask, "gpu", lambda: {"gpu_total_mb": 12288, "gpu_used_mb": 1000})
    monkeypatch.setattr(ask, "other_training_pids", lambda: [4242])
    monkeypatch.setattr(ask, "load_model", lambda *a, **k: (_ for _ in ()).throw(AssertionError("읽으면 안 됨")))
    assert ask.main([]) == ask.EXIT_RESOURCES
