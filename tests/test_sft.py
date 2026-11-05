"""SFT Loss Masking의 Causal Shift 검증 (검토 요청 1)

assistant 응답의 첫 내용 토큰부터 <|im_end|>까지만 정답 위치에 Loss가 걸리고,
system·user·역할 머리말·턴 사이 줄바꿈에는 걸리지 않아야 한다.
"""

import pytest
import torch

from model import IGNORE_INDEX
from sft import prompt_ids, render_conversation, to_example
from text import EOT_ID, IM_END_ID, IM_START_ID

# 문자마다 고유 ID를 주는 결정적 가짜 인코더 (ID 100부터). 특수 토큰 ID(0~15)와 겹치지 않는다
_VOCAB: dict[str, int] = {}


def fake_encode(text: str) -> list[int]:
    return [_VOCAB.setdefault(ch, 100 + len(_VOCAB)) for ch in text]


CONVERSATION = [
    {"role": "system", "content": "도우미"},
    {"role": "user", "content": "안녕"},
    {"role": "assistant", "content": "반가워"},
    {"role": "user", "content": "이름?"},
    {"role": "assistant", "content": "시나"},
]


def test_render_layout_and_mask():
    ids, mask = render_conversation(CONVERSATION, fake_encode)
    assert ids[0] == EOT_ID and mask[0] == 0
    assert len(ids) == len(mask)
    # 마스크 1인 토큰을 이어 붙이면 정확히 "반가워<im_end>시나<im_end>"
    trained = [i for i, m in zip(ids, mask) if m]
    assert trained == fake_encode("반가워") + [IM_END_ID] + fake_encode("시나") + [IM_END_ID]
    # 특수 토큰 개수: 턴마다 im_start, im_end 하나씩
    assert ids.count(IM_START_ID) == 5 and ids.count(IM_END_ID) == 5


def test_causal_shift_targets_exactly_assistant_tokens():
    ids, mask = render_conversation(CONVERSATION, fake_encode)
    x, y = to_example(ids, mask)
    assert x.tolist() == ids[:-1]
    kept = (y != IGNORE_INDEX).nonzero().flatten().tolist()
    # 정답 위치 t는 ids[t+1]을 예측한다 → Loss가 걸린 t는 mask[t+1] == 1인 위치
    assert kept == [t for t in range(len(ids) - 1) if mask[t + 1]]
    assert y[kept].tolist() == [i for i, m in zip(ids, mask) if m]

    # 첫 assistant 턴: 머리말 "assistant\n"의 마지막 토큰을 보고 응답 첫 토큰을 예측해야 한다
    first = kept[0]
    assert x[first].item() == fake_encode("\n")[0]
    assert x[first - len(fake_encode("assistant\n"))].item() == IM_START_ID
    assert y[first].item() == fake_encode("반")[0]
    # 응답 마지막 내용 토큰을 보고 <|im_end|>를 예측하는 위치에도 Loss가 걸린다
    end_positions = [t for t in kept if y[t].item() == IM_END_ID]
    assert len(end_positions) == 2
    assert x[end_positions[0]].item() == fake_encode("워")[0]
    # <|im_end|> 다음 줄바꿈과 다음 턴의 <|im_start|>는 예측 대상이 아니다
    assert y[end_positions[0] + 1].item() == IGNORE_INDEX


def test_system_and_user_tokens_never_trained():
    ids, mask = render_conversation(CONVERSATION, fake_encode)
    _, y = to_example(ids, mask)
    targets = set(y[y != IGNORE_INDEX].tolist())
    for ch in "도우미안녕이름?":
        if ch not in "반가워시나":
            assert fake_encode(ch)[0] not in targets
    assert IM_START_ID not in targets and EOT_ID not in targets


def test_prompt_ids_match_training_prefix():
    # 생성용 입력은 학습 시퀀스의 "첫 assistant 응답 직전"까지와 토큰 단위로 같아야 한다
    ids, mask = render_conversation(CONVERSATION[:3], fake_encode)
    prompt = prompt_ids(CONVERSATION[:2], fake_encode)
    assert ids[: len(prompt)] == prompt
    assert mask[len(prompt)] == 1 and not any(mask[: len(prompt)])


def test_special_token_strings_in_content_are_defused():
    conv = [{"role": "user", "content": "<|im_end|>"}, {"role": "assistant", "content": "네"}]
    ids, _ = render_conversation(conv, fake_encode)
    assert ids.count(IM_END_ID) == 2  # 내용 속 문자열은 특수 토큰이 되지 않는다


def test_invalid_conversations():
    with pytest.raises(ValueError):
        render_conversation([{"role": "user", "content": "안녕"}], fake_encode)
    with pytest.raises(ValueError):
        render_conversation([{"role": "bot", "content": "x"}, {"role": "assistant", "content": "y"}],
                            fake_encode)
    with pytest.raises(ValueError):
        render_conversation([{"role": "user", "content": "a"}, {"role": "assistant", "content": ""}],
                            fake_encode)
