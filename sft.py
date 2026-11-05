"""대화 → 토큰 ID와 Loss 마스크 (설계안 v0.2 4-3절 템플릿)

    <|endoftext|><|im_start|>system\n{내용}<|im_end|>\n<|im_start|>user\n{내용}<|im_end|>\n
    <|im_start|>assistant\n{내용}<|im_end|>\n

특수 토큰은 문자열로 인코딩하지 않고 ID를 직접 넣는다. 내용은 preprocess()를 거쳐 인코딩하므로
사용자 입력의 "<|im_end|>" 같은 문자열이 특수 토큰이 되지 않는다.

Loss 마스크는 "그 토큰을 정답으로 예측할 때 Loss를 주는가"이다.
assistant 응답의 내용 토큰과 그 뒤의 <|im_end|>만 1이고, 나머지(system·user 전체, 역할 머리말, 턴 사이 줄바꿈)는 0이다.
"""

from collections.abc import Callable

import torch

from model import IGNORE_INDEX
from text import EOT_ID, IM_END_ID, IM_START_ID, preprocess

ROLES = ("system", "user", "assistant")


def render_conversation(
    messages: list[dict], encode: Callable[[str], list[int]]
) -> tuple[list[int], list[int]]:
    """messages: [{"role": ..., "content": ...}, ...] → (토큰 ID, 토큰별 Loss 마스크)"""
    if not messages or messages[-1]["role"] != "assistant":
        raise ValueError("대화는 assistant 응답으로 끝나야 한다")
    ids, mask = [EOT_ID], [0]
    for message in messages:
        role = message["role"]
        if role not in ROLES:
            raise ValueError(f"알 수 없는 역할: {role!r}")
        header = [IM_START_ID] + encode(role + "\n")
        content = encode(preprocess(message["content"]))
        if not content and role == "assistant":
            raise ValueError("빈 assistant 응답")
        trained = int(role == "assistant")
        ids += header + content + [IM_END_ID]
        mask += [0] * len(header) + [trained] * (len(content) + 1)
        separator = encode("\n")
        ids += separator
        mask += [0] * len(separator)
    return ids, mask


def prompt_ids(messages: list[dict], encode: Callable[[str], list[int]]) -> list[int]:
    """생성용 입력: 주어진 대화 뒤에 assistant 머리말까지 붙인다 (학습 때와 같은 토큰 경계)"""
    ids = [EOT_ID]
    for message in messages:
        ids += [IM_START_ID] + encode(message["role"] + "\n") + encode(preprocess(message["content"]))
        ids += [IM_END_ID] + encode("\n")
    return ids + [IM_START_ID] + encode("assistant\n")


def to_example(ids: list[int], mask: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    """Causal LM 입력·정답 쌍. x[t]를 보고 y[t] = ids[t+1]을 예측하므로 마스크도 한 칸 민다.

    y[t]는 mask[t+1]이 0이면 IGNORE_INDEX다.
    """
    if len(ids) != len(mask) or len(ids) < 2:
        raise ValueError("ids와 mask는 길이가 같고 2 이상이어야 한다")
    x = torch.tensor(ids[:-1], dtype=torch.long)
    y = torch.tensor(ids[1:], dtype=torch.long)
    y[torch.tensor(mask[1:], dtype=torch.bool).logical_not()] = IGNORE_INDEX
    return x, y
