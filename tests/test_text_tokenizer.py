"""전처리와 Tokenizer 왕복 검증 (설계안 v0.2 6-2절 ①~④)"""

import unicodedata

import pytest

from text import IM_END, IM_END_ID as IM_END_ID_RAW, SPECIAL_TOKENS, preprocess
from tokenizer import decode, encode, encode_raw, train_tokenizer

CASES = [
    "안녕하세요. 오늘 날씨가 좋네요!",
    unicodedata.normalize("NFD", "한글 자모 분해형 문장"),  # NFD → NFC로 바뀌어야 함
    "Hello, world! It's 2026-11-05.",
    "def add(a, b):\n\treturn a + b  \n",  # 줄 끝 공백
    "이모지 👩‍💻 결합 문자와 é",
    "제어\x00문자\x07와 \r\n윈도우 줄바꿈\r",
    "특수 문자열 <|endoftext|> 과 <|im_end|> 그리고 <<||",
    "빈 줄\n\n\n\n\n\n많음",
    "숫자 1234567 과 3.14159",
    "  앞 공백과 뒤 공백  ",
    "",
]

CORPUS = [preprocess(t) for t in CASES] * 3 + [
    "한국어와 English가 섞인 문장입니다. 학습용 짧은 말뭉치.",
    "print('hello')\nfor i in range(10):\n    print(i)\n",
]


@pytest.fixture(scope="module")
def tok():
    return train_tokenizer(CORPUS, vocab_size=400, min_frequency=1)


@pytest.mark.parametrize("text", CASES)
def test_preprocess_idempotent(text):
    once = preprocess(text)
    assert preprocess(once) == once  # ③


def test_preprocess_rules():
    assert preprocess(unicodedata.normalize("NFD", "한글")) == "한글"
    assert preprocess("a  \nb\t\n") == "a\nb\n"
    assert preprocess("a\r\nb\rc") == "a\nb\nc"
    assert preprocess("x\x00y\x1b") == "xy"
    assert preprocess("a\n\n\n\n\n\nb") == "a\n\n\nb"
    assert preprocess("\t탭은 유지") == "\t탭은 유지"
    assert "<|" not in preprocess("<|endoftext|> <<|| <|<|")
    assert preprocess("‍") == "‍"  # ZWJ(서식 문자)는 남긴다


@pytest.mark.parametrize("text", CASES)
def test_round_trip_equals_preprocess(tok, text):
    assert decode(tok, encode(tok, text)) == preprocess(text)  # ①


@pytest.mark.parametrize("text", CASES)
def test_round_trip_exact_on_preprocessed(tok, text):
    y = preprocess(text)
    assert decode(tok, encode(tok, y)) == y  # ②


def test_raw_text_round_trip_differs_only_by_preprocess(tok):
    raw = unicodedata.normalize("NFD", "한글")
    assert decode(tok, encode_raw(tok, raw)) == raw  # Tokenizer 자체는 바이트를 보존한다
    assert decode(tok, encode(tok, raw)) != raw  # 입구 encode()는 NFC로 바꾼다


def test_special_tokens_fixed_ids_and_not_produced_from_text(tok):
    for i, token in enumerate(SPECIAL_TOKENS):
        assert tok.token_to_id(token) == i
    ids = encode(tok, f"hi {IM_END} there")
    assert all(i >= len(SPECIAL_TOKENS) for i in ids)  # ④ 전처리 후에는 특수 토큰이 나오지 않는다
    assert IM_END_ID_RAW in encode_raw(tok, IM_END)  # 전처리를 건너뛰면 특수 토큰이 된다 (그래서 입구를 하나로 둔다)


def test_digits_split_individually(tok):
    pieces = tok.encode("2026", add_special_tokens=False).tokens
    assert pieces == ["2", "0", "2", "6"]
