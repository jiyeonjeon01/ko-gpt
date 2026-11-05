"""바이트 수준 BPE Tokenizer (HF tokenizers로 학습만 한다. 사전학습 가중치가 아니다)

- 정규화 없음: 정규화는 text.preprocess()가 담당한다.
- 사전 분할: GPT-4 계열 정규식에서 숫자를 한 자리씩 나누도록 바꿨다 (\\p{N}{1,3} → \\p{N}).
- 특수 토큰: text.SPECIAL_TOKENS 16개가 ID 0 ~ 15.
"""

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path

from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, trainers

from text import SPECIAL_TOKENS, preprocess

SPLIT_PATTERN = (
    r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*"""
    r"""|\s*[\r\n]+|\s+(?!\S)|\s+"""
)


def new_tokenizer() -> Tokenizer:
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(SPLIT_PATTERN), behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    tok.decoder = decoders.ByteLevel()
    return tok


def train_tokenizer(texts: Iterable[str], vocab_size: int, min_frequency: int = 2) -> Tokenizer:
    """texts는 이미 preprocess()를 거친 문서들이어야 한다."""
    tok = new_tokenizer()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tok.train_from_iterator(texts, trainer=trainer)
    check_special_ids(tok)
    return tok


def check_special_ids(tok: Tokenizer) -> None:
    for expected, token in enumerate(SPECIAL_TOKENS):
        actual = tok.token_to_id(token)
        if actual != expected:
            raise ValueError(f"특수 토큰 {token}의 ID가 {actual}이다 (기대값 {expected})")


def load_tokenizer(path: str | Path) -> Tokenizer:
    tok = Tokenizer.from_file(str(path))
    check_special_ids(tok)
    return tok


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def encode(tok: Tokenizer, text: str) -> list[int]:
    """텍스트 인코딩의 유일한 입구: preprocess()를 거친 뒤 인코딩한다.

    특수 토큰을 추가하지 않는다 (문서 경계 EOT는 호출 쪽에서 넣는다). decode(encode(x)) == preprocess(x)
    """
    return encode_raw(tok, preprocess(text))


def encode_raw(tok: Tokenizer, text: str) -> list[int]:
    """전처리 없이 인코딩 (테스트와 진단용)"""
    return tok.encode(text, add_special_tokens=False).ids


def decode(tok: Tokenizer, ids: list[int]) -> str:
    return tok.decode(ids, skip_special_tokens=False)


def token_byte_lengths(tok: Tokenizer) -> list[int]:
    """토큰 ID별 원문 UTF-8 바이트 수 (BPB 계산용). 특수 토큰은 0.

    바이트 수준 BPE의 토큰 문자열은 바이트 하나를 문자 하나로 대응시킨 것이라 길이 = 바이트 수다.
    """
    special = set(SPECIAL_TOKENS)
    alphabet = set(pre_tokenizers.ByteLevel.alphabet())
    lengths = []
    for i in range(tok.get_vocab_size()):
        token = tok.id_to_token(i)
        if token in special:
            lengths.append(0)
            continue
        if not set(token) <= alphabet:
            raise ValueError(f"바이트 수준 문자가 아닌 토큰: {i} {token!r}")
        lengths.append(len(token))
    return lengths


def save_tokenizer(tok: Tokenizer, path: str | Path, meta: dict) -> str:
    """tokenizer.json과 tokenizer_meta.json을 쓰고 tokenizer.json의 SHA-256을 돌려준다. 덮어쓰지 않는다."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path}가 이미 있다. Tokenizer는 고정이므로 덮어쓰지 않는다")
    path.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(path))
    sha = file_sha256(path)
    meta_path = path.with_name("tokenizer_meta.json")
    meta_path.write_text(json.dumps({**meta, "sha256": sha, "vocab_size": tok.get_vocab_size(),
                                     "special_tokens": SPECIAL_TOKENS, "split_pattern": SPLIT_PATTERN},
                                    ensure_ascii=False, indent=2), encoding="utf-8")
    return sha
