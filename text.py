"""텍스트 전처리와 특수 토큰 정의

preprocess()는 데이터 준비, Tokenizer 학습, 추론 입력에 똑같이 쓰는 유일한 정규화 함수다 (설계안 v0.2 6-2절).
Tokenizer 자체에는 정규화를 두지 않으므로 decode(encode(x)) == preprocess(x)가 성립해야 한다.
"""

import re
import unicodedata

# 특수 토큰: ID 0 ~ 15 고정. 이 16개 외에는 특수 토큰을 만들지 않는다
PAD, EOT, IM_START, IM_END = "<|pad|>", "<|endoftext|>", "<|im_start|>", "<|im_end|>"
NUM_RESERVED = 12
SPECIAL_TOKENS = [PAD, EOT, IM_START, IM_END] + [f"<|reserved_{i}|>" for i in range(NUM_RESERVED)]
PAD_ID, EOT_ID, IM_START_ID, IM_END_ID = 0, 1, 2, 3

# 탭·줄바꿈을 뺀 제어 문자 (유니코드 범주 Cc). 서식 문자(Cf, 예: 이모지 결합용 ZWJ)는 남긴다
_CONTROL = "".join(chr(c) for c in range(0x110000)
                   if unicodedata.category(chr(c)) == "Cc" and chr(c) not in "\t\n")
_CONTROL_RE = re.compile(f"[{re.escape(_CONTROL)}]")
_TRAILING_WS_RE = re.compile(r"[ \t]+$", re.MULTILINE)
_MANY_NEWLINES_RE = re.compile(r"\n{4,}")  # 빈 줄 3개 이상 → 빈 줄 2개


def preprocess(text: str) -> str:
    """NFC → 줄바꿈 통일 → 제어 문자 제거 → 줄 끝 공백 제거 → 연속 빈 줄 축소 → 특수 토큰 문자열 무력화

    멱등이다: preprocess(preprocess(x)) == preprocess(x)
    """
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL_RE.sub("", text)
    text = _TRAILING_WS_RE.sub("", text)
    text = _MANY_NEWLINES_RE.sub("\n\n\n", text)
    # 원문의 "<|endoftext|>" 같은 문자열이 특수 토큰으로 인식되지 않게 한다 (결과에는 "<|"가 남지 않는다)
    return text.replace("<|", "< |")
