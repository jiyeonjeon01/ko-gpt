"""모델 설정과 크기 프리셋

설계안 v0.2 5-2절의 두 크기만 둔다 (사이 크기를 늘리지 않는다, 10-3절 규칙 2).
"""

import dataclasses
from dataclasses import dataclass


@dataclass
class GPTConfig:
    vocab_size: int = 32768
    block_size: int = 1024  # 문맥 길이 (RoPE 표의 길이)
    n_layer: int = 12
    n_embd: int = 768
    n_head: int = 12
    ffn_dim: int = 2048  # SwiGLU 은닉 폭 (약 8/3 × n_embd, 64의 배수)
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5

    def __post_init__(self):
        if self.n_embd % self.n_head:
            raise ValueError(f"n_embd({self.n_embd})는 n_head({self.n_head})로 나누어떨어져야 한다")
        if self.head_dim % 2:
            raise ValueError(f"RoPE는 짝수 head_dim이 필요하다: {self.head_dim}")

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head


MODEL_PRESETS = {
    "pilot-42m": {"n_layer": 8, "n_embd": 512, "n_head": 8, "ffn_dim": 1408},
    "main-110m": {"n_layer": 12, "n_embd": 768, "n_head": 12, "ffn_dim": 2048},
}


def model_config(preset: str, **overrides) -> GPTConfig:
    if preset not in MODEL_PRESETS:
        raise ValueError(f"지원하지 않는 모델: {preset!r} ({', '.join(MODEL_PRESETS)})")
    return GPTConfig(**{**MODEL_PRESETS[preset], **overrides})


def config_dict(cfg) -> dict:
    return dataclasses.asdict(cfg)
