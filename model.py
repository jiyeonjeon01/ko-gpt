"""Decoder-only GPT: RMSNorm, RoPE, SDPA Causal Attention, SwiGLU, 임베딩·LM Head 가중치 공유, bias 없음

이전 MiniGPT(mini-gpt-from-scratch/model.py)의 Pre-LN 구조와 입력 검사를 유지하고 구성 요소를 바꿨다.
    LayerNorm → RMSNorm, 학습형 위치 임베딩 → RoPE, 직접 계산한 Attention → SDPA,
    GELU FFN → SwiGLU, LM Head 독립 → 임베딩과 공유
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import GPTConfig

IGNORE_INDEX = -100  # 이 값의 정답 위치는 Loss에서 빠진다 (SFT의 질문 토큰 등)


def rope_tables(head_dim: int, length: int, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    """위치 0 ~ length-1의 cos, sin 표: (length, head_dim // 2), float32"""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    angles = torch.outer(torch.arange(length, dtype=torch.float32), inv_freq)
    return angles.cos(), angles.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (B, H, T, D). 앞 절반과 뒤 절반을 한 쌍으로 회전한다 (rotate-half 방식).

    cos, sin: (T, D // 2). 계산은 float32로 하고 x의 dtype으로 돌려준다.
    """
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half].float(), x[..., half:].float()
    out = torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
    return out.to(x.dtype)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.n_embd = config.n_embd
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(self.n_embd, dim=2)
        # (B, T, C) → (B, H, T, D)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        # Flash 계열 커널: (B, H, T, T) 점수 행렬을 메모리에 만들지 않는다
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class SwiGLU(nn.Module):
    """FFN(x) = W2 · (SiLU(W1 x) ⊙ W3 x)"""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.gate = nn.Linear(config.n_embd, config.ffn_dim, bias=False)
        self.up = nn.Linear(config.n_embd, config.ffn_dim, bias=False)
        self.down = nn.Linear(config.ffn_dim, config.n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class TransformerBlock(nn.Module):
    """Pre-Norm Block: H = X + Attn(Norm(X)), Y = H + FFN(Norm(H))"""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.norm_1 = nn.RMSNorm(config.n_embd, eps=config.norm_eps)
        self.attn = CausalSelfAttention(config)
        self.norm_2 = nn.RMSNorm(config.n_embd, eps=config.norm_eps)
        self.ffn = SwiGLU(config)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm_1(x), cos, sin)
        x = x + self.ffn(self.norm_2(x))
        return x


class GPT(nn.Module):
    """토큰 ID (B, T)를 받아 각 위치의 다음 토큰 Logits (B, T, vocab_size)를 출력한다."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.blocks = nn.ModuleList(TransformerBlock(config) for _ in range(config.n_layer))
        self.norm_f = nn.RMSNorm(config.n_embd, eps=config.norm_eps)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # 가중치 공유: 같은 Parameter 객체

        cos, sin = rope_tables(config.head_dim, config.block_size, config.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # 잔차 경로에 더해지는 출력 Projection은 층 수에 맞춰 줄인다 (GPT-2 방식)
        residual_std = 0.02 / math.sqrt(2 * config.n_layer)
        for block in self.blocks:
            nn.init.normal_(block.attn.proj.weight, mean=0.0, std=residual_std)
            nn.init.normal_(block.ffn.down.weight, mean=0.0, std=residual_std)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.RMSNorm):
            nn.init.ones_(module.weight)

    def num_parameters(self) -> int:
        """공유 가중치는 한 번만 센다 (parameters()가 중복을 제거한다)"""
        return sum(p.numel() for p in self.parameters())

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        idx:     (B, T) 입력 토큰 ID (torch.long)
        targets: (B, T) 정답 토큰 ID (선택). IGNORE_INDEX 위치는 Loss에서 빠진다

        반환: logits (B, T, vocab_size), loss (Loss가 걸린 정답 토큰의 평균 CE, targets가 없으면 None)
        """
        if idx.ndim != 2:
            raise ValueError(f"idx는 (B, T) 형태여야 한다: {tuple(idx.shape)}")
        if idx.dtype != torch.long:
            raise ValueError(f"idx의 dtype은 torch.long이어야 한다: {idx.dtype}")
        B, T = idx.shape
        if not 1 <= T <= self.config.block_size:
            raise ValueError(f"입력 길이 T={T}는 1 ~ block_size({self.config.block_size})여야 한다")
        if targets is not None and (targets.shape != idx.shape or targets.dtype != torch.long):
            raise ValueError(
                f"targets는 idx와 같은 Shape의 torch.long이어야 한다: {tuple(targets.shape)} {targets.dtype}"
            )

        cos, sin = self.rope_cos[:T], self.rope_sin[:T]
        x = self.tok_emb(idx)
        for block in self.blocks:
            x = block(x, cos, sin)
        logits = self.lm_head(self.norm_f(x))

        if targets is None:
            return logits, None
        if not (targets != IGNORE_INDEX).any():
            raise ValueError("Loss가 걸린 정답 토큰이 하나도 없다 (모든 targets가 IGNORE_INDEX)")
        # BF16 autocast 중에도 CE는 float32로 계산한다
        loss = F.cross_entropy(
            logits.view(B * T, -1).float(), targets.view(B * T), ignore_index=IGNORE_INDEX
        )
        return logits, loss
