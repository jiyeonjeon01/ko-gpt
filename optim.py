"""옵티마이저, 학습률 스케줄(WSD), 한 번의 가중치 업데이트 (Gradient accumulation, BF16 autocast)"""

import math
from dataclasses import dataclass

import torch

from model import GPT


@dataclass
class OptimConfig:
    max_lr: float = 6e-4
    min_lr_ratio: float = 0.0  # 감쇠 끝의 lr = max_lr × min_lr_ratio
    warmup_steps: int = 100
    decay_steps: int = 0  # 마지막 decay_steps 동안 선형 감쇠 (0이면 감쇠 없음)
    max_steps: int = 1000
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0


def wsd_lr(step: int, cfg: OptimConfig) -> float:
    """Warmup-Stable-Decay. step은 이번에 수행할 업데이트의 0부터 센 번호다.

    warmup: (step + 1) / warmup_steps × max_lr  → stable: max_lr → decay: max_lr에서 min_lr까지 선형
    """
    if cfg.warmup_steps > 0 and step < cfg.warmup_steps:
        return cfg.max_lr * (step + 1) / cfg.warmup_steps
    decay_start = cfg.max_steps - cfg.decay_steps
    if cfg.decay_steps > 0 and step >= decay_start:
        progress = (step - decay_start + 1) / cfg.decay_steps
        return cfg.max_lr * (1.0 - (1.0 - cfg.min_lr_ratio) * min(progress, 1.0))
    return cfg.max_lr


def build_optimizer(model: GPT, cfg: OptimConfig) -> torch.optim.AdamW:
    """2차원 이상 파라미터(Linear·Embedding)에만 Weight Decay. RMSNorm 가중치(1차원)는 감쇠하지 않는다.

    공유 가중치(임베딩 = LM Head)는 parameters()가 한 번만 돌려준다.
    """
    params = list(model.parameters())
    decay = [p for p in params if p.dim() >= 2]
    no_decay = [p for p in params if p.dim() < 2]
    fused = all(p.is_cuda for p in params)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=cfg.max_lr, betas=(cfg.beta1, cfg.beta2), fused=fused,
    )


class NonFiniteError(FloatingPointError):
    """Loss 또는 Gradient Norm이 NaN·Inf. 학습을 멈추고 정상 Checkpoint에서 재개를 시도해야 한다."""


def train_step(model: GPT, optimizer: torch.optim.Optimizer, micro_batches, lr: float, grad_clip: float,
               device: str, autocast: bool = True) -> tuple[float, float]:
    """micro_batches: [(x, y), ...] (CPU 텐서). Gradient를 누적한 뒤 한 번 업데이트한다.

    각 micro-batch Loss를 개수로 나눠 더한다. 사전학습은 모든 위치에 Loss가 있고 길이가 같으므로
    이것이 전체 토큰 평균과 같다. (SFT처럼 마스크 수가 다르면 호출 쪽에서 토큰 가중을 다뤄야 한다)

    반환: 평균 Loss, Clipping 전 Gradient Norm. NaN·Inf면 NonFiniteError (업데이트하지 않음)
    """
    for group in optimizer.param_groups:
        group["lr"] = lr
    optimizer.zero_grad(set_to_none=True)
    n = len(micro_batches)
    total = 0.0
    for x, y in micro_batches:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda" if device.startswith("cuda") else "cpu",
                            dtype=torch.bfloat16, enabled=autocast):
            _, loss = model(x, y)
        (loss / n).backward()
        total += loss.detach().float()
    loss_value = (total / n).item()
    if not math.isfinite(loss_value):
        optimizer.zero_grad(set_to_none=True)
        raise NonFiniteError(f"loss가 {loss_value}")
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip).item()
    if not math.isfinite(grad_norm):
        optimizer.zero_grad(set_to_none=True)
        raise NonFiniteError(f"gradient norm이 {grad_norm}")
    optimizer.step()
    return loss_value, grad_norm
