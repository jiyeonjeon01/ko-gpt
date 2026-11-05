"""출처별 토큰 샤드(memmap)에서 학습 배치를 뽑는 혼합 샘플러 (설계안 v0.2 4-1, 7-3절)

- 샤드는 uint16 .bin 파일이고 np.memmap으로 연다. RAM에는 OS 페이지 캐시만 남는다.
- 시퀀스마다: 출처를 가중치로 고름 → 샤드를 토큰 수에 비례해 고름 → 무작위 시작 위치에서 block_size + 1 토큰을 읽음.
  이전 get_batch의 복원추출 방식을 확장한 것이라 재개 시 RNG 상태만 복원하면 같은 배치가 나온다.
- 출처마다 노출 상한(고유 토큰 × max_repeat)을 둔다. 상한에 도달한 출처는 제외하고 그 사실을 기록한다.
- 모든 출처가 상한에 도달하면 DataExhausted를 낸다. 조용히 다시 뽑지 않는다.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch


class DataExhausted(RuntimeError):
    """모든 출처가 노출 상한에 도달했다. 학습을 멈추고 보고해야 한다."""


@dataclass
class Source:
    name: str
    shards: list[Path]
    weight: float
    max_repeat: float  # 고유 토큰의 몇 배까지 노출할지 (예: 1.0, 2.0)


@dataclass
class _Opened:
    source: Source
    arrays: list[np.ndarray]
    shard_tokens: torch.Tensor  # 샤드별 시작 가능한 위치 수 (float, 비례 추출용)
    unique_tokens: int
    cap_tokens: int


@dataclass
class SamplerState:
    seen_tokens: dict[str, int] = field(default_factory=dict)  # 출처별 누적 노출 (정답 토큰 수)
    exhausted: dict[str, int] = field(default_factory=dict)  # 출처 → 상한에 도달했을 때의 누적 시퀀스 수
    sequences: int = 0


class MixtureSampler:
    def __init__(self, sources: list[Source], block_size: int):
        if not sources:
            raise ValueError("출처가 하나 이상 필요하다")
        self.block_size = block_size
        self.opened: list[_Opened] = []
        for src in sources:
            if src.weight <= 0 or src.max_repeat <= 0:
                raise ValueError(f"{src.name}: weight와 max_repeat는 양수여야 한다")
            arrays = [np.memmap(p, dtype=np.uint16, mode="r") for p in src.shards]
            usable = [max(len(a) - block_size, 0) for a in arrays]  # y의 끝이 샤드 끝을 넘지 않는 시작 위치 수
            if sum(usable) == 0:
                raise ValueError(f"{src.name}: block_size({block_size})보다 긴 샤드가 없다")
            unique = sum(len(a) for a in arrays)
            self.opened.append(_Opened(src, arrays, torch.tensor(usable, dtype=torch.float64),
                                       unique, int(unique * src.max_repeat)))
        self.state = SamplerState(seen_tokens={o.source.name: 0 for o in self.opened})

    def _active(self) -> list[_Opened]:
        return [o for o in self.opened if o.source.name not in self.state.exhausted]

    def _read(self, o: _Opened, gen: torch.Generator) -> torch.Tensor:
        shard = int(torch.multinomial(o.shard_tokens, 1, generator=gen))
        start = int(torch.randint(int(o.shard_tokens[shard]), (1,), generator=gen))
        chunk = o.arrays[shard][start : start + self.block_size + 1]
        return torch.from_numpy(chunk.astype(np.int64))

    def sample(self, batch_size: int, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
        """반환: x, y (B, T) int64, 시퀀스별 출처 이름"""
        rows, names = [], []
        for _ in range(batch_size):
            active = self._active()
            if not active:
                raise DataExhausted(self.report())
            weights = torch.tensor([o.source.weight for o in active], dtype=torch.float64)
            o = active[int(torch.multinomial(weights, 1, generator=gen))]
            rows.append(self._read(o, gen))
            names.append(o.source.name)
            self.state.sequences += 1
            seen = self.state.seen_tokens[o.source.name] + self.block_size
            self.state.seen_tokens[o.source.name] = seen
            if seen >= o.cap_tokens:
                self.state.exhausted[o.source.name] = self.state.sequences
        chunk = torch.stack(rows)
        return chunk[:, :-1].contiguous(), chunk[:, 1:].contiguous(), names

    def report(self) -> str:
        lines = ["출처별 노출 (정답 토큰 / 고유 토큰 = 반복 배수, 상한)"]
        for o in self.opened:
            name = o.source.name
            seen = self.state.seen_tokens[name]
            status = f"상한 도달 (시퀀스 {self.state.exhausted[name]})" if name in self.state.exhausted else "활성"
            lines.append(f"  {name}: {seen:,} / {o.unique_tokens:,} = {seen / o.unique_tokens:.3f}배 "
                         f"(상한 {o.source.max_repeat}배) {status}")
        return "\n".join(lines)

    def state_dict(self) -> dict:
        return {"seen_tokens": dict(self.state.seen_tokens), "exhausted": dict(self.state.exhausted),
                "sequences": self.state.sequences}

    def load_state_dict(self, state: dict) -> None:
        if set(state["seen_tokens"]) != set(self.state.seen_tokens):
            raise ValueError(f"출처 구성이 다르다: {sorted(state['seen_tokens'])} ≠ {sorted(self.state.seen_tokens)}")
        self.state = SamplerState(dict(state["seen_tokens"]), dict(state["exhausted"]), state["sequences"])
