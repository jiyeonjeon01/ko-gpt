"""출처별 토큰 샤드에서 학습 배치를 뽑는 혼합 샘플러 (설계안 v0.2 4-1, 7-3절)

- 샤드는 uint16 .bin 파일이다. 창 하나(block_size + 1 토큰)만 os.pread로 읽고, 배치마다 읽은 파일의
  페이지 캐시를 버린다 (posix_fadvise DONTNEED).
  memmap으로 읽으면 WSL 파일 캐시가 2분 만에 7.9GB까지 불어 Windows 사용 가능 RAM이 0.2GB로 떨어졌다
  (Phase 1 사전 점검 실측). WSL에서는 게스트의 파일 캐시도 호스트 메모리를 차지한다.
- 시퀀스마다: 출처를 가중치로 고름 → 샤드를 토큰 수에 비례해 고름 → 무작위 시작 위치에서 block_size + 1 토큰을 읽음.
  이전 get_batch의 복원추출 방식을 확장한 것이라 재개 시 RNG 상태만 복원하면 같은 배치가 나온다.
- 출처마다 노출 상한(고유 토큰 × max_repeat)을 둔다. 상한에 도달한 출처는 제외하고 그 사실을 기록한다.
- 모든 출처가 상한에 도달하면 DataExhausted를 낸다. 조용히 다시 뽑지 않는다.
"""

import os
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
    fds: list[int]
    lengths: list[int]  # 샤드별 토큰 수
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
            fds = [os.open(p, os.O_RDONLY) for p in src.shards]
            lengths = [os.fstat(fd).st_size // 2 for fd in fds]
            usable = [max(n - block_size, 0) for n in lengths]  # y의 끝이 샤드 끝을 넘지 않는 시작 위치 수
            if sum(usable) == 0:
                raise ValueError(f"{src.name}: block_size({block_size})보다 긴 샤드가 없다")
            unique = sum(lengths)
            self.opened.append(_Opened(src, fds, lengths, torch.tensor(usable, dtype=torch.float64),
                                       unique, int(unique * src.max_repeat)))
        self.state = SamplerState(seen_tokens={o.source.name: 0 for o in self.opened})
        self._touched: set[int] = set()

    def _active(self) -> list[_Opened]:
        return [o for o in self.opened if o.source.name not in self.state.exhausted]

    def _read(self, o: _Opened, gen: torch.Generator) -> torch.Tensor:
        shard = int(torch.multinomial(o.shard_tokens, 1, generator=gen))
        start = int(torch.randint(int(o.shard_tokens[shard]), (1,), generator=gen))
        n = self.block_size + 1
        raw = os.pread(o.fds[shard], 2 * n, 2 * start)
        if len(raw) != 2 * n:
            raise OSError(f"{o.source.name} 샤드 {shard}: {start}에서 {n}토큰을 읽지 못했다 ({len(raw)}바이트)")
        self._touched.add(o.fds[shard])
        return torch.from_numpy(np.frombuffer(raw, dtype=np.uint16).astype(np.int64))

    def close(self) -> None:
        for o in self.opened:
            for fd in o.fds:
                os.close(fd)
            o.fds.clear()

    def __del__(self):
        try:
            self.close()
        except OSError:
            pass

    def drop_cache(self) -> None:
        """이번 배치에서 읽은 샤드의 페이지 캐시를 버린다 (데이터는 바뀌지 않는다)"""
        for fd in self._touched:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        self._touched.clear()

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
        self.drop_cache()
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
