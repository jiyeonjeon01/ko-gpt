"""혼합 샘플러: 출처별 반복 상한, 모든 출처 소진 시 중단 (검토 요청 2)"""

import numpy as np
import pytest
import torch

from sampler import DataExhausted, MixtureSampler, Source


def write_shard(path, values):
    np.asarray(values, dtype=np.uint16).tofile(path)
    return path


@pytest.fixture
def shards(tmp_path):
    a = write_shard(tmp_path / "a.bin", np.arange(0, 100))  # 고유 100 토큰
    b = write_shard(tmp_path / "b.bin", np.arange(1000, 1040))  # 고유 40 토큰
    return a, b


def test_windows_are_contiguous_and_shifted(shards):
    s = MixtureSampler([Source("a", [shards[0]], 1.0, 100.0)], block_size=8)
    x, y, names = s.sample(4, torch.Generator().manual_seed(0))
    assert x.dtype == torch.long and x.shape == (4, 8)
    assert torch.equal(x[:, 1:], y[:, :-1])
    assert torch.all(y - x == 1)  # arange 데이터: 연속 구간
    assert names == ["a"] * 4


def test_same_seed_same_batches(shards):
    def run():
        s = MixtureSampler([Source("a", [shards[0]], 1.0, 50.0), Source("b", [shards[1]], 1.0, 50.0)], 8)
        g = torch.Generator().manual_seed(3)
        return [s.sample(3, g) for _ in range(4)]
    for (x1, y1, n1), (x2, y2, n2) in zip(run(), run()):
        assert torch.equal(x1, x2) and n1 == n2


def test_capped_source_is_excluded_and_recorded(shards):
    # b의 상한 = 40 × 1.0 = 40 토큰 = block 8 기준 시퀀스 5개
    s = MixtureSampler([Source("a", [shards[0]], 1.0, 1000.0), Source("b", [shards[1]], 1000.0, 1.0)], 8)
    g = torch.Generator().manual_seed(0)
    _, _, names = s.sample(5, g)
    assert names == ["b"] * 5  # 가중치가 압도적이라 b만 뽑힘
    assert s.state.exhausted == {"b": 5}
    _, _, names = s.sample(20, g)
    assert set(names) == {"a"}  # 상한 이후에는 b를 뽑지 않는다
    assert s.state.seen_tokens["b"] == 40
    assert "상한 도달" in s.report()


def test_all_sources_exhausted_raises_instead_of_resampling(shards):
    s = MixtureSampler([Source("a", [shards[0]], 1.0, 0.16), Source("b", [shards[1]], 1.0, 0.2)], 8)
    g = torch.Generator().manual_seed(0)
    # a 상한 16 토큰(시퀀스 2개), b 상한 8 토큰(시퀀스 1개) → 시퀀스 3개 뒤 소진
    s.sample(3, g)
    assert set(s.state.exhausted) == {"a", "b"}
    with pytest.raises(DataExhausted, match="출처별 노출"):
        s.sample(1, g)


def test_exhaustion_mid_batch_raises(shards):
    s = MixtureSampler([Source("b", [shards[1]], 1.0, 0.4)], 8)  # 시퀀스 2개
    with pytest.raises(DataExhausted):
        s.sample(3, torch.Generator().manual_seed(0))


def test_state_round_trip(shards):
    srcs = [Source("a", [shards[0]], 1.0, 10.0), Source("b", [shards[1]], 1.0, 10.0)]
    s1 = MixtureSampler(srcs, 8)
    g = torch.Generator().manual_seed(1)
    s1.sample(6, g)
    state, rng = s1.state_dict(), g.get_state()
    expected = s1.sample(6, g)

    s2 = MixtureSampler(srcs, 8)
    s2.load_state_dict(state)
    g2 = torch.Generator()
    g2.set_state(rng)
    got = s2.sample(6, g2)
    assert torch.equal(expected[0], got[0]) and expected[2] == got[2]
    assert s2.state_dict() == s1.state_dict()


def test_state_with_different_sources_rejected(shards):
    s1 = MixtureSampler([Source("a", [shards[0]], 1.0, 10.0)], 8)
    s2 = MixtureSampler([Source("b", [shards[1]], 1.0, 10.0)], 8)
    with pytest.raises(ValueError):
        s2.load_state_dict(s1.state_dict())


def test_shard_choice_proportional_to_usable_positions(tmp_path):
    big = write_shard(tmp_path / "big.bin", np.zeros(1000))
    small = write_shard(tmp_path / "small.bin", np.ones(10))  # block 8 → 시작 위치 2개
    s = MixtureSampler([Source("a", [big, small], 1.0, 100.0)], 8)
    x, _, _ = s.sample(2000, torch.Generator().manual_seed(0))
    share = (x[:, 0] == 1).float().mean().item()
    assert share == pytest.approx(2 / 994, abs=0.004)
