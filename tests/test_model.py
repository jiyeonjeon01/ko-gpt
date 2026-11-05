import math

import pytest
import torch

from config import GPTConfig, model_config
from model import GPT, IGNORE_INDEX, apply_rope, rope_tables


def tiny(**kw) -> GPTConfig:
    return GPTConfig(**{"vocab_size": 64, "block_size": 16, "n_layer": 2, "n_embd": 32, "n_head": 4,
                        "ffn_dim": 64, **kw})


@pytest.mark.parametrize("preset, expected", [("pilot-42m", 42_476_032), ("main-110m", 110_119_680)])
def test_parameter_counts_match_design(preset, expected):
    # 설계안 5-2: 42.5M, 110.1M (블록 + 임베딩 + RMSNorm 가중치)
    cfg = model_config(preset)
    with torch.device("meta"):
        model = GPT(cfg)
    assert model.num_parameters() == expected


def test_weight_tying_shares_one_parameter():
    model = GPT(tiny())
    assert model.lm_head.weight is model.tok_emb.weight
    names = [n for n, _ in model.named_parameters()]
    assert "lm_head.weight" not in names  # 중복 제거되어 한 번만 나온다
    assert all(m.bias is None for m in model.modules() if isinstance(m, torch.nn.Linear))


def test_causal_future_tokens_do_not_change_past_logits():
    torch.manual_seed(0)
    model = GPT(tiny()).eval()
    idx = torch.randint(64, (2, 12))
    changed = idx.clone()
    changed[:, 7:] = (changed[:, 7:] + 1) % 64
    with torch.no_grad():
        a, _ = model(idx)
        b, _ = model(changed)
    assert torch.allclose(a[:, :7], b[:, :7], atol=1e-5)
    assert not torch.allclose(a[:, 7:], b[:, 7:], atol=1e-5)


def test_initial_loss_near_uniform():
    torch.manual_seed(0)
    model = GPT(tiny())
    idx = torch.randint(64, (4, 16))
    _, loss = model(idx, torch.randint(64, (4, 16)))
    assert abs(loss.item() - math.log(64)) < 0.3


def test_ignore_index_excludes_positions():
    torch.manual_seed(0)
    model = GPT(tiny())
    idx = torch.randint(64, (1, 8))
    targets = torch.randint(64, (1, 8))
    logits, _ = model(idx)
    masked = targets.clone()
    masked[0, :5] = IGNORE_INDEX
    _, loss = model(idx, masked)
    expected = torch.nn.functional.cross_entropy(logits[0, 5:].float(), targets[0, 5:])
    assert torch.allclose(loss, expected, atol=1e-5)


def test_all_ignored_targets_raise():
    model = GPT(tiny())
    with pytest.raises(ValueError):
        model(torch.zeros(1, 4, dtype=torch.long), torch.full((1, 4), IGNORE_INDEX))


def test_rope_scores_depend_only_on_relative_position():
    torch.manual_seed(0)
    cos, sin = rope_tables(8, 32, 10000.0)
    q = torch.randn(8)
    k = torch.randn(8)

    def score(i, j):
        qi = apply_rope(q.view(1, 1, 1, 8), cos[i : i + 1], sin[i : i + 1])
        kj = apply_rope(k.view(1, 1, 1, 8), cos[j : j + 1], sin[j : j + 1])
        return (qi * kj).sum().item()

    assert score(5, 2) == pytest.approx(score(13, 10), abs=1e-4)
    assert score(5, 2) != pytest.approx(score(5, 3), abs=1e-4)


def test_input_validation():
    model = GPT(tiny())
    with pytest.raises(ValueError):
        model(torch.zeros(1, 17, dtype=torch.long))  # block_size 초과
    with pytest.raises(ValueError):
        model(torch.zeros(1, 4, dtype=torch.int32))
