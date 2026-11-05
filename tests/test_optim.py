import math

import pytest
import torch

from config import GPTConfig
from model import GPT
from optim import NonFiniteError, OptimConfig, build_optimizer, train_step, wsd_lr


def test_wsd_schedule_shape():
    cfg = OptimConfig(max_lr=1.0, warmup_steps=10, decay_steps=20, max_steps=100, min_lr_ratio=0.1)
    assert wsd_lr(0, cfg) == pytest.approx(0.1)
    assert wsd_lr(9, cfg) == pytest.approx(1.0)
    assert wsd_lr(50, cfg) == 1.0
    assert wsd_lr(79, cfg) == 1.0  # 감쇠 직전
    assert wsd_lr(80, cfg) == pytest.approx(1.0 - 0.9 / 20)
    assert wsd_lr(99, cfg) == pytest.approx(0.1)  # 마지막 업데이트
    lrs = [wsd_lr(s, cfg) for s in range(80, 100)]
    assert all(a > b for a, b in zip(lrs, lrs[1:]))


def test_weight_decay_groups_count_tied_weight_once():
    model = GPT(GPTConfig(vocab_size=64, block_size=8, n_layer=2, n_embd=32, n_head=4, ffn_dim=64))
    opt = build_optimizer(model, OptimConfig())
    decay, no_decay = opt.param_groups
    total = sum(p.numel() for g in opt.param_groups for p in g["params"])
    assert total == model.num_parameters()
    assert all(p.dim() == 1 for p in no_decay["params"])


def tiny_model():
    torch.manual_seed(0)
    return GPT(GPTConfig(vocab_size=64, block_size=8, n_layer=2, n_embd=32, n_head=4, ffn_dim=64))


def test_accumulation_equals_full_batch():
    x = torch.randint(64, (4, 8), generator=torch.Generator().manual_seed(1))
    y = torch.randint(64, (4, 8), generator=torch.Generator().manual_seed(2))
    a, b = tiny_model(), tiny_model()
    oa, ob = build_optimizer(a, OptimConfig()), build_optimizer(b, OptimConfig())
    la, na = train_step(a, oa, [(x, y)], 1e-3, 1.0, "cpu", autocast=False)
    lb, nb = train_step(b, ob, [(x[:2], y[:2]), (x[2:], y[2:])], 1e-3, 1.0, "cpu", autocast=False)
    assert la == pytest.approx(lb, rel=1e-5) and na == pytest.approx(nb, rel=1e-4)
    for pa, pb in zip(a.parameters(), b.parameters()):
        assert torch.allclose(pa, pb, atol=1e-6)


def test_non_finite_loss_raises_without_update():
    model = tiny_model()
    opt = build_optimizer(model, OptimConfig())
    before = [p.detach().clone() for p in model.parameters()]
    with torch.no_grad():
        model.tok_emb.weight[0] = math.nan
    x = torch.zeros(1, 8, dtype=torch.long)
    with pytest.raises(NonFiniteError):
        train_step(model, opt, [(x, x)], 1e-3, 1.0, "cpu", autocast=False)
    for p, q in zip(list(model.parameters())[1:], before[1:]):
        assert torch.equal(p, q)
