"""SFT 학습: 패딩·마스크, 응답 토큰 가중 Loss, 결정론, 저장·정지·재개 (CPU, 작은 모델)"""

import json

import pytest
import torch

import sft_data as sd
import sft_train as st
import train as T
from config import GPTConfig
from model import GPT, IGNORE_INDEX
from optim import OptimConfig, build_optimizer
from text import PAD_ID

CFG = GPTConfig(vocab_size=256, block_size=128, n_layer=2, n_embd=32, n_head=4, ffn_dim=64)


def enc(text):
    return [16 + (ord(c) % 200) for c in text]


def rows(n=24, seed=0):
    g = torch.Generator().manual_seed(seed)
    out = []
    for i in range(n):
        k = int(torch.randint(1, 4, (1,), generator=g))
        msgs = []
        for t in range(k):
            msgs += [{"role": "user", "content": "질문" * (1 + t) + str(i)},
                     {"role": "assistant", "content": "답" * int(torch.randint(1, 20, (1,), generator=g))}]
        ids, mask = sd.render_conversation(msgs, enc)
        out.append({"id": f"c{i}", "source": "X", "lang": "ko" if i % 3 else "en", "turns": k, "ids": ids, "mask": mask})
    return out


def model():
    torch.manual_seed(0)
    return GPT(CFG)


def test_micro_batches_padding_and_targets():
    rs = rows()
    batches = st.micro_batches(rs, list(range(len(rs))), micro_tokens=200)
    seen = 0
    for x, y in batches:
        assert x.numel() <= 200 or x.shape[0] == 1
        for b in range(x.shape[0]):
            # 실제 길이 뒤는 PAD / IGNORE
            n = int((x[b] != PAD_ID).sum())
            assert (x[b, n:] == PAD_ID).all() and (y[b, n:] == IGNORE_INDEX).all()
        seen += x.shape[0]
    assert seen == len(rs)
    # 응답 토큰 수 = 각 대화 mask[1:]의 합 (causal shift)
    assert sum(int((y != IGNORE_INDEX).sum()) for _, y in batches) == sum(sum(r["mask"][1:]) for r in rs)


def test_targets_are_exactly_assistant_tokens():
    r = rows(1)[0]
    (x, y), = st.micro_batches([r], [0], 10_000)
    kept = (y[0] != IGNORE_INDEX).nonzero().flatten().tolist()
    assert kept == [t for t in range(len(r["ids"]) - 1) if r["mask"][t + 1]]
    assert y[0, kept].tolist() == [i for i, m in zip(r["ids"], r["mask"]) if m]


@torch.no_grad()
def test_padding_does_not_change_per_conversation_loss():
    m, rs = model(), rows()
    alone = sum(float(st.summed_loss(m, *st.micro_batches(rs, [i], 10_000)[0], "cpu")[0]) for i in range(len(rs)))
    together = sum(float(st.summed_loss(m, x, y, "cpu")[0]) for x, y in st.micro_batches(rs, list(range(len(rs))), 10_000))
    assert together == pytest.approx(alone, rel=1e-5)


def test_loss_is_response_token_weighted_and_split_invariant():
    rs = rows()
    idx = list(range(len(rs)))
    # 정의: 모든 응답 토큰의 CE 평균 (대화별 평균의 평균이 아님)
    m = model()
    s, n = 0.0, 0
    for i in idx:
        ls, k = st.summed_loss(m, *st.micro_batches(rs, [i], 10_000)[0], "cpu")
        s, n = s + ls.item(), n + k
    expected = s / n
    grads = []
    for micro in (10_000, 150):  # 한 묶음 vs 여러 micro-batch
        m = model()
        opt = build_optimizer(m, OptimConfig(max_lr=0.0, warmup_steps=0, max_steps=1))
        loss, gn, ntok = st.sft_step(m, opt, st.micro_batches(rs, idx, micro), 0.0, 1e9, "cpu")
        assert loss == pytest.approx(expected, rel=1e-5) and ntok == n
        grads.append(torch.cat([p.grad.flatten() for p in m.parameters()]))
    assert len(st.micro_batches(rs, idx, 150)) > 1
    assert torch.allclose(grads[0], grads[1], atol=1e-6)


def test_step_plan_is_deterministic_and_covers_each_epoch():
    p1 = st.step_plan(10, 4, 2, 7)
    assert p1 == st.step_plan(10, 4, 2, 7) and len(p1) == 6
    assert sorted(i for s in p1[:3] for i in s) == list(range(10))
    assert p1[:3] != p1[3:]  # epoch마다 다른 순서


# ---- 실행 (CPU) ----


@pytest.fixture
def setup(tmp_path):
    from tokenizer import train_tokenizer
    tok = train_tokenizer(["질문 답 안녕 abc"] * 20, vocab_size=300, min_frequency=1)
    tok_path = tmp_path / "tokenizer.json"
    tok.save(str(tok_path))
    rs = rows(40)
    data = tmp_path / "data"
    data.mkdir()
    train_rows, val_rows = rs[:32], rs[32:]
    manifest = {"train_sha256": sd.write_jsonl(data / "train.jsonl", train_rows),
                "val_sha256": sd.write_jsonl(data / "val.jsonl", val_rows), "tokenizer": str(tok_path)}
    (data / "manifest.json").write_text(json.dumps(manifest))
    torch.manual_seed(0)
    init = tmp_path / "init.pt"
    m = GPT(CFG)
    torch.save({"model": m.state_dict(), "model_config": CFG.__dict__, "run": {"model": "tiny"}, "step": 100}, init)
    return tmp_path, init, data


def argv(out, init=None, data=None, extra=()):
    a = ["--out-dir", str(out), "--conv-per-step", "8", "--micro-tokens", "300", "--eval-interval", "2",
         "--max-lr", "1e-3"]
    if init:
        a += ["--init", str(init), "--data-dir", str(data)]
    return a + list(extra)


def test_full_run_writes_logs_and_sft_checkpoint(setup):
    tmp, init, data = setup
    out = tmp / "run"
    assert st.main(argv(out, init, data, ["--model", "tiny"])) == 0
    ck = torch.load(out / "last.pt", weights_only=True)
    assert ck["kind"] == "sft" and ck["step"] == ck["total_steps"] == 4
    assert ck["run"]["tokenizer"] and ck["sft"]["init_sha256"] == T.file_sha256(init)
    val = (out / "val.csv").read_text().splitlines()
    assert val[1].startswith("0,") and len(val) == 1 + 3  # step 0, 2, 4
    steps = (out / "steps.csv").read_text().splitlines()
    assert len(steps) == 5
    with pytest.raises(FileExistsError):
        st.main(argv(out, init, data))
    with pytest.raises(ValueError, match="모델"):
        st.main(argv(tmp / "other", init, data, ["--model", "pilot-42m"]))


def test_stop_and_resume_matches_uninterrupted(setup, monkeypatch):
    tmp, init, data = setup
    assert st.main(argv(tmp / "full", init, data)) == 0
    out = tmp / "split"
    real = st.wsd_lr

    def lr_and_stop(step, cfg):
        if step == 2:
            (out / "STOP").write_text("시험 정지")
        return real(step, cfg)
    monkeypatch.setattr(st, "wsd_lr", lr_and_stop)
    assert st.main(argv(out, init, data)) == T.EXIT_STOPPED
    monkeypatch.setattr(st, "wsd_lr", real)
    assert torch.load(out / "last.pt", weights_only=True)["step"] == 3
    (out / "STOP").unlink()
    with pytest.raises(ValueError, match="학습 설정"):
        st.main(["--resume", str(out / "last.pt")] + argv(out)[2:] + ["--max-lr", "2e-3"])
    assert st.main(["--resume", str(out / "last.pt")] + argv(out)[2:]) == 0
    a = torch.load(tmp / "full" / "last.pt", weights_only=True)
    b = torch.load(out / "last.pt", weights_only=True)
    assert b["step"] == 4 and b["resume_count"] == 1
    for k in a["model"]:
        assert torch.allclose(a["model"][k], b["model"][k], atol=1e-6), k


def test_deterministic_flag(setup):
    tmp, init, data = setup
    try:
        assert st.main(argv(tmp / "det", init, data, ["--deterministic"])) == 0
        assert torch.are_deterministic_algorithms_enabled()
        assert torch.load(tmp / "det" / "last.pt", weights_only=True)["sft"]["deterministic"] is True
    finally:
        torch.use_deterministic_algorithms(False)


def test_rejects_tampered_data(setup):
    tmp, init, data = setup
    with open(data / "train.jsonl", "a") as f:
        f.write(json.dumps({"id": "x", "source": "X", "lang": "ko", "turns": 1, "ids": [1, 2, 3], "mask": [0, 1, 1]}) + "\n")
    with pytest.raises(ValueError, match="manifest"):
        st.main(argv(tmp / "run", init, data))
