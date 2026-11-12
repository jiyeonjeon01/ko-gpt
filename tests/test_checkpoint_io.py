"""checkpoint_io: 가중치만 읽기. 기존 방식(torch.load 통째로)과 가중치·생성·SFT 결과가 같은지"""

import json

import torch

import chat_generate as cg
import checkpoint_io as cio
import sft_train as st
from config import GPTConfig
from filecache import resident_bytes
from model import GPT
from optim import OptimConfig, build_optimizer
from tests.test_sft_train import CFG, argv, setup  # noqa: F401  (pytest fixture 재사용)

TINY = GPTConfig(vocab_size=300, block_size=64, n_layer=2, n_embd=16, n_head=2, ffn_dim=32)


def pretrain_like_ckpt(path, cfg=TINY, run=None):
    """사전학습 Checkpoint와 같은 모양: 가중치 + Optimizer 상태(한 번 step) + sampler + rng"""
    torch.manual_seed(0)
    m = GPT(cfg)
    opt = build_optimizer(m, OptimConfig(max_lr=1e-3, warmup_steps=1, decay_steps=1, max_steps=2))
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    _, loss = m(x, x)
    loss.backward()
    opt.step()
    torch.save({"model": m.state_dict(), "optimizer": opt.state_dict(), "sampler": {"pos": [1, 2, 3]},
                "rng": {"torch_cpu": torch.get_rng_state()}, "model_config": cfg.__dict__,
                "run": run or {"model": "tiny"}, "step": 7, "tokens_seen": 1234}, path)
    return m


def test_load_weights_keeps_only_needed_keys_and_identical_weights(tmp_path):
    p = tmp_path / "ck.pt"
    pretrain_like_ckpt(p)
    w = cio.load_weights(p)
    assert set(w) == {"model", "model_config", "run", "step", "tokens_seen"}  # optimizer·sampler·rng 없음
    full = torch.load(p, map_location="cpu", weights_only=True)  # 기존 방식
    assert w["model"].keys() == full["model"].keys()
    assert all(torch.equal(w["model"][k], full["model"][k]) for k in full["model"])
    assert cio.meta_of(w) == {k: full[k] for k in ("model_config", "run", "step", "tokens_seen")}


def test_checkpoint_sha256_matches_and_returns_cache(tmp_path):
    import hashlib
    p = tmp_path / "ck.pt"
    pretrain_like_ckpt(p)
    assert cio.checkpoint_sha256(p) == hashlib.sha256(p.read_bytes()).hexdigest()
    cio.checkpoint_sha256(p)
    # tmp_path는 저장소 밖이라 drop_cache가 거부한다 (저장소 파일에만 동작). 결과만 확인하고, 저장소 안 파일은 아래에서
    from filecache import ROOT
    q = ROOT / "runs" / ".test_checkpoint_io.pt"
    q.parent.mkdir(exist_ok=True)
    try:
        q.write_bytes(p.read_bytes())
        assert cio.checkpoint_sha256(q) == hashlib.sha256(p.read_bytes()).hexdigest()
        assert resident_bytes(q) == 0
    finally:
        q.unlink(missing_ok=True)


def test_chat_generate_same_output_as_full_load(tmp_path):
    """고정 seed·같은 설정에서 기존 방식으로 만든 모델과 생성 토큰이 같다"""
    from tokenizer import train_tokenizer
    tok = train_tokenizer(["안녕 반가워 hello"] * 20, vocab_size=300, min_frequency=1)
    tok.save(str(tmp_path / "tok.json"))
    p = tmp_path / "sft.pt"
    pretrain_like_ckpt(p, run={"model": "tiny", "tokenizer": str(tmp_path / "tok.json")})
    ck = torch.load(p, map_location="cpu", weights_only=True)
    ck["kind"] = "sft"
    torch.save(ck, p)

    new_model, _, meta = cg.load_model(p, "cpu")
    assert "optimizer" not in meta and "model" not in meta
    old_model = GPT(GPTConfig(**ck["model_config"]))
    old_model.load_state_dict(ck["model"])
    old_model.eval()
    ids = [5, 6, 7, 8]
    outs = []
    for m in (old_model, new_model):
        gen = torch.Generator().manual_seed(1234)
        outs.append(cg.generate(m, ids, 20, 0.7, 50, gen, "cpu"))
    assert outs[0] == outs[1]


def test_sft_result_identical_to_full_load(setup, monkeypatch):  # noqa: F811
    """SFT를 새 방식과 기존 방식(통째로 읽기)으로 시작했을 때 최종 가중치·Loss 기록이 같다"""
    tmp, init, data = setup
    ck = torch.load(init, weights_only=True)  # 사전학습처럼 Optimizer 상태를 넣는다
    opt = build_optimizer(GPT(CFG), OptimConfig(max_lr=1e-3, warmup_steps=1, decay_steps=1, max_steps=2))
    ck.update({"optimizer": opt.state_dict(), "sampler": {"pos": 1}, "rng": {"torch_cpu": torch.get_rng_state()}})
    torch.save(ck, init)

    assert st.main(argv(tmp / "new", init, data, ["--model", "tiny"])) == 0

    def old_load(path):  # 이전 코드: torch.load 통째로
        return torch.load(path, map_location="cpu", weights_only=True)
    monkeypatch.setattr(st, "load_weights", old_load)
    monkeypatch.setattr(st, "meta_of", lambda w: w)
    assert st.main(argv(tmp / "old", init, data, ["--model", "tiny"])) == 0

    a = torch.load(tmp / "new" / "last.pt", weights_only=True)
    b = torch.load(tmp / "old" / "last.pt", weights_only=True)
    assert all(torch.equal(a["model"][k], b["model"][k]) for k in a["model"])
    strip = lambda p: [",".join(r.split(",")[:4]) for r in p.read_text().splitlines()]  # noqa: E731 (시간 열 제외)
    assert strip(tmp / "new" / "steps.csv") == strip(tmp / "old" / "steps.csv")
    assert (tmp / "new" / "val.csv").read_text() == (tmp / "old" / "val.csv").read_text()
    # SFT Optimizer는 새로 만든다: 사전학습 Optimizer 상태(1 step)를 이어받지 않는다
    assert a["optimizer"]["state"][0]["step"] == a["step"]
    assert json.dumps(a["sft"], sort_keys=True, default=str) == json.dumps(b["sft"], sort_keys=True, default=str)


def test_eval_and_sft_code_do_not_load_full_checkpoint():
    """새 학습 시작·평가·생성 경로에 통째로 읽는 torch.load가 남지 않았는지 (재개 경로는 Optimizer가 필요해 예외)"""
    import eval_bench
    for mod in (cg, eval_bench):
        src = open(mod.__file__, encoding="utf-8").read()
        assert "torch.load(" not in src and "load_weights(" in src
    src = open(st.__file__, encoding="utf-8").read()
    assert src.count("torch.load(") == 1 and "torch.load(args.resume" in src


def mapped(path) -> bool:
    """이 프로세스에 그 파일의 mmap 영역이 남아 있는지 (남아 있으면 그 파일을 가리키는 텐서가 살아 있다)"""
    import gc
    gc.collect()
    return str(path) in open("/proc/self/maps").read()


def test_no_mmap_reference_left_after_loading(tmp_path):
    p = tmp_path / "ck.pt"
    pretrain_like_ckpt(p, run={"model": "tiny", "tokenizer": str(tmp_path / "tok.json")})
    from tokenizer import train_tokenizer
    train_tokenizer(["안녕 hello"] * 20, vocab_size=300, min_frequency=1).save(str(tmp_path / "tok.json"))
    ck = torch.load(p, weights_only=True)
    ck["kind"] = "sft"
    torch.save(ck, p)
    w = cio.load_weights(p)
    assert mapped(p)  # 측정 방법 확인: mmap으로 읽은 동안은 보인다
    del w
    assert not mapped(p)
    model, _, meta = cg.load_model(p, "cpu")
    assert not mapped(p)  # 모델로 복사한 뒤에는 원본 대응 영역이 없다
    assert next(model.parameters()).untyped_storage().data_ptr() != 0


def test_sft_releases_init_mapping_before_training(setup, monkeypatch):  # noqa: F811
    tmp, init, data = setup
    seen = []
    real = st.build_optimizer

    def spy(model, cfg):  # SFT Optimizer를 만드는 시점 = 가중치 복사·해제 직후
        seen.append(mapped(init))
        return real(model, cfg)
    monkeypatch.setattr(st, "build_optimizer", spy)
    assert st.main(argv(tmp / "run", init, data, ["--model", "tiny"])) == 0
    assert seen == [False]


def test_loading_does_not_modify_checkpoint_file(setup):  # noqa: F811
    import hashlib
    tmp, init, data = setup
    before = hashlib.sha256(init.read_bytes()).hexdigest()
    w = cio.load_weights(init)
    for t in w["model"].values():  # mmap 텐서를 고쳐도 파일에는 쓰이지 않는다 (private mapping)
        t.add_(1.0)
    del w
    assert st.main(argv(tmp / "run", init, data, ["--model", "tiny"])) == 0
    assert hashlib.sha256(init.read_bytes()).hexdigest() == before


def test_cache_release_failure_does_not_break_loading(tmp_path, monkeypatch):
    import hashlib
    p = tmp_path / "ck.pt"
    pretrain_like_ckpt(p)
    calls = []

    def boom(*paths):
        calls.extend(paths)
        raise RuntimeError("캐시 반환 실패 (시험)")
    monkeypatch.setattr(cio, "drop_cache", boom)
    assert "RuntimeError" in cio.release(p)[str(p)]
    assert cio.checkpoint_sha256(p) == hashlib.sha256(p.read_bytes()).hexdigest()
    assert calls == [p, p]  # 지정한 그 파일 하나에만 요청한다
    w = cio.load_weights(p)
    assert "model" in w


def test_release_targets_only_the_given_file(monkeypatch):
    got = []
    monkeypatch.setattr(cio, "drop_cache", lambda *p: got.append(p) or {})
    cio.release("runs/x/last.pt")
    assert got == [("runs/x/last.pt",)]
