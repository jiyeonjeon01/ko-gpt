"""학습 루프: 재개 동일성, 데이터 소진 시 중단, 덮어쓰기 방지 (CPU)"""

import json

import numpy as np
import pytest
import torch

import config
import train
from tokenizer import file_sha256, train_tokenizer


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setitem(config.MODEL_PRESETS, "tiny", {"n_layer": 2, "n_embd": 32, "n_head": 4, "ffn_dim": 64})
    tok = train_tokenizer(["가나다라 마바사 abc def 123"] * 20, vocab_size=300, min_frequency=1)
    tok_path = tmp_path / "tokenizer.json"
    tok.save(str(tok_path))

    def make_data(train_tokens: int, max_repeat: float):
        rng = np.random.default_rng(0)
        sources = []
        for name in ("a", "b"):
            shards = {}
            for split, n in (("train", train_tokens), ("val", 200)):
                path = tmp_path / f"{name}_{split}.bin"
                rng.integers(16, tok.get_vocab_size(), n).astype(np.uint16).tofile(path)
                shards[split] = [{"path": str(path), "tokens": n, "sha256": file_sha256(path)}]
            sources.append({"name": name, "weight": 1.0, "max_repeat": max_repeat, **shards})
        data = {"tokenizer": str(tok_path), "block_size": 16, "sources": sources}
        path = tmp_path / f"data_{train_tokens}_{max_repeat}.json"
        path.write_text(json.dumps(data))
        return path
    return tmp_path, make_data


def args(out, data=None, steps=6, extra=()):
    a = ["--out-dir", str(out), "--micro-batch", "2", "--grad-accum", "2", "--max-steps", str(steps),
         "--eval-interval", "3", "--eval-batches", "1", "--warmup-steps", "2", "--max-lr", "1e-3"]
    if data is not None:
        a += ["--model", "tiny", "--data", str(data)]
    return a + list(extra)


def test_resume_matches_uninterrupted_run(setup):
    tmp, make_data = setup
    data = make_data(5000, 100.0)
    assert train.main(args(tmp / "full", data, steps=6)) == 0
    assert train.main(args(tmp / "split", data, steps=3)) == 0
    assert train.main(["--resume", str(tmp / "split" / "last.pt")] + args(tmp / "split", steps=6)[2:]) == 0

    full = torch.load(tmp / "full" / "last.pt", weights_only=True)
    split = torch.load(tmp / "split" / "last.pt", weights_only=True)
    assert split["step"] == full["step"] == 6 and split["resume_count"] == 1
    assert split["tokens_seen"] == full["tokens_seen"]
    assert split["sampler"] == full["sampler"]
    for k in full["model"]:
        assert torch.allclose(full["model"][k], split["model"][k], atol=1e-6), k


def test_resume_rejects_changed_settings(setup):
    tmp, make_data = setup
    data = make_data(5000, 100.0)
    assert train.main(args(tmp / "run", data, steps=3)) == 0
    with pytest.raises(ValueError, match="학습 설정"):
        train.main(["--resume", str(tmp / "run" / "last.pt")] + args(tmp / "run", steps=6)[2:]
                   + ["--max-lr", "2e-3"])


def test_fresh_run_refuses_existing_output(setup):
    tmp, make_data = setup
    data = make_data(5000, 100.0)
    assert train.main(args(tmp / "run", data, steps=3)) == 0
    with pytest.raises(FileExistsError):
        train.main(args(tmp / "run", data, steps=3))


def test_stops_and_reports_when_all_sources_exhausted(setup, capsys):
    tmp, make_data = setup
    # 출처마다 고유 200 토큰 × 상한 1배, Step당 2×2 시퀀스 × 16토큰 = 64토큰 → 몇 Step 안에 둘 다 소진
    data = make_data(200, 1.0)
    code = train.main(args(tmp / "run", data, steps=50))
    assert code == train.EXIT_EXHAUSTED
    out = capsys.readouterr().out
    assert "노출 상한" in out and "상한 도달" in out
    ckpt = torch.load(tmp / "run" / "last.pt", weights_only=True)
    assert ckpt["step"] < 50
    assert set(ckpt["sampler"]["exhausted"]) == {"a", "b"}


def test_stop_file_saves_checkpoint_and_resume_continues(setup):
    tmp, make_data = setup
    data = make_data(5000, 100.0)
    out = tmp / "run"
    assert train.main(args(out, data, steps=3)) == 0
    (out / "STOP").write_text("[monitor] 시험용 정지 요청\n", encoding="utf-8")
    resume = ["--resume", str(out / "last.pt")] + args(out, steps=6)[2:]
    assert train.main(resume) == train.EXIT_STOPPED
    stops = (out / "stops.csv").read_text(encoding="utf-8")
    assert "시험용 정지 요청" in stops
    assert torch.load(out / "last.pt", weights_only=True)["step"] == 3  # 업데이트 없이 저장만
    (out / "STOP").unlink()
    assert train.main(resume) == 0
    assert torch.load(out / "last.pt", weights_only=True)["step"] == 6


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA 필요")
def test_full_loop_on_cuda(setup):
    # CPU 테스트만으로는 장치 불일치(평가의 바이트 길이 표 인덱싱)를 잡지 못했다 (Phase 0 Smoke에서 발견)
    tmp, make_data = setup
    data = make_data(5000, 100.0)
    assert train.main(args(tmp / "run", data, steps=3)) == 0
    assert (tmp / "run" / "eval.csv").exists() and (tmp / "run" / "last.pt").exists()


def test_eval_writes_per_source_bpb(setup):
    tmp, make_data = setup
    data = make_data(5000, 100.0)
    assert train.main(args(tmp / "run", data, steps=3)) == 0
    header = (tmp / "run" / "eval.csv").read_text().splitlines()[0].split(",")
    assert {"a_bpb", "b_bpb", "a_loss", "seen_a"} <= set(header)
