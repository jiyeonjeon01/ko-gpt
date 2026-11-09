"""대화 평가 도구: 생성 정지 처리, 고정본만 사용, 자동 지표, 블라인드 채점지"""

import csv
import json
import shutil
import sys
from pathlib import Path

import pytest
import torch

import chat_generate as cg
import chat_metrics as cm
import scoring_sheet as ss
from config import GPTConfig
from model import GPT
from text import EOT_ID, IM_END_ID, IM_START_ID

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import freeze_eval_set as fz


class Scripted(torch.nn.Module):
    """정해진 토큰 순서를 확신하는 가짜 모델 (생성된 토큰 수로 다음 토큰을 정한다)"""

    def __init__(self, seq, prompt_len, vocab=64):
        super().__init__()
        self.seq, self.prompt_len, self.vocab = seq, prompt_len, vocab
        self.config = GPTConfig(vocab_size=vocab, block_size=512, n_layer=1, n_embd=8, n_head=2, ffn_dim=16)

    def forward(self, x, targets=None):
        k = x.shape[1] - self.prompt_len
        logits = torch.full((*x.shape, self.vocab), -1e9)
        logits[..., self.seq[min(k, len(self.seq) - 1)]] = 0.0
        return logits, None


@pytest.mark.parametrize("seq,stop,out", [
    ([20, 21, IM_END_ID], "im_end", [20, 21]),
    ([20, EOT_ID], "eot", [20]),
    ([IM_START_ID], "im_start", []),
    ([22], "max_tokens", [22] * 5),
])
def test_generate_stops(seq, stop, out):
    prompt = [1, 5, 6]
    toks, why = cg.generate(Scripted(seq, len(prompt)), prompt, 5, 0.7, 50, torch.Generator().manual_seed(0), "cpu")
    assert (toks, why) == (out, stop)


def test_eval_set_policy(tmp_path):
    draft = ROOT / "eval/chat120/draft_v0.1.json"
    with pytest.raises(PermissionError, match="고정본"):
        cg.check_eval_set(draft, allow_unfrozen=True)  # chat120 초안은 플래그가 있어도 거부
    other = tmp_path / "mini.json"
    other.write_text(json.dumps({"name": "mini", "status": "draft", "items": []}), encoding="utf-8")
    with pytest.raises(PermissionError):
        cg.check_eval_set(other, allow_unfrozen=False)
    assert cg.check_eval_set(other, allow_unfrozen=True)["name"] == "mini"
    # 고정본은 통과, 고정 뒤 수정되면 거부
    d = tmp_path / "c"
    d.mkdir()
    shutil.copy(draft, d / "draft.json")
    (d / "rubric.md").write_text("기준", encoding="utf-8")
    frozen = fz.freeze(d / "draft.json", 1, d / "rubric.md")
    assert cg.check_eval_set(frozen, False)["status"] == "frozen"
    doc = json.loads(frozen.read_text(encoding="utf-8"))
    doc["items"][0]["messages"][0]["content"] = "바뀐 질문"
    frozen.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="해시"):
        cg.check_eval_set(frozen, False)


def test_generate_main_end_to_end(tmp_path):
    from tokenizer import train_tokenizer
    tok = train_tokenizer(["안녕 반가워 hello"] * 20, vocab_size=300, min_frequency=1)
    tok.save(str(tmp_path / "tok.json"))
    torch.manual_seed(0)
    m = GPT(GPTConfig(vocab_size=300, block_size=128, n_layer=1, n_embd=16, n_head=2, ffn_dim=32))
    ck = tmp_path / "sft.pt"
    torch.save({"kind": "sft", "step": 3, "model": m.state_dict(), "model_config": m.config.__dict__,
                "run": {"model": "tiny", "tokenizer": str(tmp_path / "tok.json")}}, ck)
    es = tmp_path / "mini.json"
    es.write_text(json.dumps({"name": "mini", "status": "draft", "items": [
        {"id": "a1", "category": "인사", "messages": [{"role": "user", "content": "안녕"}],
         "reference": {"kind": "none"}, "expected_language": "ko"}]}), encoding="utf-8")
    monkey = dict(cg.SETTINGS)
    cg.SETTINGS["max_new_tokens"] = 8
    try:
        assert cg.main([str(ck), str(es), "--out", str(tmp_path / "gen"), "--allow-unfrozen", "--device", "cpu"]) == 0
    finally:
        cg.SETTINGS.update(monkey)
    row = json.loads((tmp_path / "gen" / "generations.jsonl").read_text(encoding="utf-8"))
    assert row["id"] == "a1" and row["n_tokens"] == len(row["completion_ids"]) <= 8
    meta = json.loads((tmp_path / "gen" / "meta.json").read_text(encoding="utf-8"))
    assert meta["checkpoint_kind"] == "sft" and meta["eval_status"] == "draft"
    with pytest.raises(FileExistsError):
        cg.main([str(ck), str(es), "--out", str(tmp_path / "gen"), "--allow-unfrozen", "--device", "cpu"])


# ---- 자동 지표 ----


def test_language_detection_follows_rubric():
    assert cm.detect_language("안녕하세요, 반가워요.") == "ko"
    assert cm.detect_language("Hello, nice to meet you.") == "en"
    assert cm.detect_language("파이썬에서 print(len(items))를 쓰면 길이를 출력해요") == "ko"
    assert cm.detect_language("12345 !!!") == "other"
    assert cm.detect_language("ok 네") == "ko"  # 글자 기준 한글 1/3 ≥ 0.3
    assert cm.detect_language("okay okay okay 네") == "en"  # 한글 1/13 < 0.1, 라틴 ≥ 0.5
    assert cm.detect_language("okay okay 네") == "other"  # 한글 1/9: ko 기준(0.3) 미만, en 조건(한글 < 0.1) 불만족


def test_collapse_and_distinct():
    assert cm.collapsed([1, 2, 3, 4] * 4) and not cm.collapsed([1, 2, 3, 4] * 3)
    assert cm.distinct2([1, 2, 1, 2]) == pytest.approx(2 / 3)
    assert cm.distinct2([5]) is None


def test_answer_inclusion_respects_digit_boundaries():
    assert cm.contains_answer("정답은 5개예요", ["5개", "5"])
    assert not cm.contains_answer("15개예요", ["5"])
    assert not cm.contains_answer("18시예요", ["8시"])
    assert cm.contains_answer("저녁 8시예요", ["8시"])
    assert not cm.contains_answer("2,0000원", ["2,000원"]) and cm.contains_answer("2,000원", ["2,000원"])
    assert cm.contains_answer("It is PARIS.", ["Paris"])
    assert not cm.contains_answer("1년은 120일", ["12"])


def test_answer_inclusion_and_aggregate():
    items = [
        {"id": "f1", "category": "사실", "expected_language": "ko", "reference": {"kind": "exact", "accept": ["2,000원", "2000원"]}},
        {"id": "g1", "category": "영어", "expected_language": "en", "reference": {"kind": "none"}},
    ]
    gens = {
        "f1": {"stop_reason": "im_end", "completion": "모두 2 000원이에요.", "completion_ids": [1, 2, 3]},
        "g1": {"stop_reason": "max_tokens", "completion": "안녕하세요", "completion_ids": [7, 8] * 10},
    }
    res = cm.evaluate_run(items, gens)
    o = res["overall"]
    assert o["stop_rate"] == 0.5 and o["language_match_rate"] == 0.5 and o["collapse_rate"] == 0.5
    assert o["answer_included_rate"] == 1.0 and o["answer_included_n"] == 1
    assert res["by_category"]["영어"]["stop_reasons"] == {"max_tokens": 1}
    with pytest.raises(ValueError):
        cm.evaluate_run(items, {"f1": gens["f1"]})


# ---- 블라인드 채점지 ----


def fake_items(n=6):
    return [{"id": f"x{i}", "category": "인사" if i % 2 else "사실", "messages": [{"role": "user", "content": f"질문 {i}"}],
             "reference": {"kind": "exact", "accept": ["답"]} if i % 2 == 0 else {"kind": "none"},
             "expected_language": "ko"} for i in range(n)]


def test_scoring_make_and_collect(tmp_path):
    items = fake_items()
    runs = {name: {it["id"]: {"completion": f"{name} 응답 {it['id']}"} for it in items} for name in ("pre", "post")}
    out = tmp_path / "s"
    ss.make(items, runs, out)
    key = json.loads((out / "key.json").read_text(encoding="utf-8"))["assignment"]
    assert set(key) == {it["id"] for it in items} and all(set(v.values()) == {"pre", "post"} for v in key.values())
    sheet_text = (out / "sheet.csv").read_text(encoding="utf-8-sig")
    assert "pre" not in sheet_text.replace("pre 응답", "") and "post" not in sheet_text.replace("post 응답", "")
    # 응답 본문에 모델 이름이 들어간 것은 가짜 데이터 탓. 열 이름에는 모델 이름이 없다
    with open(out / "sheet.csv", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 12 and "model" not in rows[0]
    with pytest.raises(ValueError, match="빈칸"):
        ss.collect(out)
    # post 응답에 2점, pre에 1점, A/B는 post 선호
    for r in rows:
        model = key[r["item_id"]][r["blind"]]
        for a in ss.AXES:
            r[a] = "2" if model == "post" else "1"
    with open(out / "sheet.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(out / "ab.csv", encoding="utf-8-sig") as f:
        ab = list(csv.DictReader(f))
    for r in ab:
        r["preference"] = next(b for b, mm in key[r["item_id"]].items() if mm == "post")
    ab[0]["preference"] = "같음"
    with open(out / "ab.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(ab[0]))
        w.writeheader()
        w.writerows(ab)
    res = ss.collect(out)
    assert res["axis_mean"]["post"]["naturalness"] == 2 and res["axis_mean"]["pre"]["format"] == 1
    assert res["ab_preference"] == {"post": 5, "같음": 1}
    with pytest.raises(FileExistsError):
        ss.make(items, runs, out)


def test_accept_regex_word_start_and_code_forms():
    items = {it["id"]: it for it in json.loads((Path(__file__).resolve().parents[1] / "eval/chat120/draft_v0.2.json").read_text(encoding="utf-8"))["items"]}
    f13 = items["f13"]["reference"]["accept_regex"]
    for t in ["물이 돼요.", "물.", "물로 변해요"]:
        assert cm.matches_regex(t, f13), t
    for t in ["동물이 됩니다", "물질이 돼요", "물체예요"]:
        assert not cm.matches_regex(t, f13), t
    k09 = items["k09"]["reference"]["accept_regex"]
    for t in ["if key in d:\n    pass", "`key in my_dict`로 확인", "d.keys()", "d.get('a')", "in 연산자를 써요"]:
        assert cm.matches_regex(t, k09), t
    for t in ["print(string)", "파이썬에서는 in Python.", "It is built in python."]:
        assert not cm.matches_regex(t, k09), t
    m05 = items["m05"]["reference"]["accept"]
    assert cm.contains_answer("오후 8시예요", m05) and not cm.contains_answer("오전 8시", m05)
    assert not cm.contains_answer("8시예요", m05)
    gen = {"completion": "동물이 돼요", "completion_ids": [5, 6], "stop_reason": "im_end"}
    assert cm.item_metrics(items["f13"], gen)["answer_included"] is False
