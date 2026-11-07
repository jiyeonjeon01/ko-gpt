"""선택지 로그우도 평가: lm-eval 템플릿·토큰 분할·점수·지표 재현"""

import sys

import pytest
import torch
import torch.nn.functional as F

import eval_bench as eb
from config import GPTConfig
from model import GPT


# ---- 템플릿 (lm-eval kobest yaml / utils.py, hellaswag utils.py 원문과 같은 문자열) ----


def test_kobest_templates():
    d = eb.kobest_boolq({"paragraph": "P.", "question": "Q?", "label": 1})
    assert (d.context, d.choices, d.gold) == ("P. 질문: Q? 답변: ", ["아니오", "예"], 1)
    d = eb.kobest_copa({"premise": "물을 오래 끓였다.", "question": "결과", "alternative_1": "a", "alternative_2": "b",
                        "label": 0})
    assert d.context == "물을 오래 끓였다.  그래서" and d.choices == ["a", "b"]  # f"{premise} {' 그래서'}" → 공백 2개
    d = eb.kobest_copa({"premise": "x", "question": " 원인 ", "alternative_1": "a", "alternative_2": "b", "label": 1})
    assert d.context == "x  왜냐하면" and d.gold == 1
    d = eb.kobest_wic({"word": "양분", "context_1": "A", "context_2": "B", "label": 0})
    assert d.context == "문장1: A 문장2: B 두 문장에서 양분가 같은 뜻으로 쓰였나?" and d.choices == ["아니오", "예"]
    d = eb.kobest_hellaswag({"context": "C", "ending_1": "1", "ending_2": "2", "ending_3": "3", "ending_4": "4",
                             "label": 3})
    assert (d.context, d.choices, d.gold) == ("문장: C", ["1", "2", "3", "4"], 3)
    d = eb.kobest_sentineg({"sentence": "좋다", "label": 1})
    assert (d.context, d.choices) == ("문장: 좋다 긍부정:", ["부정", "긍정"])


def test_hellaswag_preprocess_matches_lm_eval():
    r = {"activity_label": "Removing ice from car", "ctx_a": "Then, the man writes over the snow.",
         "ctx_b": "a woman smiles", "endings": [" [title] Wait.  Then go", "b", "c", "d"], "label": "2"}
    d = eb.hellaswag(r)
    assert d.context == "Removing ice from car: Then, the man writes over the snow. A woman smiles"
    # lm-eval 순서 그대로: strip이 먼저라 맨 앞 " [title]"은 치환되지 않고 정규식으로만 지워진다 (앞 공백 남음)
    assert d.choices[0] == " Wait. Then go"
    assert eb.hellaswag_preprocess("Do it [title] Next step") == "Do it. Next step"
    assert d.gold == 2
    assert eb.hellaswag_preprocess("a [header] b") == "a b"


# ---- 토큰 분할 ----


def char_encode(tok, text):
    return [ord(ch) % 50 + 1 for ch in text]


def test_encode_pair_moves_trailing_space(monkeypatch):
    monkeypatch.setattr(eb, "encode", char_encode)
    ctx, cont, ok = eb.encode_pair(None, "답변: ", " 예")
    assert ctx == char_encode(None, "답변:") and cont == char_encode(None, "  예") and ok


def test_encode_pair_reports_boundary_merge(monkeypatch):
    # 문맥 끝과 선택지 앞이 합쳐지는 토크나이저: "ab" → 99 하나
    def merging(tok, text):
        out, i = [], 0
        while i < len(text):
            if text[i:i + 2] == "ab":
                out.append(99)
                i += 2
            else:
                out.append(ord(text[i]))
                i += 1
        return out
    monkeypatch.setattr(eb, "encode", merging)
    ctx, cont, ok = eb.encode_pair(None, "xa", "b")
    assert not ok and ctx == [ord("x"), ord("a")] and cont == []  # lm-eval처럼 길이로 자르면 선택지가 사라진다


# ---- 점수 ----


def tiny_model():
    torch.manual_seed(0)
    m = GPT(GPTConfig(vocab_size=64, block_size=16, n_layer=2, n_embd=32, n_head=4, ffn_dim=64))
    return m.eval()


def manual_ll(model, ctx, cont):
    x = torch.tensor([ctx + cont])
    logits, _ = model(x[:, :-1])
    lp = F.log_softmax(logits.float(), -1)[0]
    return sum(float(lp[len(ctx) - 1 + j, t].detach()) for j, t in enumerate(cont))


def test_score_matches_manual_and_is_batch_invariant():
    model = tiny_model()
    reqs = [([1, 2, 3], [4, 5]), ([7], [8]), ([1, 2, 3, 4, 5, 6, 7], [9, 10, 11]), ([3, 3], [3])]
    batched = eb.score(model, reqs, max_len=16, device="cpu", batch_tokens=64)
    one = [eb.score(model, [r], max_len=16, device="cpu")[0] for r in reqs]
    for (ll_b, g_b), (ll_1, g_1), (ctx, cont) in zip(batched, one, reqs):
        assert ll_b == pytest.approx(manual_ll(model, ctx, cont), abs=1e-4)
        assert ll_b == pytest.approx(ll_1, abs=1e-4) and g_b == g_1


def test_score_truncates_left_like_lm_eval():
    model = tiny_model()
    ctx, cont = list(range(1, 30)), [5, 6]
    ll, _ = eb.score(model, [(ctx, cont)], max_len=16, device="cpu")[0]
    kept = (ctx + cont)[-17:]  # 입력 16개 + 마지막 정답 1개
    assert ll == pytest.approx(manual_ll(model, kept[:-2], kept[-2:]), abs=1e-4)


def test_score_rejects_empty():
    with pytest.raises(ValueError):
        eb.score(tiny_model(), [([], [1])], max_len=16, device="cpu")
    with pytest.raises(ValueError):
        eb.score(tiny_model(), [([1], [])], max_len=16, device="cpu")


class Oracle(torch.nn.Module):
    """토큰 good 다음에 오는 것은 늘 good이라고 확신하는 가짜 모델"""

    def __init__(self, good, vocab=64):
        super().__init__()
        self.good, self.vocab = good, vocab

    def forward(self, x, targets=None):
        logits = torch.zeros(*x.shape, self.vocab)
        logits[..., self.good] = 20.0
        return logits, None


def test_oracle_gets_full_accuracy():
    docs = [eb.Doc("c", ["bad", "good"], 1), eb.Doc("c", ["good", "bad"], 0)]
    ids = {"c": [1], " good": [7], " bad": [8]}
    lls = [[eb.score(Oracle(7), [(ids["c"], ids[" " + ch])], 16, "cpu")[0][0] for ch in d.choices] for d in docs]
    res, items = eb.summarize("kobest_copa", docs, lls)
    assert res["acc"]["value"] == 1.0 and res["f1"]["value"] == 1.0
    assert [it["pred"] for it in items] == [1, 0]


def test_acc_norm_divides_by_choice_characters():
    docs = [eb.Doc("c", ["a", "bbbb", "c", "d"], 1)]
    lls = [[-2.0, -4.0, -3.0, -3.0]]  # 합은 0번이 최대, 문자 수로 나누면 1번(-1.0)이 최대
    res, items = eb.summarize("hellaswag", docs, lls)
    assert items[0]["pred"] == 0 and items[0]["pred_norm"] == 1
    assert res["acc"]["value"] == 0.0 and res["acc_norm"]["value"] == 1.0


# ---- 지표 ----


def test_macro_f1_matches_sklearn_examples():
    # sklearn 문서 예시: f1_score([0,1,2,0,1,2], [0,2,1,0,0,1], average="macro") == 0.26666...
    assert eb.macro_f1([0, 1, 2, 0, 1, 2], [0, 2, 1, 0, 0, 1]) == pytest.approx(0.266666, abs=1e-5)
    # 늘 한 클래스만 예측: 예측 안 된 클래스 F1 = 0
    assert eb.macro_f1([0, 0, 1, 1], [1, 1, 1, 1]) == pytest.approx((0 + 2 * 2 / (2 * 2 + 2)) / 2)


def test_binomial_and_intervals():
    assert eb.binom_two_sided_p(10, 10, 0.5) == pytest.approx(2 / 1024)
    assert eb.binom_two_sided_p(50, 100, 0.5) == pytest.approx(1.0)
    assert eb.binom_two_sided_p(25, 100, 0.25) == pytest.approx(1.0, abs=1e-6)
    lo, hi = eb.wilson(50, 100)
    assert lo < 0.5 < hi and hi - 0.5 == pytest.approx(0.5 - lo)
    assert eb.mean_stderr([1.0, 0.0, 1.0, 0.0]) == pytest.approx(0.2886751, abs=1e-6)
    lo, hi = eb.bootstrap_f1([0, 1] * 50, [0, 1] * 50)
    assert lo == hi == 1.0


def test_refuses_existing_output(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["eval_bench.py", "x.pt", "--out", str(tmp_path)])
    with pytest.raises(FileExistsError):
        eb.main()
