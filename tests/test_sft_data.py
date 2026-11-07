"""SFT 데이터 생성: 평가 문항 제외, 길이 제외, 중복 제거, Val 분리, 영어 비율, S11 반복·치환"""

import pytest

import sft_data as sd


def enc(text):
    return [16 + (ord(c) % 200) for c in text]


def conv(u, a, *more):
    msgs = [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
    for i, t in enumerate(more):
        msgs.append({"role": "user" if i % 2 == 0 else "assistant", "content": t})
    return msgs


def loaders(n_ko=200, n_en=200):
    ko = [("ko", conv(f"한국어 질문 {i}번", f"한국어 답변입니다 {i}" * (1 + i % 3))) for i in range(n_ko)]
    ko += [("ko", conv("대한민국의 수도는?", "서울입니다"))]  # 평가 문항과 같음 → 제외
    ko += [("ko", conv("긴 질문", "가" * 2000))]  # 길이 초과 → 제외
    ko += [ko[0]]  # 완전 중복 → 제외
    en_multi = [("en", conv(f"question {i}", f"answer {i} " * 5, "follow up", f"second answer {i}")) for i in range(n_en)]
    en_single = [("en", conv("single", "only one turn"))] * 3
    s11 = [("ko", conv("너 누구야?", "저는 {MODEL_NAME}이에요.")), ("ko", conv("오랜만", "반가워요", "응", "잘 지냈어요?"))]
    return {"KO": lambda: iter(ko), "EN": lambda: iter(en_multi + en_single), "S11_direct": lambda: iter(s11)}


CFG = {"name": "t", "seed": 1, "val_frac": 0.1, "max_len": 512, "model_name": "ko-gpt-42m-pilot",
       "korean": [{"source": "KO"}], "s11": {"enabled": True, "repeat": 3},
       "english": {"source": "EN", "multi_turn_only": True, "ratio_to_korean_loss_tokens": 0.5}}
EVAL = [("chat120:f01", "대한민국의 수도는?")]


def test_build_composition():
    res = sd.build(CFG, enc, EVAL, loaders())
    st = res["source_stats"]["KO/ko"]
    assert st["excluded_eval_overlap"] == 1 and st["excluded_over_max_len"] == 1 and st["excluded_duplicate"] == 1
    assert res["excluded"]["eval_overlap"][0]["eval_item"] == "chat120:f01"
    srcs = {e["source"] for e in res["train"]}
    assert srcs == {"KO", "S11_direct", "EN"}
    # 영어: 다중 턴만, 목표(한국어 학습 응답 토큰 × 0.5, S11 반복 포함)를 넘지 않음
    en = [e for e in res["train"] if e["source"] == "EN"]
    assert en and all(e["turns"] >= 2 for e in en)
    s = res["summary"]
    ko_loss = sum(sum(e["mask"]) for e in res["train"] if e["lang"] == "ko")
    assert s["korean_train_loss_tokens"] == ko_loss
    assert s["english_train_loss_tokens"] <= s["english_target_loss_tokens"] == int(0.5 * ko_loss)
    assert s["english_train_loss_tokens"] > 0.9 * s["english_target_loss_tokens"]  # 거의 채운다
    # S11: 3회 반복, Val 없음, 이름 치환
    s11 = [e for e in res["train"] if e["source"] == "S11_direct"]
    assert len(s11) == 6 and not [e for e in res["val"] if e["source"] == "S11_direct"]
    name_ids = enc("ko-gpt-42m-pilot")
    assert any(e["ids"][i:i + len(name_ids)] == name_ids for e in s11 for i in range(len(e["ids"])))


def test_val_split_is_deterministic_and_disjoint():
    a = sd.build(CFG, enc, EVAL, loaders())
    b = sd.build(CFG, enc, EVAL, loaders())
    assert [e["id"] for e in a["train"]] == [e["id"] for e in b["train"]]
    assert [e["id"] for e in a["val"]] == [e["id"] for e in b["val"]]
    assert not {e["id"] for e in a["train"]} & {e["id"] for e in a["val"]}
    ko_val = [e for e in a["val"] if e["source"] == "KO"]
    assert 5 <= len(ko_val) <= 40  # 약 10% (200개 중)


def test_without_s11_and_mask_is_assistant_only():
    cfg = {**CFG, "s11": {"enabled": False, "repeat": 3}}
    res = sd.build(cfg, enc, EVAL, loaders())
    assert not [e for e in res["train"] if e["source"] == "S11_direct"]
    assert res["summary"]["s11_repeat"] == 0
    for e in res["train"][:20]:
        assert len(e["ids"]) == len(e["mask"]) and e["mask"][0] == 0 and sum(e["mask"]) > 0


def test_missing_dolly_ko_raises(monkeypatch, tmp_path):
    monkeypatch.setitem(sd.SOURCE_FILES, "S4_dolly_ko", [tmp_path / "none.jsonl"])
    with pytest.raises(FileNotFoundError):
        list(sd.load_dolly_ko())
