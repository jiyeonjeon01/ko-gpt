"""블라인드 사람 채점지 만들기와 집계 (eval/chat120/rubric.md 2·3절)

    # 채점지 만들기: 두 모델의 응답을 문항마다 무작위로 A/B에 배정. 모델 이름은 key.json에만 둔다
    uv run python scoring_sheet.py make eval/chat120/frozen_v1.json pre=runs/.../gen_pre post=runs/.../gen_post \
        --out runs/phase2/sft42m/scoring

    # 채점 뒤 집계 (sheet.csv의 점수 칸과 ab.csv의 preference 칸을 채운 뒤)
    uv run python scoring_sheet.py collect runs/phase2/sft42m/scoring

- sheet.csv: 응답 하나당 한 줄. format, naturalness, relevance, factuality, safety에 0·1·2를 적는다
- ab.csv: 문항 하나당 한 줄. preference에 A, B, 같음 중 하나
- sheet.html: 읽기용 (같은 내용)
- key.json: 블라인드 키. 채점이 끝날 때까지 열지 않는다
- CSV는 엑셀에서 한글이 깨지지 않도록 UTF-8 BOM으로 쓴다
"""

import argparse
import csv
import html
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

AXES = ("format", "naturalness", "relevance", "factuality", "safety")
AXES_KO = {"format": "형식", "naturalness": "자연스러움", "relevance": "관련성", "factuality": "사실 정확성",
           "safety": "안전성"}


def prompt_text(item: dict) -> str:
    return "\n".join(f"[{'사용자' if m['role'] == 'user' else '모델'}] {m['content']}" for m in item["messages"])


def reference_text(item: dict) -> str:
    ref = item["reference"]
    if ref["kind"] == "exact":
        s = "정답: " + " / ".join(ref["accept"])
        return s + (f" ({ref['note']})" if ref.get("note") else "")
    if ref["kind"] == "key_points":
        return "핵심: " + " · ".join(ref["points"])
    return "(정답 없음: 틀린 사실 주장이 없으면 사실 정확성 2점)"


def make(items: list[dict], runs: dict[str, dict], out: Path, seed: int = 20261107) -> None:
    if len(runs) != 2:
        raise ValueError("블라인드 A/B 채점지는 두 모델을 비교한다")
    if out.exists():
        raise FileExistsError(f"{out}가 이미 있다")
    names = list(runs)
    rng = random.Random(seed)
    order = list(items)
    rng.shuffle(order)  # 문항 순서도 섞어 범주별로 몰아서 채점하지 않게
    key, sheet, ab = {}, [], []
    for it in order:
        pair = names[:]
        rng.shuffle(pair)
        key[it["id"]] = {"A": pair[0], "B": pair[1]}
        responses = {}
        for blind, name in zip("AB", pair):
            responses[blind] = runs[name][it["id"]]["completion"]
            sheet.append({"item_id": it["id"], "blind": blind, "category": it["category"], "prompt": prompt_text(it),
                          "reference": reference_text(it), "safety_rule": it.get("safety", ""),
                          "response": responses[blind], **{a: "" for a in AXES}, "comment": ""})
        ab.append({"item_id": it["id"], "category": it["category"], "prompt": prompt_text(it),
                   "A": responses["A"], "B": responses["B"], "preference": "", "comment": ""})
    out.mkdir(parents=True)
    for name, rows in (("sheet.csv", sheet), ("ab.csv", ab)):
        with open(out / name, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    (out / "key.json").write_text(json.dumps({"seed": seed, "assignment": key}, ensure_ascii=False, indent=1),
                                  encoding="utf-8")
    parts = ["<!doctype html><meta charset='utf-8'><title>채점지</title>",
             "<style>body{font-family:'Malgun Gothic',sans-serif;max-width:900px;margin:auto}"
             "pre{white-space:pre-wrap;background:#f5f5f3;padding:8px}h3{margin-top:2em}</style>",
             "<p>점수는 sheet.csv에, A/B 선호는 ab.csv에 적는다. 기준: rubric.md. key.json은 채점이 끝날 때까지 열지 않는다.</p>"]
    for row in ab:
        it_rows = [r for r in sheet if r["item_id"] == row["item_id"]]
        parts.append(f"<h3>{html.escape(row['item_id'])} · {html.escape(row['category'])}</h3>"
                     f"<pre>{html.escape(row['prompt'])}</pre><p>{html.escape(it_rows[0]['reference'])}</p>")
        if it_rows[0]["safety_rule"]:
            parts.append(f"<p>안전 조건: {html.escape(it_rows[0]['safety_rule'])}</p>")
        for r in it_rows:
            parts.append(f"<b>{r['blind']}</b><pre>{html.escape(r['response'])}</pre>")
    (out / "sheet.html").write_text("\n".join(parts), encoding="utf-8")


def collect(d: Path) -> dict:
    key = json.loads((d / "key.json").read_text(encoding="utf-8"))["assignment"]
    with open(d / "sheet.csv", encoding="utf-8-sig") as f:
        sheet = list(csv.DictReader(f))
    with open(d / "ab.csv", encoding="utf-8-sig") as f:
        ab = list(csv.DictReader(f))
    errors, scores = [], defaultdict(lambda: defaultdict(list))
    by_cat = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in sheet:
        model = key[r["item_id"]][r["blind"]]
        for axis in AXES:
            v = r[axis].strip()
            if v not in ("0", "1", "2"):
                errors.append(f"{r['item_id']}{r['blind']} {AXES_KO[axis]}: '{v}' (0·1·2만)")
                continue
            scores[model][axis].append(int(v))
            by_cat[model][r["category"]][axis].append(int(v))
    pref = defaultdict(int)
    for r in ab:
        v = r["preference"].strip().upper()
        if v in ("A", "B"):
            pref[key[r["item_id"]][v]] += 1
        elif r["preference"].strip() in ("같음", "SAME", "="):
            pref["같음"] += 1
        else:
            errors.append(f"{r['item_id']} A/B 선호: '{r['preference']}' (A, B, 같음)")
    if errors:
        raise ValueError("채점지에 빈칸이나 잘못된 값이 있다:\n  " + "\n  ".join(errors[:30])
                         + (f"\n  … 외 {len(errors) - 30}개" if len(errors) > 30 else ""))
    mean = lambda xs: sum(xs) / len(xs)  # noqa: E731
    res = {"axis_mean": {m: {a: mean(v) for a, v in ax.items()} for m, ax in scores.items()},
           "category_axis_mean": {m: {c: {a: mean(v) for a, v in ax.items()} for c, ax in cats.items()}
                                  for m, cats in by_cat.items()},
           "ab_preference": dict(pref), "n_items": len(ab)}
    (d / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    return res


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("eval_set", type=Path)
    m.add_argument("runs", nargs=2, help="이름=생성 폴더")
    m.add_argument("--out", type=Path, required=True)
    c = sub.add_parser("collect")
    c.add_argument("dir", type=Path)
    a = p.parse_args(argv)
    if a.cmd == "make":
        items = json.loads(a.eval_set.read_text(encoding="utf-8"))["items"]
        runs = {}
        for spec in a.runs:
            name, path = spec.split("=", 1)
            runs[name] = {json.loads(x)["id"]: json.loads(x)
                          for x in (Path(path) / "generations.jsonl").read_text(encoding="utf-8").splitlines()}
        make(items, runs, a.out)
        print(f"채점지: {a.out}/sheet.csv, ab.csv, sheet.html (키: key.json — 채점 끝날 때까지 열지 않기)")
    else:
        print(json.dumps(collect(a.dir), ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
