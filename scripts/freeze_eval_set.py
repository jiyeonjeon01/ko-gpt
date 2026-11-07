"""평가 세트 초안을 검사한 뒤 고정본으로 저장한다 (한 번만)

    python3 scripts/freeze_eval_set.py eval/chat120/draft_v0.1.json [--version 1]

- 형식 검사: 문항 수, 범주별 개수, id 중복, 메시지 역할 순서(마지막은 user), reference 종류
- 고정본 eval/chat120/frozen_v{N}.json: 문항·채점 기준(rubric.md) 내용과 각 SHA-256, 고정 시각, 원본 초안 기록
- 같은 버전의 고정본이 있으면 멈춘다. 고친 내용은 새 버전으로 고정한다 (README 규칙)
"""

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

CATEGORIES = ["인사·잡담", "감정·공감", "일상 조언", "상식 사실", "설명 요청", "간단한 산수·추론", "코드", "영어"]
PER_CATEGORY = 15
REF_KINDS = {"none", "exact", "key_points"}


def validate(doc: dict) -> list[str]:
    errors = []
    items = doc.get("items", [])
    if len(items) != len(CATEGORIES) * PER_CATEGORY:
        errors.append(f"문항 수 {len(items)} ≠ {len(CATEGORIES) * PER_CATEGORY}")
    cats = Counter(it.get("category") for it in items)
    for c in CATEGORIES:
        if cats.get(c, 0) != PER_CATEGORY:
            errors.append(f"범주 {c}: {cats.get(c, 0)}개")
    extra = set(cats) - set(CATEGORIES)
    if extra:
        errors.append(f"알 수 없는 범주 {sorted(extra)}")
    ids = Counter(it.get("id") for it in items)
    dup = [i for i, n in ids.items() if n > 1]
    if dup:
        errors.append(f"id 중복 {dup}")
    for it in items:
        msgs = it.get("messages") or []
        roles = [m.get("role") for m in msgs]
        if not msgs or roles[-1] != "user" or any(r != ("user" if k % 2 == 0 else "assistant")
                                                   for k, r in enumerate(roles)):
            errors.append(f"{it.get('id')}: 메시지 역할 순서 {roles}")
        if any(not str(m.get("content", "")).strip() for m in msgs):
            errors.append(f"{it.get('id')}: 빈 메시지")
        ref = it.get("reference", {})
        if ref.get("kind") not in REF_KINDS:
            errors.append(f"{it.get('id')}: reference.kind {ref.get('kind')}")
        if ref.get("kind") == "exact" and not ref.get("accept"):
            errors.append(f"{it.get('id')}: exact인데 accept가 없음")
        if ref.get("kind") == "key_points" and not ref.get("points"):
            errors.append(f"{it.get('id')}: key_points인데 points가 없음")
        if it.get("expected_language") not in ("ko", "en"):
            errors.append(f"{it.get('id')}: expected_language {it.get('expected_language')}")
    return errors


def canonical(obj) -> bytes:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def freeze(draft: Path, version: int, rubric: Path) -> Path:
    doc = json.loads(draft.read_text(encoding="utf-8"))
    errors = validate(doc)
    if errors:
        raise ValueError("고정할 수 없다:\n  " + "\n  ".join(errors))
    out = draft.parent / f"frozen_v{version}.json"
    if out.exists():
        raise FileExistsError(f"{out}가 이미 있다. 고친 내용은 --version {version + 1}로 고정한다")
    rubric_text = rubric.read_text(encoding="utf-8")
    frozen = {
        "name": doc["name"], "version": str(version), "status": "frozen",
        "frozen_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_draft": draft.name, "source_draft_sha256": hashlib.sha256(draft.read_bytes()).hexdigest(),
        "items_sha256": hashlib.sha256(canonical(doc["items"])).hexdigest(),
        "rubric": rubric_text, "rubric_sha256": hashlib.sha256(rubric_text.encode("utf-8")).hexdigest(),
        "rule": "고정 뒤에는 items·reference·safety·rubric을 고치지 않는다. 고칠 일이 있으면 새 버전으로 고정한다.",
        "items": doc["items"],
    }
    out.write_text(json.dumps(frozen, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


def verify(frozen_path: Path) -> bool:
    """고정본이 고정 이후 바뀌지 않았는지 (items·rubric 해시 대조)"""
    d = json.loads(frozen_path.read_text(encoding="utf-8"))
    return (hashlib.sha256(canonical(d["items"])).hexdigest() == d["items_sha256"]
            and hashlib.sha256(d["rubric"].encode("utf-8")).hexdigest() == d["rubric_sha256"])


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("draft", type=Path)
    p.add_argument("--version", type=int, default=1)
    p.add_argument("--rubric", type=Path)
    a = p.parse_args(argv)
    rubric = a.rubric or a.draft.parent / "rubric.md"
    out = freeze(a.draft, a.version, rubric)
    print(f"고정: {out} (items_sha256 {json.loads(out.read_text(encoding='utf-8'))['items_sha256'][:16]}…)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
