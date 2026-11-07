"""dolly-ko 품질 검사 함수"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import dolly_ko_check as dk


def test_classify():
    assert dk.classify("안녕하세요. 반갑습니다.") == "ko"
    assert dk.classify("This is not translated.") == "en"
    assert dk.classify("Python의 list comprehension은 for 문을 짧게 쓰는 방법") == "ko"
    assert dk.classify("The 수도 is Seoul and Busan is big") == "mixed"
    assert dk.classify("  ") == "empty"


def test_check_counts_duplicates_and_languages():
    rows = [{"instruction": "질문 하나", "context": "", "response": "답입니다", "category": "open_qa"},
            {"instruction": "질문 하나", "context": "", "response": "답입니다", "category": "open_qa"},
            {"instruction": "질문 하나", "context": "", "response": "다른 답이에요", "category": "open_qa"},
            {"instruction": "번역 안 됨", "context": "", "response": "Not translated at all", "category": "brainstorming"}]
    r = dk.check(rows)
    assert r["rows"] == 4 and r["exact_duplicate_pairs"] == 1 and r["duplicate_instructions"] == 2
    assert r["response_lang"] == {"ko": 3, "en": 1} and r["untranslated_examples"] == ["번역 안 됨"]
    assert r["context_lang"] == {"empty": 4}
