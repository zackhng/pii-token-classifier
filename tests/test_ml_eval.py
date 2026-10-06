"""Multilingual zero-shot test sets (build_ml_eval.py) and the decoding behaviour they measure."""
import re

import pytest

from build_ml_eval import check, ml_birth_cue, script_class, synth_set, zh_hant_rows
from tokenize_bio import word_starts


def test_synth_spans_are_the_inserted_values():
    rows = synth_set(40, seed=7)
    check(rows)                       # bounds, no surrounding whitespace, no overlaps
    langs = {r["lang"] for r in rows}
    assert {"en", "zh-Hans", "zh-Hant", "ja", "ko", "hi", "ar", "th", "vi", "ms", "id", "tl"} <= langs
    for r in rows:
        for s in r["spans"] + r["negs"]:
            assert "{" not in r["text"][s["start"]:s["end"]]
        assert not re.search(r"\{[A-Z0-9]+\}", r["text"]), r["text"]   # no unrendered placeholder


def test_synth_is_deterministic():
    a, b = synth_set(5, seed=3), synth_set(5, seed=3)
    assert [r["text"] for r in a] == [r["text"] for r in b]


def test_negatives_never_overlap_gold():
    for r in synth_set(30, seed=11):
        for n in r["negs"]:
            assert not any(n["start"] < g["end"] and g["start"] < n["end"] for g in r["spans"]), r["id"]


def test_zh_hant_conversion_keeps_offsets():
    text = "客户张伟的头发很好，出生日期：1985年3月30日，地址：上海市浦东新区世纪大道100号。"
    spans = [{"start": 2, "end": 4, "label": "PERSON"},
             {"start": 15, "end": 25, "label": "DOB"}]
    row = {"id": "ai4p_zh-Hans-1", "source": "ai4p_zh-Hans", "lang": "zh-Hans", "text": text,
           "spans": spans, "negs": []}
    [out] = zh_hant_rows([row])
    assert out["lang"] == "zh-Hant" and len(out["text"]) == len(text)
    assert out["text"][2:4] == "張偉" and out["text"][15:25] == "1985年3月30日"
    assert "頭髮" in out["text"]


@pytest.mark.parametrize("text,date,expected", [
    ("出生日期：1985年3月30日", "1985年3月30日", True),
    ("생년월일 1985년 3월 30일", "1985년 3월 30일", True),
    ("1985年3月30日生まれ", "1985年3月30日", True),
    ("जन्म तिथि: 15/08/1985", "15/08/1985", True),
    ("تاريخ الميلاد: 15/08/1985", "15/08/1985", True),
    ("เกิดวันที่ 30 มีนาคม 2528", "30 มีนาคม 2528", True),
    ("Tanggal Lahir: 30-03-1985", "30-03-1985", True),
    ("本协议自2024年1月1日生效", "2024年1月1日", False),
    ("申请日期：2024年1月1日", "2024年1月1日", False),
])
def test_ml_birth_cue(text, date, expected):
    start = text.index(date)
    assert ml_birth_cue(text, start, start + len(date)) is expected


def test_script_class():
    assert script_class("张伟") == "nonlatin"
    assert script_class("Nguyễn Văn An") == "latin"
    assert script_class("wang.wei@163.com") == "latin"
    assert script_class("١٥/٠٨/١٩٨٥") == "nonlatin"
    assert script_class("+91 98765 43210") == "latin"


def test_word_starts_treats_unspaced_cjk_and_thai_as_one_word():
    """Pins the decoding artifact eval_multilingual.py measures: word boundaries come only from
    spaces / punctuation, so a CJK or Thai run is one word and default decoding labels it from its
    first sub-token. Changing this changes training (OpenMed first-sub-token labelling)."""
    def n_words(text, offsets):
        return sum(word_starts(text, offsets))

    zh = "张伟任职于"
    assert n_words(zh, [(i, i + 1) for i in range(len(zh))]) == 1
    th = "สมชายใจดี"
    assert n_words(th, [(0, 5), (5, 9)]) == 1
    en = "John Tan"
    assert n_words(en, [(0, 4), (4, 8)]) == 2
