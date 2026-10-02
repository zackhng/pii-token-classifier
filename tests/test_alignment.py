"""Round-trip checks: char spans -> BIO token labels -> char spans."""
import json

import pytest
from transformers import AutoTokenizer

from build_dataset import clean_spans
from common import IGNORE, ROOT, label_maps, load_yaml
from tokenize_bio import (IGNORE_ID, fill_continuations, labels_to_spans, make_tokenize_fn,
                          window_encode, word_starts)

CFG = load_yaml("train.yaml")
LABELS, LABEL2ID, ID2LABEL = label_maps(load_yaml("label_map.yaml")["entities"])


@pytest.fixture(scope="module")
def tok():
    return AutoTokenizer.from_pretrained(CFG["model_name"])


def roundtrip(tok, text, spans, max_length=384, stride=96, label_all_tokens=False):
    fn = make_tokenize_fn(tok, LABEL2ID, max_length, stride, label_all_tokens)
    out = fn({"text": [text], "spans": [spans]})
    wins = window_encode(tok, text, max_length, stride)
    recovered = []
    for win, labs in zip(wins, out["labels"]):
        offs = win["offset_mapping"]
        names = [None if l == IGNORE_ID else ID2LABEL[l] for l in labs]
        if not label_all_tokens:
            names = fill_continuations(names, word_starts(text, offs))
        recovered += labels_to_spans(offs, names, text)
    return out, {(s["start"], s["end"], s["label"]) for s in recovered}


def test_simple_roundtrip(tok):
    text = "Contact Jane Doe at jane.doe@example.com or +1 (555) 123-4567. SSN 123-45-6789."
    spans = []
    for sub, lab in [("Jane Doe", "PERSON"), ("jane.doe@example.com", "EMAIL"),
                     ("+1 (555) 123-4567", "PHONE"), ("123-45-6789", "TIN")]:
        i = text.index(sub)
        spans.append({"start": i, "end": i + len(sub), "label": lab})
    for all_tok in (False, True):
        _, rec = roundtrip(tok, text, spans, label_all_tokens=all_tok)
        assert rec == {(s["start"], s["end"], s["label"]) for s in spans}


def test_ignore_spans_masked(tok):
    text = "Meeting on 22/07/1979 with Bob."
    i = text.index("22/07/1979")
    spans = [{"start": i, "end": i + 10, "label": IGNORE}]
    out, rec = roundtrip(tok, text, spans, label_all_tokens=True)
    assert rec == set()
    assert IGNORE_ID in out["labels"][0][1:-1]


def test_merge_adjacent():
    text = "John Smith lives at 12 Main St,\nSpringfield, IL 62701"
    raw = [
        {"start": 0, "end": 4, "label": "first_name"},
        {"start": 5, "end": 10, "label": "last_name"},
        {"start": 20, "end": 30, "label": "street_address"},
        {"start": 32, "end": 43, "label": "city"},
        {"start": 45, "end": 47, "label": "state"},
        {"start": 48, "end": 53, "label": "postcode"},
    ]
    from collections import Counter
    spans = clean_spans(text, raw, load_yaml("label_map.yaml")["nemotron"], Counter())
    assert [(text[s["start"]:s["end"]], s["label"]) for s in spans] == [
        ("John Smith", "PERSON"), ("12 Main St,\nSpringfield, IL 62701", "ADDRESS")]


def test_windows_cover_whole_text(tok):
    text = " ".join(f"word{i}" for i in range(3000))
    wins = window_encode(tok, text, 64, 16)
    assert all(len(w["input_ids"]) <= 64 for w in wins)
    assert wins[-1]["offset_mapping"][-2][1] == len(text)


def test_windows_start_with_B(tok):
    text = ("word " * 200) + "Alexander Hamilton " * 100
    spans, pos = [], 0
    while True:
        i = text.find("Alexander Hamilton", pos)
        if i < 0:
            break
        spans.append({"start": i, "end": i + 18, "label": "PERSON"})
        pos = i + 1
    out, _ = roundtrip(tok, text, spans, max_length=64, stride=16)
    for labs in out["labels"]:
        first = next((l for l in labs if l != IGNORE_ID), None)
        assert first is None or not ID2LABEL[first].startswith("I-")


@pytest.mark.skipif(not (ROOT / "data/processed/train.jsonl").exists(), reason="no processed data")
def test_processed_roundtrip(tok):
    rows = []
    with open(ROOT / "data/processed/train.jsonl", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
            if len(rows) >= 300:
                break
    total = hit = 0
    for r in rows:
        gold = {(s["start"], s["end"], s["label"]) for s in r["spans"] if s["label"] != IGNORE}
        _, rec = roundtrip(tok, r["text"], r["spans"])
        total += len(gold)
        hit += len(gold & rec)
    assert total == 0 or hit / total > 0.97, f"recovered {hit}/{total}"


def test_multiline_span_stays_one_entity(tok):
    text = "Address:\n12 Main Street\nLondon SW1A 2AA\nTel: 020 7946 0000"
    a = text.index("12 Main")
    b = text.index("2AA") + 3
    spans = [{"start": a, "end": b, "label": "ADDRESS"}]
    out, rec = roundtrip(tok, text, spans)
    assert rec == {(a, b, "ADDRESS")}
    names = [ID2LABEL[l] for l in out["labels"][0] if l != IGNORE_ID]
    assert names.count("B-ADDRESS") == 1
