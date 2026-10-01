"""Date/cue augmentation, the DATE_CUE rule and snippet records keep spans aligned."""
import random
from collections import Counter
from datetime import date

from augment import (DATE_FORMATS, SHORT_DATE_FORMATS, augment_doc, has_birth_cue, parse_date,
                     splice)
from build_dataset import clean_spans
from common import IGNORE
from snippets import make_records


def spans_of(text, subs):
    return [{"start": text.index(v), "end": text.index(v) + len(v), "label": lab}
            for v, lab in subs]


def test_parse_date_shapes():
    d = date(1987, 5, 22)
    for v in ["1987-05-22", "1987-05-22T00:00:00", "05/22/1987", "22/05/1987", "22.05.1987",
              "May 22, 1987", "May 22nd, 1987", "22nd May 1987", "22 May 1987", "22-May-1987",
              "1987/05/22"]:
        assert parse_date(v) == d, v
    for v in ["[Redacted]", "MM/DD/YYYY", "2023", "Q1 2023", "31/02/1990"]:
        assert parse_date(v) is None, v


def test_splice_shifts_later_spans():
    text = "ab XYZ cd EFG"
    spans = [{"start": 3, "end": 6, "label": "a"}, {"start": 10, "end": 13, "label": "b"}]
    out = splice(text, spans, 3, 6, "LONGER")
    assert [out[s["start"]:s["end"]] for s in spans] == ["LONGER", "EFG"]


def test_augment_keeps_alignment():
    text = ("Customer: Jane Roe\nDate of Birth: 1987-05-22\nStatement date 03/15/2024, "
            "account 12345678. My date of birth is 1961-05-25 and I was born on 2002-03-05.")
    raw = spans_of(text, [("Jane Roe", "first_name"), ("1987-05-22", "date_of_birth"),
                          ("03/15/2024", "date"), ("12345678", "account_number"),
                          ("1961-05-25", "date_of_birth"), ("2002-03-05", "date_of_birth")])
    originals = {s["label"]: text[s["start"]:s["end"]] for s in raw}
    rng = random.Random(0)
    changed = 0
    for _ in range(200):
        new, spans = augment_doc(text, raw, rng, p_date=1.0, p_cue=1.0)
        assert len(spans) == len(raw)
        for old, s in zip(raw, spans):
            val = new[s["start"]:s["end"]]
            if s["label"] in ("date_of_birth", "date"):
                d = parse_date(text[old["start"]:old["end"]])
                fmts = DATE_FORMATS + SHORT_DATE_FORMATS
                assert val in {f(d) for f in fmts} | {text[old["start"]:old["end"]]}, val
            else:
                assert val == originals[s["label"]]
        changed += new != text
    assert changed > 150


def test_date_cue_rule():
    text = "DOB: 10/02/1965. Payment received 03/04/2023.\nborn\n12/12/2000"
    raw = spans_of(text, [("10/02/1965", "DATE"), ("03/04/2023", "DATE"), ("12/12/2000", "DATE")])
    negs = []
    spans = clean_spans(text, raw, {"DATE": "DATE_CUE"}, Counter(), negs, {"DATE": "date"})
    assert [(text[s["start"]:s["end"]], s["label"]) for s in spans] == [("10/02/1965", "DOB")]
    # the cue on the previous line does not count; both ordinary dates become negatives
    assert [text[n["start"]:n["end"]] for n in negs] == ["03/04/2023", "12/12/2000"]


def test_birth_cue_phrasings():
    for pre in ["DOB: ", "D.O.B. ", "Date of Birth (DD/MM/YYYY): ", "my date of birth is ",
                "born on ", "Born in Springfield on ", "Birthdate: "]:
        assert has_birth_cue(pre + "x", len(pre)), pre
    for pre in ["Issue Date: ", "DOB: 1/2/65. Paid ", "Name | DOB | Txn Date\nJo | ",
                "stubborn delay until "]:
        assert not has_birth_cue(pre + "x", len(pre)), pre


def test_dob_placeholder_ignored():
    text = "Date of Birth: [Redacted]"
    raw = spans_of(text, [("[Redacted]", "date_of_birth")])
    spans = clean_spans(text, raw, {"date_of_birth": "DOB"}, Counter())
    assert spans[0]["label"] == IGNORE


def test_snippet_spans_valid():
    rows = [{"text": "Jane Roe of Acme Bank, account 0012-3456-789, jane@x.com",
             "spans": spans_of("Jane Roe of Acme Bank, account 0012-3456-789, jane@x.com",
                               [("Jane Roe", "PERSON"), ("Acme Bank", "BUSINESS"),
                                ("0012-3456-789", "ACCOUNT"), ("jane@x.com", "EMAIL")])}]
    for style in ("train", "stress"):
        for r in make_records(rows, 300, 0, style, "t"):
            for s in r["spans"] + r["negs"]:
                assert 0 <= s["start"] < s["end"] <= len(r["text"])
                assert r["text"][s["start"]:s["end"]].strip() == r["text"][s["start"]:s["end"]]
            assert any(s["label"] for s in r["spans"])

