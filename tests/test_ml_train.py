"""Multilingual training data: source readers, partial labels, source rules."""
from common import label_maps, load_yaml
from ml_sources import detokenize, strip_korean_title
from tokenize_bio import IGNORE_ID, token_labels

_, L2I, _ = label_maps(load_yaml("label_map.yaml")["entities"])


def test_detokenize_spaced_and_unspaced():
    text, spans = detokenize(["Sinabi", "ni", "Juan", "Dela", "Cruz", "."], ["O", "O", "B-PER", "I-PER", "I-PER", "O"])
    assert text == "Sinabi ni Juan Dela Cruz."
    assert [(text[s["start"]:s["end"]], s["label"]) for s in spans] == [("Juan Dela Cruz", "PER")]
    # ThaiNER: spaces are tokens, joined without a separator; a space inside a name stays in it
    text, spans = detokenize(["ทักษิณ", " ", "ชินวัตร", " ", "ทวีต"], ["B-PERSON", "I-PERSON", "I-PERSON", "O", "O"], "")
    assert text == "ทักษิณ ชินวัตร ทวีต" and text[spans[0]["start"]:spans[0]["end"]] == "ทักษิณ ชินวัตร"


def test_detokenize_adjacent_entities_and_brackets():
    text, spans = detokenize(["(", "MILF", ")", "Juan", "Pedro"], ["O", "B-ORG", "O", "B-PER", "B-PER"])
    assert text == "(MILF) Juan Pedro"
    assert [text[s["start"]:s["end"]] for s in spans] == ["MILF", "Juan", "Pedro"]


def test_strip_korean_title():
    assert strip_korean_title("박호 어머님") == "박호"
    assert strip_korean_title("이ㅁㅁ 대리") == "이ㅁㅁ"
    assert strip_korean_title("김준준님") == "김준준"
    assert strip_korean_title("이성은") == "이성은"


def _labels(text, pieces, spans, partial):
    offs, i = [], 0
    for p in pieces:
        i = text.index(p, i)
        offs.append((i, i + len(p)))
        i += len(p)
    return token_labels(text, offs, spans, L2I, partial=partial, label_all_tokens=True)


def test_partial_digits_masks_unlabelled_numbers():
    text = "Ali lahir 12 Mei 1990"
    spans = [{"start": 0, "end": 3, "label": "PERSON"}]
    full = _labels(text, ["Ali", " lahir", " 12", " Mei", " 1990"], spans, None)
    part = _labels(text, ["Ali", " lahir", " 12", " Mei", " 1990"], spans, "digits")
    assert full[2] == L2I["O"] and part[2] == IGNORE_ID and part[4] == IGNORE_ID
    assert part[1] == L2I["O"] and part[0] == L2I["B-PERSON"]


def test_partial_all_keeps_only_spans_and_explicit_o():
    text = "الاسم سارة الهاتف 0501234567 الرمز 4821"
    spans = [{"start": 18, "end": 28, "label": "PHONE"}, {"start": 35, "end": 39, "label": "O"}]
    labs = _labels(text, ["الاسم", " سارة", " الهاتف", " 0501234567", " الرمز", " 4821"], spans, "all")
    assert labs[0] == labs[1] == labs[2] == labs[4] == IGNORE_ID      # unlabelled name not taught as O
    assert labs[3] == L2I["B-PHONE"] and labs[5] == L2I["O"]           # OTP code is a known non-PII


def test_no_medical_sources():
    srcs = load_yaml("ml_sources.yaml")["sources"]
    for name, cfg in srcs.items():
        hub = str(cfg.get("hub", "")).lower()
        assert "meddies" not in hub and "phoner" not in hub and "covid" not in hub, name
    from ml_sources import SITR_MEDICAL
    assert {"medical_record_number", "medicine_name"} <= SITR_MEDICAL


def test_every_source_maps_to_known_labels():
    ents = set(load_yaml("label_map.yaml")["entities"]) | {"IGNORE", "DATE_CUE"}
    for name, cfg in load_yaml("ml_sources.yaml")["sources"].items():
        assert set(cfg.get("map", {}).values()) <= ents, name
        assert cfg["kind"] in {"financial", "pii", "ner", "generated"}, name
        assert cfg.get("partial", "") in {"", "digits", "all"}, name


def test_generated_training_docs_disjoint_from_test_templates():
    """Training generators (ml_train_docs) share no prose template, title or name / company value
    with the test-only ml_templates pools."""
    import random
    import ml_templates
    import ml_train_docs
    test_langs = ml_templates.languages()
    for lang, train_L in ml_train_docs.LANGS.items():
        test_L = test_langs[lang]
        assert not set(train_L["prose"]) & set(test_L["prose"]), lang
        for k in ("kyc", "statement", "job"):
            assert not set(train_L["titles"][k]) & set(test_L["titles"][k]), (lang, k)
        for ent in ("PERSON", "BUSINESS", "ADDRESS"):
            r1, r2 = random.Random(1), random.Random(2)
            a = {str(train_L["gen"][ent](r1)) for _ in range(3000)}
            b = {str(test_L["gen"][ent](r2)) for _ in range(3000)}
            assert not a & b, (lang, ent, sorted(a & b)[:3])


def test_generated_docs_cover_all_entities():
    from ml_train_docs import generate
    ents = set(load_yaml("label_map.yaml")["entities"])
    for lang in ("th", "ar", "hi"):
        rows = generate(lang, 300, seed=5)
        seen = {s["label"] for r in rows for s in r["spans"]}
        assert ents <= seen, (lang, ents - seen)
        for r in rows:
            for s in r["spans"]:
                assert r["text"][s["start"]:s["end"]].strip() == r["text"][s["start"]:s["end"]]
