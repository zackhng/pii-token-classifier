"""v5 data: account format registry, account phrases, ADDRESS fragment merging."""
import random
import re

from address_merge import merge_address_spans
from common import IGNORE, load_yaml


def _spans(text, parts):
    """[(substring, label)] -> spans at the first occurrence after the previous one."""
    out, i = [], 0
    for sub, lab in parts:
        i = text.index(sub, i)
        out.append({"start": i, "end": i + len(sub), "label": lab})
        i += len(sub)
    return out


def _texts(text, spans, label="ADDRESS"):
    return [text[s["start"]:s["end"]] for s in spans if s["label"] == label]


# ---------------------------------------------------------------- ADDRESS fragments
def test_merge_fragments_with_joiners():
    t = "地址：389 号 临江大道 ，邮编 710000。电话 13800000000"
    sp = _spans(t, [("389", "ADDRESS"), ("临江大道", "ADDRESS"), ("710000", "ADDRESS"), ("13800000000", "PHONE")])
    new, st = merge_address_spans(t, sp)
    assert _texts(t, new) == ["389 号 临江大道 ，邮编 710000"] and st["merged"] == 2
    assert _texts(t, new, "PHONE") == ["13800000000"]
    t = "Số nhà 1925 trên đường Lê Duẩn."
    new, _ = merge_address_spans(t, _spans(t, [("1925", "ADDRESS"), ("Lê Duẩn", "ADDRESS")]))
    assert _texts(t, new) == ["1925 trên đường Lê Duẩn"]
    t = "대전로1331번길 273이며, 우편번호는 28291입니다"
    new, _ = merge_address_spans(t, _spans(t, [("대전로1331번길 273", "ADDRESS"), ("28291", "ADDRESS")]))
    assert _texts(t, new) == ["대전로1331번길 273이며, 우편번호는 28291"]


def test_no_merge_across_prose_entities_or_slash():
    t = "Ship to 12 Orchard Road. Contact Jane Tan at 9 Bukit Timah Road"
    sp = _spans(t, [("12 Orchard Road", "ADDRESS"), ("Jane Tan", "PERSON"), ("9 Bukit Timah Road", "ADDRESS")])
    new, st = merge_address_spans(t, sp)
    assert st["merged"] == 0 and len(_texts(t, new)) == 2
    t = "광주 북구 올림픽로 252 / 광주 북구 올림픽로 252"
    new, st = merge_address_spans(t, _spans(t, [("광주 북구 올림픽로 252", "ADDRESS"), ("광주 북구 올림픽로 252", "ADDRESS")]))
    assert st["merged"] == 0


def test_lone_postcode_is_loss_masked_not_o():
    t = "未央区 张家堡街道，邮编 510140 项目明细"
    new, st = merge_address_spans(t, _spans(t, [("510140", "ADDRESS")]))
    assert new[0]["label"] == IGNORE and st["postcode_ignored"] == 1
    t = "Unit 5, 10 Anson Road, Singapore 079903"         # a whole address that ends in a postcode
    new, _ = merge_address_spans(t, _spans(t, [("Unit 5, 10 Anson Road, Singapore 079903", "ADDRESS")]))
    assert new[0]["label"] == "ADDRESS"


# ---------------------------------------------------------------- account registry
def test_account_registry_is_sourced():
    reg = load_yaml("account_formats.yaml")
    for inst in reg["institutions"]:
        assert inst["status"] in {"verified", "partial", "unverified"}, inst["id"]
        assert set(inst["sources"]) <= set(reg["sources"]), inst["id"]
        if inst["status"] != "unverified":
            assert inst["sources"] and inst["patterns"], inst["id"]     # never a format without a source


def test_account_values_match_their_pattern():
    from ml_values import pattern_regex, registry, render_pattern
    rng = random.Random(0)
    for inst in registry():
        for pat in inst["patterns"]:
            rx = pattern_regex(pat)
            for _ in range(20):
                v = render_pattern(pat, rng)
                assert rx.match(v), (inst["id"], pat, v)


def test_account_phrases_have_exact_spans_and_no_held_out_in_train():
    from ml_values import HELD_OUT, MARKETS, account_phrase, institutions
    rng = random.Random(1)
    for lang in MARKETS:
        assert not {i["id"] for i in institutions(lang, "train")[0]} & HELD_OUT
        for _ in range(300):
            text, spans = account_phrase(lang, rng, n=rng.choice([1, 2, 3]), english_cue=rng.random() < 0.3)
            assert any(s["label"] == "ACCOUNT" for s in spans)
            for s in spans:
                v = text[s["start"]:s["end"]]
                assert v and v == v.strip(), (text, s)
            assert not re.search(r"# #|##(?!#)|: :", text.replace("###", "")), text


# ---------------------------------------------------------------- frames
def test_frames_fill_exact_spans():
    import frames
    rows = frames.render("train", per_frame=3, seed=7)
    assert rows
    for r in rows:
        assert r["lang"] in frames.LANGS and (r["cs"] is None or r["cs"] in frames.LANGS)
        last = 0
        for s in sorted(r["spans"], key=lambda s: s["start"]):
            v = r["text"][s["start"]:s["end"]]
            assert v and v == v.strip() and s["start"] >= last, (r["id"], v)
            last = s["end"]
        assert "{" not in r["text"] or "}" not in r["text"], r["id"]        # every slot was filled


def test_frame_slots_known_and_cs_is_english_plus_one():
    import frames
    for f in frames.load_frames("train") + frames.load_frames("dev") + frames.load_frames("test"):
        for m in frames.SLOT.finditer(f["template"]):
            assert m.group(1) in frames.ENTITY or m.group(1) in frames.LOOKALIKE, (f["id"], m.group(1))
            assert m.group(2) in (None, *frames.LANGS), (f["id"], m.group(2))
        if f["cs"]:
            assert "en" in (f["lang"], f["cs"]) and f["lang"] != f["cs"], f["id"]   # user rule: English + one


def test_train_dev_test_frames_share_no_structure():
    """ml_struct_dev (used for choices) and ml_struct_test (reported once) share no structure with
    training or with each other."""
    import frames
    sk = {sp: {frames.skeleton(f["template"]): f["id"] for f in frames.load_frames(sp)} for sp in ("train", "dev", "test")}
    for a, b in (("train", "dev"), ("train", "test"), ("dev", "test")):
        both = set(sk[a]) & set(sk[b])
        assert not both, (a, b, [sk[b][k] for k in both][:5])


def test_lookalike_dates_share_dob_formats():
    import frames
    rng = random.Random(3)
    for lang in frames.LANGS:
        dob = {re.sub(r"\d", "#", frames.date_value(lang, rng, 1940, 2005)) for _ in range(300)}
        date = {re.sub(r"\d", "#", frames.date_value(lang, rng, 2016, 2026)) for _ in range(300)}
        assert dob & date, lang                                       # only the context can tell them apart


def test_literal_letters_in_account_patterns():
    from ml_values import render_pattern
    rng = random.Random(0)
    assert all(render_pattern("[A][E]####", rng).startswith("AE") for _ in range(50))
    assert all(render_pattern("[X]##", rng).startswith("X") for _ in range(50))


# ---------------------------------------------------------------- ml_wm grid specs
def test_wm_specs_complete_and_split_evenly():
    import re
    import wm_gen
    from frames import LANGS
    for lang in LANGS:
        S = wm_gen.spec(lang)
        for key in ("field_seps", "and", "comma", "period", "speakers", "table_headers", "table_note", "cues",
                    "plural", "patterns", "ask", "narrative", "titles", "fillers"):
            assert key in S, (lang, key)
        for lab in wm_gen.LABELS + wm_gen.LOOK:
            assert len(S["cues"][lab]) % 3 == 0 and len(S["cues"][lab]) >= 3, (lang, lab)
        for lab in wm_gen.LABELS:
            assert len(S["plural"][lab]) % 3 == 0, (lang, "plural", lab)
            assert len(S["narrative"][lab]) >= 3 and all("{v}" in t for t in S["narrative"][lab]), (lang, lab)
        for ctx, pats in S["patterns"].items():
            assert len(pats) % 3 == 0, (lang, ctx)
            need = {"list": {"{c}", "{vs}"}, "bracket": {"{a}", "{v}"}}.get(ctx, {"{v}"})
            assert all(need <= set(re.findall(r"\{\w+\}", p)) for p in pats), (lang, ctx)
        assert len(S["ask"]) % 3 == 0 and all("{c}" in a for a in S["ask"]), lang
        for dt in wm_gen.DOC_TYPES:
            assert len(S["titles"][dt]) >= 3, (lang, dt)
            if dt != "call":
                assert dt in S["fillers"] or dt == "call", (lang, dt)
        for dt, fl in S["fillers"].items():
            for f in fl:
                assert set(re.findall(r"\{(\w+)\}", f)) <= set(wm_gen.LOOK), (lang, dt, f)


def test_wm_documents_have_exact_spans_every_language():
    import wm_gen
    from frames import LANGS
    g = wm_gen.Gen("dev", 3)
    for lang in LANGS:
        seen = set()
        for mode in wm_gen.MODES:
            if lang == "en" and mode != "mono":
                continue
            for fmt in wm_gen.FORMATS:
                ctxs = ["spoken"] if fmt == "transcript" else ["field", "list", "bracket"] if fmt == "md_table" else \
                    ["field", "before", "after", "narrative", "list", "bracket"]
                plan = [(lab, ctxs[k % len(ctxs)]) for k, lab in enumerate(wm_gen.LABELS)]
                d = g.document(plan, lang, mode, fmt, "call" if fmt == "transcript" else "kyc")
                for s in d["spans"] + d["negs"]:
                    v = d["text"][s["start"]:s["end"]]
                    assert v and v == v.strip(), (lang, mode, fmt, v)
                seen |= {s["label"] for s in d["spans"]}
        assert seen == set(wm_gen.LABELS), (lang, set(wm_gen.LABELS) - seen)
