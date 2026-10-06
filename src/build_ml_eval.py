"""Multilingual zero-shot test sets for the English-only model (evaluation only, never trained on).

  ml_real.jsonl   AI4Privacy validation docs (OpenPII-1.5M: zh ja ko vi ms id tl en;
                  open-pii-masking-500k: hi), zh-Hant = zh converted with OpenCC, same labels and
                  label map as the English training data
  ml_synth.jsonl  native-language wealth-management documents from src/ml_templates.py
                  (KYC forms, statement headers, RM e-mails, transfer instructions, signatures)

Rows: {id, source, lang, text, spans: [{start, end, label, script}], negs: [{start, end, label}]}.
`script` is "nonlatin" when the span contains a non-Latin letter or digit (CJK, kana, hangul,
Devanagari, Arabic, Thai, Arabic-Indic digits ...), else "latin" (romanised names, e-mails, ASCII
numbers). `negs` are look-alikes (amount / date / code) that must stay O.

    python build_ml_eval.py [--n_real 1000] [--n_synth 300]
"""
import argparse
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict

from datasets import load_from_disk

from augment import has_birth_cue
from build_dataset import normalize, write_jsonl
from common import IGNORE, ROOT, load_yaml
from ml_templates import ENTITY_PH, NEG_PH, languages

REAL_LANGS = ["en", "zh", "ja", "ko", "vi", "ms", "id", "tl"]   # in OpenPII-1.5M validation
HINDI_500K = ROOT / "data/raw/ai4privacy500k/validation.jsonl"

# Birth cues of the target languages: before the date ("出生日期：", "생년월일", "जन्म तिथि",
# "ngày sinh", "tarikh lahir", "تاريخ الميلاد", "เกิดวันที่") or right after it ("…日生まれ", "…년생").
ML_BIRTH_BEFORE = re.compile(
    r"(出生|生日|生于|生於|生年月日|誕生日|생년월일|생일|출생|जन्म|ngày sinh|sinh ngày|sinh năm|lahir|"
    r"kapanganakan|ipinanganak|kaarawan|الميلاد|ولد|مواليد|เกิด)[^.;|\t\n\d。]{0,25}$", re.I)
ML_BIRTH_AFTER = re.compile(r"^\s*(生まれ|出生|生(?!效|产|產|成)|에 태어|년생|생(?![가-힣]))")


def ml_birth_cue(text: str, start: int, end: int) -> bool:
    if has_birth_cue(text, start):
        return True
    before = text[max(0, start - 40):start].rsplit("\n", 1)[-1]
    return bool(ML_BIRTH_BEFORE.search(before) or ML_BIRTH_AFTER.match(text[end:end + 8]))


def script_class(s: str) -> str:
    for c in s:
        if c.isdigit() and not c.isascii():
            return "nonlatin"
        if c.isalpha() and not unicodedata.name(c, "").startswith("LATIN"):
            return "nonlatin"
    return "latin"


def tag_scripts(row: dict) -> dict:
    for s in row["spans"]:
        s["script"] = script_class(row["text"][s["start"]:s["end"]])
    return row


# ---------------------------------------------------------------- real (AI4Privacy) set
def real_set(n: int, seed: int) -> list[dict]:
    label_cfg = load_yaml("label_map.yaml")
    mapping = label_cfg["ai4privacy"]
    neg_groups = {lab: g for g, labs in label_cfg["negatives"].items() for lab in labs}
    val = load_from_disk(str(ROOT / "data/raw/ai4privacy"))["validation"]
    rows, stats = [], Counter()

    def take(lang_code, split, source_lang):
        out = normalize("ai4privacy", split, mapping, stats, neg_groups=neg_groups, keep_negs=True,
                        lang_ok=lambda _s, r: r["language"] == source_lang, birth_cue=ml_birth_cue)
        out = [r for r in out if any(s["label"] != IGNORE for s in r["spans"])][:n]
        for r in out:
            r.update(source=f"ai4p_{lang_code}", lang=lang_code, id=f"ai4p_{lang_code}-{r['id'].split('-', 1)[1]}")
        return out

    for lang in REAL_LANGS:
        sub = val.filter(lambda b: [l == lang for l in b["language"]], batched=True).shuffle(seed=seed)
        code = "zh-Hans" if lang == "zh" else lang
        got = take(code, sub.select(range(min(len(sub), 2 * n))), lang)
        rows += got
        print(f"real {code}: {len(got)} docs")
        if lang == "zh":
            rows += zh_hant_rows(got)

    if HINDI_500K.exists():
        hi = [json.loads(l) for l in open(HINDI_500K, encoding="utf-8")]
        hi = [r for r in hi if r.get("language") == "hi"]
        random.Random(seed).shuffle(hi)
        got = take("hi", hi[:2 * n], "hi")
        rows += got
        print(f"real hi: {len(got)} docs")
    else:
        print(f"[warn] {HINDI_500K} missing - Hindi real set skipped")
    print(f"real cleanup: {dict(stats)}")
    return [tag_scripts(r) for r in rows]


def zh_hant_rows(rows: list[dict]) -> list[dict]:
    """Same docs in Traditional characters. OpenCC s2t is phrase-aware; a doc is kept only if
    its length is unchanged, so every span offset stays valid."""
    import opencc
    cc = opencc.OpenCC("s2t")
    out, dropped = [], 0
    for r in rows:
        t = cc.convert(r["text"])
        if len(t) != len(r["text"]):
            dropped += 1
            continue
        out.append({**r, "id": r["id"].replace("zh-Hans", "zh-Hant"), "source": "ai4p_zh-Hant",
                    "lang": "zh-Hant", "text": t, "spans": [dict(s) for s in r["spans"]],
                    "negs": [dict(s) for s in r.get("negs", [])]})
    print(f"real zh-Hant: {len(out)} docs ({dropped} dropped: length changed)")
    return out


# ---------------------------------------------------------------- synthetic set
ADDR_JOIN = {"zh-Hans": "", "zh-Hant": "", "ja": " ", "ko": " ", "th": " ", "ar": "، "}


class Doc:
    def __init__(self):
        self.parts, self.spans, self.negs, self.n = [], [], [], 0

    def add(self, s: str, label: str | None = None, neg: str | None = None):
        if label:
            self.spans.append({"start": self.n, "end": self.n + len(s), "label": label})
        if neg:
            self.negs.append({"start": self.n, "end": self.n + len(s), "label": neg})
        self.parts.append(s)
        self.n += len(s)

    @property
    def text(self):
        return "".join(self.parts)


def join_address(lang: str, lines: list[str]) -> str:
    sep = ADDR_JOIN.get(lang, ", ")
    out = lines[0]
    for ln in lines[1:]:
        out += (" " if out.endswith((",", "،")) else sep) + ln
    return out


def value(L, lang, ph, rng, multiline=False):
    g = L["gen"]
    key = "PERSON" if ph == "PERSON2" else ph
    v = g[key](rng)
    if key == "ADDRESS":
        v = "\n".join(v) if multiline else join_address(lang, v)
    return v


def render(doc: Doc, L, lang, template: str, rng):
    pos = 0
    for m in re.finditer(r"\{(\w+)\}", template):
        doc.add(template[pos:m.start()])
        ph = m.group(1)
        doc.add(value(L, lang, ph, rng), label=ENTITY_PH.get(ph), neg=NEG_PH.get(ph))
        pos = m.end()
    doc.add(template[pos:])


def field(doc: Doc, L, lang, key: str, rng, sep: str):
    """One form line: cue + separator + value."""
    is_neg = key in ("amount", "date", "code")
    doc.add(rng.choice(L["cues"][key]) + sep)
    ph = {"amount": "AMOUNT", "date": "DATE", "code": "CODE"}.get(key, key)
    doc.add(value(L, lang, ph, rng), label=None if is_neg else key, neg=key if is_neg else None)
    doc.add("\n")


def kyc_form(doc, L, lang, rng):
    sep = rng.choice(L["seps"])
    doc.add(rng.choice(L["titles"]["kyc"]) + "\n" + rng.choice(["", "\n"]))
    keys = rng.sample(["DOB", "ACCOUNT", "PHONE", "EMAIL", "TIN", "ADDRESS", "BUSINESS"], rng.randint(3, 6))
    keys += rng.sample(["amount", "date", "code"], rng.randint(1, 2))
    rng.shuffle(keys)
    for k in ["PERSON"] + keys:
        field(doc, L, lang, k, rng, sep)


def statement(doc, L, lang, rng):
    sep = rng.choice(L["seps"])
    doc.add(value(L, lang, "BUSINESS", rng), label="BUSINESS")
    doc.add("\n" + rng.choice(L["titles"]["statement"]) + "\n\n")
    doc.add(value(L, lang, "PERSON", rng), label="PERSON")
    doc.add("\n")
    doc.add(value(L, lang, "ADDRESS", rng, multiline=True), label="ADDRESS")
    doc.add("\n\n")
    for k in ["ACCOUNT", "date", "amount"]:
        field(doc, L, lang, k, rng, sep)


def signature(doc, L, lang, rng):
    doc.add(rng.choice(L["titles"]["closing"]) + "\n")
    doc.add(value(L, lang, "PERSON", rng), label="PERSON")
    doc.add("\n" + rng.choice(L["titles"]["job"]) + "\n")
    doc.add(value(L, lang, "BUSINESS", rng), label="BUSINESS")
    doc.add("\n")
    sep = rng.choice(L["seps"][:2])
    for k in ["PHONE", "EMAIL"]:
        if rng.random() < 0.5:
            doc.add(value(L, lang, k, rng), label=k)
            doc.add("\n")
        else:
            field(doc, L, lang, k, rng, sep)


def prose(doc, L, lang, rng):
    for i, t in enumerate(rng.sample(L["prose"], rng.randint(1, 2))):
        if i:
            doc.add("\n\n" if rng.random() < 0.5 else " ")
        render(doc, L, lang, t, rng)
    if rng.random() < 0.4:
        doc.add("\n\n")
        signature(doc, L, lang, rng)


DOC_TYPES = [(kyc_form, 0.3), (statement, 0.2), (prose, 0.5)]


def synth_set(n: int, seed: int) -> list[dict]:
    rows = []
    for lang, L in languages().items():
        rng = random.Random(f"{seed}-{lang}")
        for i in range(n):
            doc = Doc()
            fn = rng.choices([f for f, _ in DOC_TYPES], [w for _, w in DOC_TYPES])[0]
            fn(doc, L, lang, rng)
            rows.append(tag_scripts({"id": f"synth_{lang}-{i}", "source": f"synth_{lang}", "lang": lang,
                                     "doc_type": fn.__name__, "text": doc.text,
                                     "spans": doc.spans, "negs": doc.negs}))
    return rows


# ---------------------------------------------------------------- checks / report
def check(rows: list[dict], min_per_entity: int = 0):
    per_lang = defaultdict(Counter)
    for r in rows:
        t = r["text"]
        prev_end = -1
        for s in sorted(r["spans"], key=lambda s: s["start"]):
            assert 0 <= s["start"] < s["end"] <= len(t), (r["id"], s)
            assert t[s["start"]:s["end"]].strip() == t[s["start"]:s["end"]], (r["id"], s, t[s["start"]:s["end"]])
            assert s["start"] >= prev_end, (r["id"], "overlapping spans")
            prev_end = s["end"]
            per_lang[r["lang"]][s["label"]] += 1
            per_lang[r["lang"]][f"script:{s['script']}"] += 1
        for g in r.get("negs", []):
            assert 0 <= g["start"] < g["end"] <= len(t), (r["id"], g)
    for lang, c in per_lang.items():
        ents = {k: v for k, v in c.items() if not k.startswith("script:")}
        print(f"  {lang:8s} {dict(sorted(ents.items()))}  "
              f"latin={c['script:latin']} nonlatin={c['script:nonlatin']}")
        if min_per_entity:
            low = {k: v for k, v in ents.items() if v < min_per_entity and k != IGNORE}
            assert not low, (lang, low)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_real", type=int, default=1000)
    ap.add_argument("--n_synth", type=int, default=300)
    ap.add_argument("--skip_real", action="store_true")
    args = ap.parse_args()
    seed = load_yaml("train.yaml")["seed"]
    out_dir = ROOT / load_yaml("train.yaml")["data"]["processed_dir"]

    synth = synth_set(args.n_synth, seed)
    print(f"\nml_synth: {len(synth)} docs")
    check(synth, min_per_entity=30)
    write_jsonl(out_dir / "ml_synth.jsonl", synth)

    if not args.skip_real:
        real = real_set(args.n_real, seed)
        print(f"\nml_real: {len(real)} docs")
        check(real)
        write_jsonl(out_dir / "ml_real.jsonl", real)


if __name__ == "__main__":
    main()
