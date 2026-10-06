"""Readers for the multilingual training sources (configs/ml_sources.yaml) -> raw rows

    {"id", "source", "lang", "text", "spans": [{"start", "end", "label"}]}

with the source's own labels; build_ml_train.py maps them to the 8 target labels. Token-tagged
corpora (CoNLL style) are detokenised here: tokens joined by a space (none for zh / ja / th / the
character-level KLUE tokens), no space before closing punctuation or after opening punctuation.
"""
import json
import random
import re
from pathlib import Path

from datasets import load_from_disk

from common import ROOT

RAW = ROOT / "data/raw"
NO_SPACE_BEFORE = set(".,!?;:%)]}」』）、。，！？；：")
NO_SPACE_AFTER = set("([{「『（")


def detokenize(tokens: list[str], tags: list[str], joiner: str = " ") -> tuple[str, list[dict]]:
    """Tokens + BIO tag strings ("B-PER", "I-PER", "O") -> (text, char spans with raw types)."""
    parts, spans, cur, n = [], [], None, 0
    prev = None
    for tok, tag in zip(tokens, tags):
        if tok == "":
            continue
        sep = joiner if (prev is not None and joiner and tok[0] not in NO_SPACE_BEFORE
                         and prev[-1] not in NO_SPACE_AFTER) else ""
        parts.append(sep)
        n += len(sep)
        start, end = n, n + len(tok)
        parts.append(tok)
        n = end
        prev = tok
        kind = tag[2:] if tag[:2] in ("B-", "I-") else None
        if kind and tok.strip():
            if tag.startswith("I-") and cur and cur["label"] == kind:
                cur["end"] = end
            else:
                cur = {"start": start, "end": end, "label": kind}
                spans.append(cur)
        elif not tok.strip() and cur and tag.startswith("I-"):
            pass                            # a space token inside an entity (ThaiNER, KLUE)
        else:
            cur = None
    return "".join(parts), spans


def _tagged(source, lang, rows, joiner, names=None, id_prefix=None):
    out = []
    for i, r in enumerate(rows):
        tags = [names[t] for t in r["tags"]] if names else r["tags"]
        text, spans = detokenize(r["tokens"], tags, joiner)
        if text.strip():
            out.append({"id": f"{id_prefix or source}-{r.get('id', i)}", "source": source, "lang": lang,
                        "text": text, "spans": spans})
    return out


def _hf_tagged(source, lang, path, split, tok_col, tag_col, joiner):
    ds = load_from_disk(str(RAW / path))[split]
    names = ds.features[tag_col].feature.names
    rows = [{"tokens": t, "tags": g} for t, g in zip(ds[tok_col], ds[tag_col])]
    return _tagged(source, lang, rows, joiner, names, f"{source}-{split}")


# ---------------------------------------------------------------- NER corpora
# Thai titles ThaiNER includes in PERSON spans; our label excludes them ("คุณ{PERSON}")
TH_TITLE = re.compile(r"(?:(?:รองศาสตราจารย์|ผู้ช่วยศาสตราจารย์|ศาสตราจารย์|ด็อกเตอร์|ดร\.|นางสาว|นาง|นาย|คุณ|พล\.?[ตอท]\.?\S*)\s*)+")


def thainer(split):
    rows = _hf_tagged("thainer", "th", "ml/thainer", split, "words", "ner", "")
    for r in rows:
        for s in r["spans"]:
            if s["label"] == "PERSON":
                m = TH_TITLE.match(r["text"], s["start"], s["end"])
                if m and m.end() < s["end"]:
                    s["start"] = m.end()
    return rows


def klue_ner(split):
    return _hf_tagged("klue_ner", "ko", "ml/klue_ner", split, "tokens", "ner_tags", "")


def tlunified(split):
    return _hf_tagged("tlunified", "tl", "ml/tlunified", split, "tokens", "ner_tags", " ")


def wikiann(lang):
    def read(split):
        return _hf_tagged(f"wikiann_{lang}", lang, f"ml/wikiann_{lang}", split, "tokens", "ner_tags", " ")
    return read


def stockmark_ja(split):
    """Only a train split: a seeded 10% is used as validation."""
    ds = load_from_disk(str(RAW / "ml/stockmark_ja"))["train"]
    idx = list(range(len(ds)))
    random.Random(0).shuffle(idx)
    cut = len(idx) // 10
    keep = idx[cut:] if split == "train" else idx[:cut]
    out = []
    for i in keep:
        r = ds[i]
        spans = [{"start": e["span"][0], "end": e["span"][1], "label": e["type"]} for e in r["entities"]]
        out.append({"id": f"stockmark_ja-{r['curid']}-{i}", "source": "stockmark_ja", "lang": "ja",
                    "text": r["text"], "spans": spans})
    return out


def cluener(split):
    ds = load_from_disk(str(RAW / "ml/cluener"))[split]
    return [{"id": f"cluener-{split}-{i}", "source": "cluener", "lang": "zh-Hans", "text": r["text"],
             "spans": [{"start": e["start_offset"], "end": e["end_offset"], "label": e["label"]}
                       for e in r["entities"]]}
            for i, r in enumerate(ds)]


def anercorp(split):
    """One word per row; sentences are cut at sentence-final punctuation."""
    ds = load_from_disk(str(RAW / "ml/anercorp"))[split]
    rows, toks, tags = [], [], []
    for w, t in zip(ds["word"], ds["tag"]):
        toks.append(w)
        tags.append(t.replace("PERS", "PER"))
        if w in (".", "!", "?", "؟") and len(toks) > 3:
            rows.append({"tokens": toks, "tags": tags})
            toks, tags = [], []
    if toks:
        rows.append({"tokens": toks, "tags": tags})
    return _tagged("anercorp", "ar", rows, " ", id_prefix=f"anercorp-{split}")


def naamapadam_hi(split, limit=60000):
    """985k training sentences: the first `limit` (file order is random) are enough."""
    name = {"train": "hi_train.json", "validation": "hi_val.json", "test": "hi_test.json"}[split]
    rows = []
    with open(RAW / "ml/naamapadam_hi" / name, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            r = json.loads(line)
            rows.append({"tokens": r["words"], "tags": r["ner"], "id": i})
    return _tagged("naamapadam_hi", "hi", rows, " ", id_prefix=f"naamapadam_hi-{split}")


def nergrit(split):
    name = {"train": "train.txt", "validation": "valid.txt", "test": "test.txt"}[split]
    rows, toks, tags = [], [], []
    for line in open(RAW / "ml/nergrit" / name, encoding="utf-8"):
        line = line.rstrip("\n")
        if not line.strip():
            if toks:
                rows.append({"tokens": toks, "tags": tags})
            toks, tags = [], []
            continue
        w, t = line.split("\t")
        toks.append(w)
        tags.append(t)
    if toks:
        rows.append({"tokens": toks, "tags": tags})
    return _tagged("nergrit", "id", rows, " ", id_prefix=f"nergrit-{split}")


# ---------------------------------------------------------------- PII corpora
SITR_MEDICAL = {"medical_record_number", "medicine_name"}


def sitr(split):
    """Each row annotates its own PII type(s) only -> used positive-only (partial: all).
    Rows about medical identifiers are dropped (no medical data)."""
    ds = load_from_disk(str(RAW / "ml/sitr"))[split]
    out = []
    for i, r in enumerate(ds):
        if r["type"] in SITR_MEDICAL or any(e["label"] in SITR_MEDICAL for e in r["entities"]):
            continue
        out.append({"id": f"sitr-{split}-{i}", "source": "sitr", "lang": "ar", "text": r["text"],
                    "spans": [{"start": e["start"], "end": e["end"], "label": e["label"]} for e in r["entities"]]})
    return out


# trailing title / kinship word after a Korean name ("박호 어머님", "이ㅁㅁ 대리", "김민수 고객님")
KIII_TITLES = ("고객", "사장", "선생", "대표", "팀장", "과장", "부장", "차장", "대리", "주임", "사원", "이사",
               "상무", "전무", "실장", "본부장", "지점장", "어머", "아버", "할머", "할아버", "배우자", "씨")
KIII_HONORIFIC = re.compile(r"(\s+\S+|님|씨)$")


def strip_korean_title(name: str) -> str:
    m = KIII_HONORIFIC.search(name)
    if not m or m.start() == 0:
        return name
    word = m.group(0).strip().removesuffix("님")
    if m.group(0) in ("님", "씨") or word in KIII_TITLES or m.group(0).strip().endswith("님"):
        return strip_korean_title(name[:m.start()].rstrip())
    return name
KO_DATE = re.compile(r"\d{4}\s*년\s*\d{1,2}\s*월\s*\d{1,2}\s*일|\d{4}[-./]\s?\d{1,2}[-./]\s?\d{1,2}|\d{6}(?=-)")
KIII_SPLIT = 1200            # first 1,200 docs (by doc_id) train, the remaining 240 held out


def kiii(split):
    """Korean financial documents. Spans are phrases in places: dob_age ("생년월일은 1960년 6월
    2일이며 만 66세") is reduced to its date, honorifics are stripped from names."""
    ds = load_from_disk(str(RAW / "ml/kiii"))["test"]
    docs = sorted(ds, key=lambda r: r["doc_id"])
    docs = docs[:KIII_SPLIT] if split == "train" else docs[KIII_SPLIT:]
    out = []
    for r in docs:
        t, spans = r["text"], []
        for s in r["spans"]:
            start, end, cat = s["start"], s["end"], s["category"]
            if cat == "dob_age":
                m = KO_DATE.search(t, start, end)
                if not m:
                    continue
                start, end = m.start(), m.end()
            elif cat == "person_name":
                seg = t[start:end]
                start += len(seg) - len(seg.lstrip())
                end = start + len(strip_korean_title(seg.strip()))
            spans.append({"start": start, "end": end, "label": cat})
        for h in r["hard_negatives"] or []:
            spans.append({"start": h["start"], "end": h["end"], "label": "hard_negative"})
        out.append({"id": f"kiii-{r['doc_id']}", "source": "kiii", "lang": "ko", "text": t, "spans": spans})
    return out


def ai4privacy_15m(lang):
    """AI4Privacy OpenPII-1.5M train split, one language (rows in AI4Privacy format)."""
    def read(split):
        hf_split = {"train": "train", "validation": "validation"}[split]
        ds = load_from_disk(str(RAW / "ai4privacy"))[hf_split]
        ds = ds.filter(lambda b: [l == lang for l in b["language"]], batched=True)
        return ds
    return read


def ai4privacy_500k_hi(split):
    path = RAW / "ai4privacy500k" / ("train.jsonl" if split == "train" else "validation.jsonl")
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("language") == "hi":
                rows.append(r)
    return rows


READERS = {
    "thainer": thainer, "klue_ner": klue_ner, "tlunified": tlunified, "stockmark_ja": stockmark_ja,
    "cluener": cluener, "anercorp": anercorp, "naamapadam_hi": naamapadam_hi, "nergrit": nergrit,
    "wikiann_ar": wikiann("ar"), "wikiann_ms": wikiann("ms"), "wikiann_vi": wikiann("vi"),
    "sitr": sitr, "kiii": kiii,
}
