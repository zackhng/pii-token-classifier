"""Wealth-management grid generator (v5): benchmark ml_wm_{dev,test} and the matching training data.

The grid (user, 2026-10-08): Span (8 labels) x Context x Language x Format.
  context   field (form cue) | before (cue before value in a sentence) | after (cue after value) |
            narrative (no explicit cue) | list (several values) | bracket (attached to another
            entity) | spoken (call transcript)
  language  12 languages, each as mono / cs_sentence (English + X inside one sentence) /
            cs_document (English and X sentences in one document); en has mono only
  format    md_table | bullets | numbered | lines | prose | transcript
  doc_type  kyc, sow, rm_note, call, email, consult, background, review (wealth-management firm)
Look-alikes (amounts, dates, references, postcodes, SWIFT codes, ISINs) are rendered through the same
context patterns and labelled O, so the context has to decide (Ma et al., EMNLP 2023).

Per-language specs live in frames/wm/<lang>.yaml. Every list in a spec is split by position into
train / dev / test (index % 3), so dev and test use cue words, context patterns and narrative
sentences that training never saw; values come from frames.Values (training pools vs test-only pools).
Coverage: pairwise - every pair of factors gets >= min_cell spans (a full 4-way grid at 50 per cell
would be ~475k spans per split).

    python wm_gen.py --split dev --min_cell 50     -> data/processed/ml_wm_dev.jsonl + coverage report
"""
import argparse
import itertools
import json
import random
from collections import Counter, defaultdict
from functools import lru_cache

import yaml

from common import ROOT
from frames import CJK, LANGS, Values

SPEC_DIR = ROOT / "frames/wm"
LABELS = ["PERSON", "BUSINESS", "ADDRESS", "DOB", "ACCOUNT", "PHONE", "EMAIL", "TIN"]
LOOK = ["AMOUNT", "DATE", "REF", "POSTCODE", "CODE", "ISIN"]
CONTEXTS = ["field", "before", "after", "narrative", "list", "bracket", "spoken"]
FORMATS = ["md_table", "bullets", "numbered", "lines", "prose", "transcript"]
MODES = ["mono", "cs_sentence", "cs_document"]
DOC_TYPES = ["kyc", "sow", "rm_note", "call", "email", "consult", "background", "review"]
SPLIT_IDX = {"train": 0, "dev": 1, "test": 2}
BULLETS = ["- ", "* ", "• ", "– ", "+ "]
NUMBERS = [lambda i: f"{i}. ", lambda i: f"{i}) ", lambda i: f"({chr(96 + i)}) ", lambda i: f"{['i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii', 'viii', 'ix', 'x', 'xi', 'xii'][i - 1]}. ",
           lambda i: f"({i}) ", lambda i: f"{chr(64 + i)}. "]
# context x format combinations that occur in real documents
VALID = {("field", f) for f in ["md_table", "bullets", "numbered", "lines", "prose"]} | \
        {(c, f) for c in ["before", "after", "narrative", "list", "bracket"] for f in ["bullets", "numbered", "lines", "prose"]} | \
        {("spoken", "transcript"), ("list", "md_table"), ("bracket", "md_table")}
SLOT_FOR = {"PERSON": "PERSON", "BUSINESS": "BUSINESS", "ADDRESS": "ADDRESS", "DOB": "DOB", "ACCOUNT": "ACCOUNT",
            "PHONE": "PHONE", "EMAIL": "EMAIL", "TIN": "TIN"}


@lru_cache(maxsize=None)
def spec(lang: str) -> dict:
    return yaml.safe_load(open(SPEC_DIR / f"{lang}.yaml", encoding="utf-8"))


def part(xs: list, split: str) -> list:
    """The split's third of a spec list (index % 3); falls back to the whole list if it is short."""
    out = [x for i, x in enumerate(xs) if i % 3 == SPLIT_IDX[split]]
    return out or list(xs)


class Piece:
    """Text with spans; spans carry label, context and the cue / pattern language."""

    def __init__(self, text="", spans=None):
        self.text, self.spans = text, spans or []

    def add(self, other, sep=""):
        off = len(self.text) + len(sep)
        self.text += sep + other.text
        self.spans += [{**s, "start": s["start"] + off, "end": s["end"] + off} for s in other.spans]
        return self


def fill(pattern: str, slots: dict) -> Piece:
    """Fill {name} placeholders; slots: name -> (text, label or None, ctx) or list of those for {vs}."""
    out, i = Piece(), 0
    import re
    for m in re.finditer(r"\{(\w+)\}", pattern):
        out.text += pattern[i:m.start()]
        val = slots[m.group(1)]
        if isinstance(val, Piece):
            out.add(val)
        else:
            text, label, ctx = val
            if label:
                out.spans.append({"start": len(out.text), "end": len(out.text) + len(text), "label": label, "ctx": ctx})
            out.text += text
        i = m.end()
    out.text += pattern[i:]
    return out


def in_sentence(cue: str) -> str:
    """Form-label cue used mid-sentence: 'Beneficial owner' -> 'beneficial owner'; acronyms (DOB,
    TIN, A/C, SWIFT) and non-Latin cues stay as they are."""
    w = cue.split(" ")[0]
    if not w[:1].isascii() or not w[:1].isupper() or (len(w) > 1 and (w[1].isupper() or not w[1:].isalpha())):
        return cue
    return cue[0].lower() + cue[1:]


def cap(p: Piece) -> Piece:
    """Capitalise a sentence that starts with a lower-case Latin letter (not inside a span)."""
    if p.text[:1].islower() and p.text[:1].isascii() and not any(s["start"] == 0 for s in p.spans):
        p.text = p.text[0].upper() + p.text[1:]
    return p


class Gen:
    def __init__(self, split: str, seed: int):
        self.split = split
        self.values = Values("train" if split == "train" else "test")
        self.rng = random.Random(f"wm-{split}-{seed}")

    # ------------------------------------------------------------ values
    def value(self, label: str, lang: str) -> tuple[str, str]:
        """(text, gold label) for an entity or look-alike; look-alikes return label 'O'."""
        self.values.start_row()
        if label in LOOK:
            v, _ = self.values.value(label, lang, self.rng)
            return v, "O"
        slot = SLOT_FOR[label]
        if label == "BUSINESS" and self.rng.random() < 0.3:
            slot = "TICKER"
        v, sp = self.values.value(slot, lang, self.rng)
        v = v.replace("\n", ", ")
        return v, label

    def cue(self, label: str, lang: str, plural=False) -> str:
        key = "plural" if plural else "cues"
        cues = spec(lang)[key].get(label) or spec(lang)["cues"][label]
        return self.rng.choice(part(cues, self.split))

    # ------------------------------------------------------------ one item = one labelled value in a context
    def item(self, label: str, ctx: str, lang: str, cue_lang: str) -> Piece:
        """A sentence / field containing `label` in context `ctx`. cue_lang != lang = code-switched."""
        S, r = spec(lang), self.rng
        v, gold = self.value(label, lang if label not in LOOK else lang)
        if ctx == "field":
            return fill("{c}{sep}{v}", {"c": (self.cue(label, cue_lang), None, None), "sep": (r.choice(S["field_seps"]), None, None),
                                        "v": (v, gold, ctx)})
        if ctx in ("before", "after", "spoken"):
            pat = r.choice(part(S["patterns"][ctx], self.split))
            c = self.cue(label, cue_lang)
            p = cap(fill(pat, {"c": (in_sentence(c), None, None), "v": (v, gold, ctx)}))
            if ctx == "spoken":         # the RM's question names the field; the client answers
                p.ask = cap(fill(r.choice(part(spec(cue_lang)["ask"], self.split)), {"c": (in_sentence(c), None, None)})).text
            return p
        if ctx == "list":
            n = r.randint(2, 3)
            vals = [v] + [self.value(label, lang)[0] for _ in range(n - 1)]
            vs = Piece()
            for k, x in enumerate(vals):
                if k:
                    vs.text += S["and"] if k == n - 1 and r.random() < 0.6 else S["comma"]
                vs.add(Piece(x, [{"start": 0, "end": len(x), "label": gold, "ctx": ctx}]))
            pat = r.choice(part(S["patterns"]["list"], self.split))
            return cap(fill(pat, {"c": (in_sentence(self.cue(label, cue_lang, plural=True)), None, None), "vs": vs}))
        if ctx == "bracket":
            anchor_label = "BUSINESS" if label == "PERSON" and r.random() < 0.3 else "PERSON"
            if label == "PERSON":
                anchor_label = "BUSINESS"
            a, ag = self.value(anchor_label, lang)
            pat = r.choice(part(S["patterns"]["bracket"], self.split))
            return fill(pat, {"a": (a, ag, "bracket_anchor"), "c": (self.cue(label, cue_lang), None, None), "v": (v, gold, ctx)})
        if ctx == "narrative":
            key = label if label in S["narrative"] else "LOOKALIKE"
            pat = r.choice(part(S["narrative"][key], self.split))
            return fill(pat, {"v": (v, gold, ctx)})
        raise ValueError(ctx)

    # ------------------------------------------------------------ documents
    def document(self, plan: list[tuple], lang: str, mode: str, fmt: str, doc_type: str) -> dict:
        """plan: [(label, ctx)], rendered in `fmt`; fillers with look-alikes are mixed in."""
        r = self.rng
        S = spec(lang)
        langs_for = lambda k: ("en" if (mode == "cs_document" and k % 2) else lang)
        items = []
        for k, (label, ctx) in enumerate(plan):
            il = langs_for(k)
            cue_lang = il
            if mode == "cs_sentence":
                cue_lang = "en" if lang != "en" and r.random() < 0.7 else lang
                if lang != "en" and cue_lang == lang:
                    il = "en" if ctx in ("before", "after", "spoken") else lang     # English sentence, X value
            items.append((self.item(label, ctx, il, cue_lang), ctx, il))
        # look-alikes in the same contexts as the entities (CrossCategory contrasts): "Ref no.: 1234567890"
        for _ in range(r.randint(1, 2)):
            ctx = "field" if fmt == "md_table" else "spoken" if fmt == "transcript" else r.choice(["field", "before", "after"])
            il = langs_for(r.randint(0, 1))
            items.insert(r.randint(0, len(items)), (self.item(r.choice(LOOK), ctx, il, il), ctx, il))
        # look-alike fillers from the document type (amounts, dates, refs ...)
        for _ in range(r.randint(1, 2)):
            fl = lang if mode != "cs_document" or r.random() < 0.5 else "en"
            pat = r.choice(part(spec(fl)["fillers"][doc_type], self.split))
            slots = {}
            import re
            for name in re.findall(r"\{(\w+)\}", pat):
                v, _ = self.value(name, fl)
                slots[name] = (v, "O", "filler")
            items.insert(r.randint(0, len(items)), (fill(pat, slots), "filler", fl))
        title = r.choice(part(S["titles"][doc_type], self.split))
        doc = self.render(items, fmt, title, lang)
        ents = [s for s in doc.spans if s["label"] != "O"]
        negs = [{"start": s["start"], "end": s["end"], "label": "lookalike", "ctx": s["ctx"]} for s in doc.spans if s["label"] == "O"]
        return {"text": doc.text, "spans": ents, "negs": negs, "lang": lang, "mode": mode, "format": fmt, "doc_type": doc_type}

    def render(self, items, fmt, title, lang) -> Piece:
        r, S = self.rng, spec(lang)
        out = Piece(title)
        if fmt == "md_table":
            head = r.choice(S["table_headers"])
            out.text += f"\n\n| {head[0]} | {head[1]} |\n|---|---|"
            for p, ctx, il in items:
                if ctx == "field":          # "cue: value" -> two cells
                    c_end = min(s["start"] for s in p.spans) if p.spans else len(p.text)
                    cue = p.text[:c_end].rstrip(" :：=-–\t").strip()
                    cell = Piece(p.text[c_end:], [{**s, "start": s["start"] - c_end, "end": s["end"] - c_end} for s in p.spans])
                    out.text += f"\n| {cue} | "
                    out.add(cell)
                    out.text += " |"
                else:
                    out.text += "\n| " + r.choice(S["table_note"]) + " | "
                    out.add(p)
                    out.text += " |"
            return out
        if fmt == "transcript":
            spk = S["speakers"]
            for k, (p, ctx, il) in enumerate(items):
                if ctx == "spoken" and getattr(p, "ask", None) and r.random() < 0.85:
                    out.text += f"\n{spk['rm']}: {p.ask}"
                who = spk["client"] if ctx == "spoken" else spk["rm"]
                out.text += f"\n{who}: "
                out.add(p)
            return out
        if fmt == "bullets":
            b = r.choice(BULLETS)
            for p, *_ in items:
                out.text += "\n" + b
                out.add(p)
            return out
        if fmt == "numbered":
            nf = r.choice(NUMBERS)
            for k, (p, *_) in enumerate(items, 1):
                out.text += "\n" + nf(k)
                out.add(p)
            return out
        if fmt == "lines":
            for p, *_ in items:
                out.text += "\n"
                out.add(p)
            return out
        # prose: sentences joined into one or two paragraphs; fields become "cue: value" clauses
        out.text += "\n\n"
        sep = "" if lang in CJK else " "
        for k, (p, ctx, il) in enumerate(items):
            if k:
                out.text += "\n\n" if r.random() < 0.15 else sep      # every item already ends a sentence
            out.add(p)
            if not p.text.rstrip().endswith(tuple(".。!?！？।")):
                out.text += S["period"]
        return out


def plans_for(split: str, min_cell: int, seed: int, langs: list[str], max_docs: int | None = None):
    """Document plans with pairwise coverage: language settings and formats cycle round-robin; each
    document takes the least-covered (label, context) pairs for its setting and format. Stops when
    every factor pair (label x context, label x format, label x language, context x format, context x
    language, format x language) has >= min_cell spans."""
    rng = random.Random(f"wm-plan-{split}-{seed}")
    settings = [(l, m) for l in langs for m in MODES if not (l == "en" and m != "mono")]
    counts = Counter()

    def keys(label, ctx, setting, fmt):
        f = {"label": label, "ctx": ctx, "lang": f"{setting[0]}/{setting[1]}", "format": fmt}
        return [(a, f[a], b, f[b]) for a, b in itertools.combinations(sorted(f), 2)]

    combos = [(lab, ctx, fmt) for lab in LABELS for ctx in CONTEXTS for fmt in FORMATS if (ctx, fmt) in VALID]
    all_keys = {k for st in settings for lab, ctx, fmt in combos for k in keys(lab, ctx, st, fmt)}
    docs, i = [], 0
    while True:
        setting = settings[i % len(settings)]
        fmt = FORMATS[(i // len(settings)) % len(FORMATS)]
        i += 1
        cand = [(lab, ctx) for lab, ctx, f in combos if f == fmt]
        cand.sort(key=lambda lc: min(counts[k] for k in keys(lc[0], lc[1], setting, fmt)) + rng.random())
        n = rng.randint(3, 6)
        plan, seen = [], set()
        for lab, ctx in cand:
            if lab not in seen or rng.random() < 0.2:
                plan.append((lab, ctx))
                seen.add(lab)
            if len(plan) == n:
                break
        rng.shuffle(plan)
        for lab, ctx in plan:
            for k in keys(lab, ctx, setting, fmt):
                counts[k] += 1
        doc_type = "call" if fmt == "transcript" else rng.choice([d for d in DOC_TYPES if d != "call"])
        docs.append((plan, setting, fmt, doc_type))
        if max_docs and len(docs) >= max_docs:
            return docs
        if not max_docs and i % 1000 == 0 and min(counts[k] for k in all_keys) >= min_cell:
            return docs


def generate(split: str, min_cell: int, seed: int = 42, langs=None, max_docs: int | None = None,
             known_formats: bool = False) -> list[dict]:
    """Benchmark splits: until every factor pair has >= min_cell spans. Training: max_docs documents
    (same round-robin balance over languages / modes / formats)."""
    langs = langs or [l for l in LANGS if (SPEC_DIR / f"{l}.yaml").exists()]
    gen, out = Gen(split, seed + (7 if known_formats else 0)), []
    if known_formats:               # account institutions from the training pool (formats seen in training)
        gen.values.acct_split = "train"
    for i, (plan, (lang, mode), fmt, doc_type) in enumerate(plans_for(split, min_cell, seed, langs, max_docs)):
        d = gen.document(plan, lang, mode, fmt, doc_type)
        name = "known" if known_formats else split
        d.update(id=f"wm_{name}_{i}", source=f"ml_wm_{name}" if split != "train" else "wm_frames")
        out.append(d)
    return out


def coverage(rows: list[dict]) -> str:
    c = Counter()
    for r in rows:
        for s in r["spans"]:
            if s["ctx"] == "bracket_anchor":
                continue
            c[("label x ctx", s["label"], s["ctx"])] += 1
            c[("label x format", s["label"], r["format"])] += 1
            c[("label x lang", s["label"], f"{r['lang']}/{r['mode']}")] += 1
            c[("ctx x format", s["ctx"], r["format"])] += 1
            c[("ctx x lang", s["ctx"], f"{r['lang']}/{r['mode']}")] += 1
            c[("format x lang", r["format"], f"{r['lang']}/{r['mode']}")] += 1
    lines = []
    for grid in ["label x ctx", "label x format", "ctx x format", "label x lang", "ctx x lang", "format x lang"]:
        cells = {(k[1], k[2]): v for k, v in c.items() if k[0] == grid}
        lines.append(f"{grid}: {len(cells)} cells, min {min(cells.values())}, median {sorted(cells.values())[len(cells) // 2]}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "dev", "test"], required=True)
    ap.add_argument("--min_cell", type=int, default=50)
    ap.add_argument("--langs", nargs="*")
    ap.add_argument("--show", type=int, default=0)
    ap.add_argument("--known_formats", action="store_true",
                    help="test phrasing / values but accounts from the training institutions -> ml_wm_known.jsonl")
    a = ap.parse_args()
    rows = generate(a.split, a.min_cell, langs=a.langs, known_formats=a.known_formats)
    from build_dataset import write_jsonl
    from build_ml_eval import tag_scripts
    if a.split != "train":
        rows = [tag_scripts(r) for r in rows]
        write_jsonl(ROOT / f"data/processed/ml_wm_{'known' if a.known_formats else a.split}.jsonl", rows)
    else:
        write_jsonl(ROOT / "data/processed/wm_train_frames.jsonl", rows)
    print(f"{a.split}: {len(rows)} docs, {sum(len(r['spans']) for r in rows)} spans, "
          f"{sum(len(r['negs']) for r in rows)} look-alikes\n{coverage(rows)}")
    for r in random.Random(0).sample(rows, min(a.show, len(rows))):
        print("-" * 70, f"\n[{r['lang']} {r['mode']} {r['format']} {r['doc_type']}]\n{r['text']}")
        print("   ", [(r["text"][s["start"]:s["end"]], s["label"], s["ctx"]) for s in r["spans"]])


if __name__ == "__main__":
    main()
