"""Language router for Model B: text segment -> language code (adapters.LANG2ADAPTER keys).

  1. script rules on the letters of the segment: Hangul -> ko, any kana -> ja, Han -> zh,
     Devanagari -> hi, Arabic -> ar, Thai -> th
  2. zh-Hans vs zh-Hant: characters that only exist in one form (OpenCC s2t / t2s per character)
  3. Latin script (en / vi / ms / id / tl): character 1-4-gram logistic regression trained on the
     training texts (Vietnamese diacritics, Malay vs Indonesian spelling ...)

Documents are routed per segment (blank-line separated blocks, see `segments`), so a mixed
English / Chinese document uses both adapters.
"""
import pickle
import re
import unicodedata
from collections import Counter
from pathlib import Path

LATIN_LANGS = ["en", "vi", "ms", "id", "tl"]
ROUTER_NAME = "router.pkl"
SEGMENT_BREAK = re.compile(r"\n\s*\n")


def letter_scripts(text: str) -> Counter:
    c = Counter()
    for ch in text:
        if not ch.isalpha():
            continue
        o = ord(ch)
        if 0xAC00 <= o <= 0xD7AF or 0x1100 <= o <= 0x11FF or 0x3130 <= o <= 0x318F:
            c["hangul"] += 1
        elif 0x3040 <= o <= 0x30FF or 0x31F0 <= o <= 0x31FF or 0xFF66 <= o <= 0xFF9F:
            c["kana"] += 1
        elif 0x3400 <= o <= 0x9FFF or 0xF900 <= o <= 0xFAFF or 0x20000 <= o <= 0x2FA1F:
            c["han"] += 1
        elif 0x0900 <= o <= 0x097F:
            c["devanagari"] += 1
        elif 0x0600 <= o <= 0x06FF or 0x0750 <= o <= 0x077F or 0xFB50 <= o <= 0xFEFF:
            c["arabic"] += 1
        elif 0x0E00 <= o <= 0x0E7F:
            c["thai"] += 1
        elif unicodedata.name(ch, "").startswith("LATIN"):
            c["latin"] += 1
        else:
            c["other"] += 1
    return c


class Router:
    def __init__(self, latin_clf=None):
        self.latin_clf = latin_clf
        self._s2t = self._t2s = None

    # -------------------------------------------------------------- rules
    def script_lang(self, text: str) -> str | None:
        c = letter_scripts(text)
        n = sum(c.values())
        if not n:
            return None
        if c["kana"] >= 2 or (c["kana"] and c["kana"] >= 0.05 * n):
            return "ja"
        for script, lang in [("hangul", "ko"), ("thai", "th"), ("arabic", "ar"), ("devanagari", "hi"), ("han", "zh")]:
            if c[script] >= 0.2 * n:
                return lang
        return "latin"

    def zh_variant(self, text: str) -> str:
        if self._s2t is None:
            import opencc
            self._s2t, self._t2s = opencc.OpenCC("s2t"), opencc.OpenCC("t2s")
        han = [ch for ch in text if 0x3400 <= ord(ch) <= 0x9FFF]
        simp = sum(self._s2t.convert(ch) != ch for ch in han)    # has a different Traditional form
        trad = sum(self._t2s.convert(ch) != ch for ch in han)    # has a different Simplified form
        return "zh-Hant" if trad > simp else "zh-Hans"

    # -------------------------------------------------------------- routing
    def route(self, text: str, default: str = "en") -> str:
        lang = self.script_lang(text)
        if lang is None:
            return default
        if lang == "zh":
            return self.zh_variant(text)
        if lang == "latin":       # untrained router: Latin script falls back to English
            return "en" if self.latin_clf is None else str(self.latin_clf.predict([text])[0])
        return lang

    def route_segments(self, text: str, default: str = "en") -> list[tuple[int, int, str]]:
        """(start, end, lang) per segment; consecutive segments with one language are merged, and
        a segment without letters (numbers only) joins its neighbour."""
        out = []
        for s, e in segments(text):
            lang = self.route(text[s:e], default=None) if text[s:e].strip() else None
            if out and (lang is None or lang == out[-1][2]):
                out[-1] = (out[-1][0], e, out[-1][2])
            else:
                out.append((s, e, lang))
        first = next((l for _, _, l in out if l), default)
        return [(s, e, l or first) for s, e, l in out]

    # -------------------------------------------------------------- train / persist
    @classmethod
    def train(cls, texts: list[str], langs: list[str]) -> "Router":
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        pairs = [(t, l) for t, l in zip(texts, langs) if l in LATIN_LANGS]
        clf = make_pipeline(TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 4), min_df=3, sublinear_tf=True,
                                            max_features=300_000),
                            LogisticRegression(max_iter=2000, C=4.0))
        clf.fit([t for t, _ in pairs], [l for _, l in pairs])
        return cls(clf)

    def save(self, model_dir):
        with open(Path(model_dir) / ROUTER_NAME, "wb") as f:
            pickle.dump(self.latin_clf, f)

    @classmethod
    def load(cls, model_dir) -> "Router":
        with open(Path(model_dir) / ROUTER_NAME, "rb") as f:
            return cls(pickle.load(f))


def segments(text: str) -> list[tuple[int, int]]:
    """Blank-line separated blocks as (start, end); the separators belong to the preceding block."""
    out, start = [], 0
    for m in SEGMENT_BREAK.finditer(text):
        out.append((start, m.end()))
        start = m.end()
    if start < len(text) or not out:
        out.append((start, len(text)))
    return out


def main():
    """Train the Latin-script classifier on ml_train and report routing accuracy on the test sets.

        python router.py --out ../outputs/router
    """
    import argparse
    import json
    import random
    from common import ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "outputs/router"))
    ap.add_argument("--per_lang", type=int, default=8000)
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(ROOT / "data/processed/ml_train.jsonl", encoding="utf-8")]
    random.Random(0).shuffle(rows)
    by = {l: [r["text"] for r in rows if r["lang"] == l][:args.per_lang] for l in LATIN_LANGS}
    router = Router.train([t for l in LATIN_LANGS for t in by[l]], [l for l in LATIN_LANGS for _ in by[l]])
    Path(args.out).mkdir(parents=True, exist_ok=True)
    router.save(args.out)
    report = {}
    for split in ["ml_real", "ml_synth", "ml_kiii_test"]:
        conf = {}
        for line in open(ROOT / "data/processed" / f"{split}.jsonl", encoding="utf-8"):
            r = json.loads(line)
            pred = router.route(r["text"])
            conf.setdefault(r["lang"], Counter())[pred] += 1
        report[split] = {g: {"acc": c[g] / sum(c.values()), "n": sum(c.values()), "confused": dict(c.most_common(3))}
                         for g, c in conf.items()}
        print(f"\n{split}: routing accuracy per language (document level)")
        for g, v in report[split].items():
            print(f"  {g:8s} {v['acc']:.3f}  n={v['n']:5d}  {v['confused']}")
    with open(ROOT / "results/router_accuracy.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
