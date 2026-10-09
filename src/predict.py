"""Text -> PII spans, with sliding windows for long documents.

    python src/predict.py --text "Call John Smith at 555-123-4567"
"""
import argparse
import bisect
import json

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from adapters import LANG_IDS, AdapterTokenClassifier, is_adapter_model
from common import ROOT, load_yaml
from tokenize_bio import fill_continuations, labels_to_spans, window_encode, word_starts


OPEN, CLOSE = "([{【「『（［", ")]}】」』）］"
PAIRS = dict(zip(CLOSE, OPEN))
LEAD = "、，,：:;；·・"
# a trailing full stop / comma is trimmed only after a digit for number-like labels; BUSINESS keeps
# "Pte. Ltd." and PERSON keeps initials
NUMERIC = {"ACCOUNT", "PHONE", "TIN", "DOB", "ADDRESS"}


def trim_span(text: str, sp: dict) -> dict | None:
    """Drop punctuation the tokenizer glued onto an entity (v5 error analysis): an unmatched closing
    bracket ('2317299980)'), a leading list comma / colon ('、5819224667', '：2888...'), and a final
    '.' / '。' / ',' after a digit for number-like labels ('607009209.'). Returns None if nothing is left."""
    a, b = sp["start"], sp["end"]
    while a < b and (text[a] in LEAD or text[a].isspace() or (text[a] in OPEN and not any(c in CLOSE for c in text[a:b]))):
        a += 1
    while a < b:
        c = text[b - 1]
        if c.isspace():
            b -= 1
        elif c in CLOSE and PAIRS[c] not in text[a:b - 1]:
            b -= 1
        elif c in ".。,，、;；:：" and sp["label"] in NUMERIC and b - 2 >= a and (text[b - 2].isdigit() or text[b - 2] in CLOSE):
            b -= 1
        else:
            break
    # a span fully wrapped in one pair of brackets: "(陈伟明)" -> "陈伟明"
    if b - a > 2 and text[a] in OPEN and PAIRS.get(text[b - 1]) == text[a] and not any(c in OPEN + CLOSE for c in text[a + 1:b - 1]):
        a, b = a + 1, b - 1
    if a >= b:
        return None
    return {**sp, "start": a, "end": b, "text": text[a:b]} if (a, b) != (sp["start"], sp["end"]) else sp


class PIIPredictor:
    def __init__(self, model_dir: str, max_length: int = 384, stride: int = 96,
                 label_all_tokens: bool = False, device=None, batch_windows: int = 16,
                 script_boundaries: bool | None = None, trim_punct: bool | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        # Model B: language adapters on a shared encoder, routed per segment (router.py)
        self.adapter = is_adapter_model(model_dir)
        if self.adapter:
            from router import ROUTER_NAME, Router
            from pathlib import Path
            self.model = AdapterTokenClassifier.from_pretrained(model_dir).to(self.device).eval()
            self.router = Router.load(model_dir) if (Path(model_dir) / ROUTER_NAME).exists() else Router()
        else:
            self.model = AutoModelForTokenClassification.from_pretrained(
                model_dir, dtype=torch.float32).to(self.device).eval()
        self.id2label = self.model.config.id2label
        # models trained with line/tab markers record it in their config (v2.1+)
        self.show_breaks = bool(getattr(self.model.config, "visible_breaks", False))
        # word boundaries in unspaced scripts (multilingual models); None = as the model was trained
        self.script_boundaries = (bool(getattr(self.model.config, "script_boundaries", False))
                                  if script_boundaries is None else script_boundaries)
        self.max_length, self.stride = max_length, stride
        self.label_all_tokens = label_all_tokens
        self.batch_windows = batch_windows
        # punctuation trimming (v5.0+ releases set "trim_punct": true in config.json); models without
        # the key behave as before, so earlier results stay comparable. None = as the config says.
        self.trim_punct = (bool(getattr(self.model.config, "trim_punct", False))
                           if trim_punct is None else trim_punct)

    def predict(self, text: str, lang: str | None = None) -> list[dict]:
        """PII spans in `text`. Adapter models route each segment to its language's adapter
        (`lang` forces one language for the whole text, e.g. the gold language)."""
        spans = self._predict_routed(text, lang)
        if self.trim_punct:
            spans = [t for t in (trim_span(text, sp) for sp in spans) if t]
        return spans

    def _predict_routed(self, text: str, lang: str | None) -> list[dict]:
        if not self.adapter:
            return self._predict(text, None)
        routed = [(0, len(text), lang)] if lang else self.router.route_segments(text)
        spans = []
        for s, e, seg_lang in routed:
            for sp in self._predict(text[s:e], LANG_IDS[seg_lang]):
                spans.append({**sp, "start": sp["start"] + s, "end": sp["end"] + s, "lang": seg_lang})
        return spans

    @torch.no_grad()
    def _predict(self, text: str, lang_id: int | None) -> list[dict]:
        wins = window_encode(self.tok, text, self.max_length, self.stride, self.show_breaks,
                             blank_lone_space=self.script_boundaries)
        best: dict[tuple[int, int], tuple[float, int]] = {}
        for b in range(0, len(wins), self.batch_windows):
            chunk = wins[b:b + self.batch_windows]
            width = max(len(w["input_ids"]) for w in chunk)
            pad = self.tok.pad_token_id
            ids = torch.tensor([w["input_ids"] + [pad] * (width - len(w["input_ids"])) for w in chunk])
            mask = torch.tensor([[1] * len(w["input_ids"]) + [0] * (width - len(w["input_ids"]))
                                 for w in chunk])
            extra = {} if lang_id is None else {"lang_ids": torch.full((len(chunk),), lang_id, device=self.device)}
            logits = self.model(input_ids=ids.to(self.device), attention_mask=mask.to(self.device), **extra).logits
            top_p, top_lab = logits.float().softmax(-1).max(-1)
            top_p, top_lab = top_p.tolist(), top_lab.tolist()
            # a token seen in several windows keeps its most confident prediction
            for w, win in enumerate(chunk):
                for i, (s, e) in enumerate(win["offset_mapping"]):
                    if win["special_tokens_mask"][i] or s == e:
                        continue
                    p = top_p[w][i]
                    if (s, e) not in best or p > best[(s, e)][0]:
                        best[(s, e)] = (p, top_lab[w][i])

        offsets = sorted(best)
        starts = word_starts(text, offsets, self.script_boundaries)
        names = [self.id2label[best[o][1]] for o in offsets]
        if not self.label_all_tokens:  # OpenMed-style: word label = first sub-token's label
            names = fill_continuations(names, starts)
        spans = labels_to_spans(offsets, names, text)
        # span score = mean confidence of the word-initial tokens it covers
        ends = [o[1] for o in offsets]
        for sp in spans:
            ps, k = [], bisect.bisect_left(ends, sp["start"])
            while k < len(offsets) and offsets[k][0] < sp["end"]:
                if starts[k]:
                    ps.append(best[offsets[k]][0])
                k += 1
            sp["score"] = round(sum(ps) / len(ps), 4) if ps else 0.0
            sp["text"] = text[sp["start"]:sp["end"]]
        return spans


def main():
    cfg = load_yaml("train.yaml")
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default=str(ROOT / cfg["output_dir"]))
    ap.add_argument("--text", required=True)
    args = ap.parse_args()
    t = cfg["tokenize"]
    pred = PIIPredictor(args.model_dir, t["max_length"], t["stride"], t["label_all_tokens"])
    print(json.dumps(pred.predict(args.text), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
