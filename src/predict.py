"""Text -> PII spans, with sliding windows for long documents.

    python src/predict.py --text "Call John Smith at 555-123-4567"
"""
import argparse
import bisect
import json

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from adapters import LANG_IDS, AdapterTokenClassifier, is_adapter_model
from deep_head import DeepHeadTokenClassifier, is_deep_head_model
from heads import StructHeadTokenClassifier, is_struct_head_model
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
                 script_boundaries: bool | None = None, trim_punct: bool | None = None, merge: str = "max",
                 int8: bool = False):
        # how a token seen in several overlapping windows is decided (v5 no-retraining experiments):
        #   max      the window with the most confident prediction (default; all results so far)
        #   central  the window where the token is furthest from an edge (most context on both sides)
        #   avg      softmax heads: mean of the windows' label probabilities; span head: each candidate
        #            span's logit averaged over every window that contains it, then decoded once
        assert merge in ("max", "central", "avg"), merge
        self.merge = merge
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        # Model B: language adapters on a shared encoder, routed per segment (router.py)
        self.adapter = is_adapter_model(model_dir)
        if self.adapter:
            from router import ROUTER_NAME, Router
            from pathlib import Path
            self.model = AdapterTokenClassifier.from_pretrained(model_dir).to(self.device).eval()
            self.router = Router.load(model_dir) if (Path(model_dir) / ROUTER_NAME).exists() else Router()
        elif is_struct_head_model(model_dir):   # v5 CRF / span head
            self.model = StructHeadTokenClassifier.from_pretrained(model_dir).to(self.device).eval()
        elif is_deep_head_model(model_dir):     # v5 deep head: extra transformer layers before the classifier
            self.model = DeepHeadTokenClassifier.from_pretrained(model_dir).to(self.device).eval()
        else:
            self.model = AutoModelForTokenClassification.from_pretrained(
                model_dir, dtype=torch.float32).to(self.device).eval()
        if int8:
            # v5 latency experiment: dynamic int8 weights for the encoder's Linear layers (CPU only); the
            # classifier / span head stays fp32, as recommended for quantised NER heads
            assert self.device == "cpu", "int8 dynamic quantisation runs on CPU"
            keep = ("classifier", "qk", "score")
            enc = {n for n, m in self.model.named_modules() if isinstance(m, torch.nn.Linear)
                   and not any(n.split(".")[-1] == k for k in keep)}
            self.model = torch.ao.quantization.quantize_dynamic(self.model, enc, dtype=torch.qint8)
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

    @staticmethod
    def _finish_spans(text: str, spans: list[dict]) -> list[dict]:
        return [{**sp, "text": text[sp["start"]:sp["end"]]} for sp in sorted(spans, key=lambda x: x["start"])]

    def _span_avg(self, text: str, inputs, chunks) -> list[dict]:
        """Span head, merge=avg: every positive candidate span (type, char start, char end) from any
        window gets the mean of its logits over all windows that contain both of its words, then one
        greedy non-overlapping decode over the whole document."""
        from tokenize_bio import _trim_offset
        m = self.model
        scores, wmaps, cand = [], [], {}
        for c, ((ids, att, valid), chunk) in enumerate(zip(inputs, chunks)):
            sc_t = m.span_scores_masked(ids, att, valid)
            scores.append(sc_t)
            pos = (sc_t > 0).nonzero().tolist()
            vl = valid.tolist()
            base = len(wmaps)
            for w, win in enumerate(chunk):
                offs, n, spec = win["offset_mapping"], len(win["input_ids"]), win["special_tokens_mask"]
                vt = [t for t in range(n) if vl[w][t]]
                start_c, end_c = {}, {}
                for k, t in enumerate(vt):          # a word runs up to the token before the next word start
                    nxt = vt[k + 1] if k + 1 < len(vt) else n
                    # whitespace-only tokens (line / tab markers, a lone "▁") are not part of the word
                    end_c[t] = max((offs[u][1] for u in range(t, nxt) if not spec[u]
                                    and _trim_offset(text, *offs[u])[0] < offs[u][1]), default=offs[t][1])
                    start_c[t] = _trim_offset(text, offs[t][0], offs[t][1])[0]
                wmaps.append((c, w, start_c, end_c, {v: t for t, v in start_c.items()}, {v: t for t, v in end_c.items()}))
            for r, e, a, b in pos:
                _, _, sc, ec, _, _ = wmaps[base + r]
                cand[(e, sc[a], ec[b])] = None
        # each candidate's logit in every window that contains both of its words (one gather per chunk)
        logit_sum = dict.fromkeys(cand, 0.0)
        logit_n = dict.fromkeys(cand, 0)
        per_chunk: dict[int, list] = {}
        for (c, w, _, _, s2t, e2t) in wmaps:
            for k in cand:
                e, cs, ce = k
                if cs in s2t and ce in e2t and s2t[cs] <= e2t[ce]:
                    per_chunk.setdefault(c, []).append((k, w, e, s2t[cs], e2t[ce]))
        for c, items in per_chunk.items():
            idx = torch.tensor([it[1:] for it in items], device=scores[c].device)
            vals = scores[c][idx[:, 0], idx[:, 1], idx[:, 2], idx[:, 3]].tolist()
            for it, v in zip(items, vals):
                if v > -1e11:                       # pair outside that window's span-width limit: no vote
                    logit_sum[it[0]] += v
                    logit_n[it[0]] += 1
        avg = sorted(((logit_sum[k] / logit_n[k], k) for k in cand if logit_n[k]), key=lambda x: -x[0])
        taken, spans = [], []
        for v, (e, cs, ce) in avg:
            if v <= 0:
                break
            if any(cs < b and a < ce for a, b in taken):
                continue
            taken.append((cs, ce))
            spans.append({"start": cs, "end": ce, "label": m.entities[e],
                          "score": round(float(torch.sigmoid(torch.tensor(v))), 4)})
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
        struct = isinstance(self.model, StructHeadTokenClassifier)
        batches, inputs = [], []
        for b in range(0, len(wins), self.batch_windows):
            chunk = wins[b:b + self.batch_windows]
            width = max(len(w["input_ids"]) for w in chunk)
            pad = self.tok.pad_token_id
            ids = torch.tensor([w["input_ids"] + [pad] * (width - len(w["input_ids"])) for w in chunk])
            mask = torch.tensor([[1] * len(w["input_ids"]) + [0] * (width - len(w["input_ids"]))
                                 for w in chunk])
            extra = {} if lang_id is None else {"lang_ids": torch.full((len(chunk),), lang_id, device=self.device)}
            if struct:
                # structured decode over the word-initial tokens of each window (Viterbi / span pointer);
                # all batches are decoded together below (heads.decode_many)
                valid = torch.zeros(ids.shape, dtype=torch.bool)
                for w, win in enumerate(chunk):
                    st = word_starts(text, win["offset_mapping"], self.script_boundaries)
                    valid[w, :len(st)] = torch.tensor(st) & ~torch.tensor(win["special_tokens_mask"], dtype=torch.bool)
                inputs.append((ids.to(self.device), mask.to(self.device), valid.to(self.device)))
                batches.append((chunk, None, None, None))
                continue
            logits = self.model(input_ids=ids.to(self.device), attention_mask=mask.to(self.device), **extra).logits
            probs = logits.float().softmax(-1)
            top_p, top_lab = probs.max(-1)
            batches.append((chunk, top_lab.tolist(), top_p.tolist(), probs.cpu() if self.merge == "avg" else None))
        if struct:
            if self.merge == "avg":
                assert self.model.head == "span", "merge=avg for structured heads: span head only"
                return self._finish_spans(text, self._span_avg(text, inputs, [b[0] for b in batches]))
            batches = [(c, lab, p, None) for (c, _, _, _), (lab, p) in zip(batches, self.model.decode_many(inputs))]
        acc: dict[tuple[int, int], list] = {}
        for chunk, top_lab, top_p, probs in batches:
            # a token seen in several windows: most confident window (max), most central window, or mean
            for w, win in enumerate(chunk):
                body = [i for i, sp in enumerate(win["special_tokens_mask"]) if not sp]
                lo, hi = body[0], body[-1]
                for i, (s, e) in enumerate(win["offset_mapping"]):
                    if win["special_tokens_mask"][i] or s == e:
                        continue
                    if probs is not None:
                        a = acc.setdefault((s, e), [0, 0])
                        a[0] = a[0] + probs[w, i]
                        a[1] += 1
                        continue
                    p = top_p[w][i]
                    key = p if self.merge == "max" else (min(i - lo, hi - i), p)
                    if (s, e) not in best or key > best[(s, e)][2]:
                        best[(s, e)] = (p, top_lab[w][i], key)
        for o, (tot, n) in acc.items():
            pm, lab = (tot / n).max(-1)
            best[o] = (float(pm), int(lab), None)

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
