"""Text -> PII spans, with sliding windows for long documents.

    python src/predict.py --text "Call John Smith at 555-123-4567"
"""
import argparse
import bisect
import json

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from common import ROOT, load_yaml
from tokenize_bio import fill_continuations, labels_to_spans, window_encode, word_starts


class PIIPredictor:
    def __init__(self, model_dir: str, max_length: int = 384, stride: int = 96,
                 label_all_tokens: bool = False, device=None, batch_windows: int = 16,
                 script_boundaries: bool | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_dir)
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

    @torch.no_grad()
    def predict(self, text: str) -> list[dict]:
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
            logits = self.model(input_ids=ids.to(self.device), attention_mask=mask.to(self.device)).logits
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
