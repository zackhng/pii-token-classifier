"""ml_wm grid breakdown from saved predictions (no GPU): recall per context x label, per format,
per language setting, and look-alike false positives per context.

eval_multilingual.py --split ml_wm_dev --out ../results/X_ml_wm_dev writes
outputs/X_ml_wm_dev_predictions.jsonl; this reads it next to data/processed/ml_wm_dev.jsonl.
Strict = same label and exact boundaries; lenient = same label, any overlap.

    python eval_wm.py --split ml_wm_dev --pred ../outputs/A_ml_wm_dev_predictions.jsonl --out ../results/A_ml_wm_dev_grid
"""
import argparse
import json
from collections import Counter, defaultdict

from common import ROOT
from evaluate import overlaps
from wm_gen import CONTEXTS, FORMATS, LABELS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="ml_wm_dev")
    ap.add_argument("--pred", required=True)
    ap.add_argument("--mode", default="default", choices=["default", "alltok"])
    ap.add_argument("--out")
    a = ap.parse_args()
    gold = {json.loads(l)["id"]: json.loads(l) for l in open(ROOT / "data/processed" / f"{a.split}.jsonl", encoding="utf-8")}
    preds = {json.loads(l)["id"]: json.loads(l)[a.mode] for l in open(a.pred, encoding="utf-8")}
    hit = defaultdict(Counter)            # key -> n / strict / lenient
    neg = defaultdict(Counter)            # ctx -> n / flagged
    for rid, row in gold.items():
        P = preds[rid]
        setting = f"{row['lang']}/{row['mode']}"
        for g in row["spans"]:
            if g["ctx"] == "bracket_anchor":
                continue
            same = [p for p in P if p["label"] == g["label"]]
            st = any((p["start"], p["end"]) == (g["start"], g["end"]) for p in same)
            le = any(overlaps(p, g) for p in same)
            for k in (("ctx", g["ctx"], g["label"]), ("ctx", g["ctx"], "ALL"), ("label", g["label"]), ("format", row["format"]),
                      ("setting", setting), ("mode", row["mode"]), ("ALL",)):
                hit[k]["n"] += 1
                hit[k]["strict"] += st
                hit[k]["lenient"] += le
        for ng in row["negs"]:
            for k in (ng.get("ctx", "?"), "ALL"):
                neg[k]["n"] += 1
                neg[k]["flagged"] += any(overlaps(p, ng) for p in P)
    r = lambda k, m="strict": hit[k][m] / hit[k]["n"] if hit[k]["n"] else float("nan")
    lines = [f"split {a.split}  predictions {a.pred}  decoding {a.mode}",
             f"overall recall strict {r(('ALL',)):.3f}  lenient {r(('ALL',), 'lenient'):.3f}  (n={hit[('ALL',)]['n']})",
             "", "recall (strict) by context x label"]
    lines.append(f"{'context':10s} " + " ".join(f"{l[:8]:>8s}" for l in LABELS) + f" {'ALL':>8s}")
    for c in CONTEXTS:
        lines.append(f"{c:10s} " + " ".join(f"{r(('ctx', c, l)):8.3f}" for l in LABELS) + f" {r(('ctx', c, 'ALL')):8.3f}")
    lines += ["", "recall (strict) by format: " + "  ".join(f"{f} {r(('format', f)):.3f}" for f in FORMATS)]
    lines += ["recall (strict) by mode: " + "  ".join(f"{m} {r(('mode', m)):.3f}" for m in ("mono", "cs_sentence", "cs_document"))]
    lines += ["", "recall (strict) by language setting"]
    for k in sorted(k for k in hit if k[0] == "setting"):
        lines.append(f"  {k[1]:22s} {r(k):.3f}  (n={hit[k]['n']})")
    lines += ["", "look-alikes tagged as PII, by context: " +
              "  ".join(f"{k} {v['flagged'] / v['n']:.3f}" for k, v in sorted(neg.items()) if v["n"])]
    txt = "\n".join(lines)
    print(txt)
    if a.out:
        with open(a.out + ".txt", "w", encoding="utf-8") as f:
            f.write(txt + "\n")
        with open(a.out + ".json", "w", encoding="utf-8") as f:
            json.dump({"recall": {"|".join(k): dict(v) for k, v in hit.items()},
                       "lookalikes": {k: dict(v) for k, v in neg.items()}}, f, indent=1)


if __name__ == "__main__":
    main()
