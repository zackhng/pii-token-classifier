"""Span-level P/R/F1 on data/processed/<split>.jsonl, overall / per entity / per source.

Reports strict (exact char boundaries + label) and lenient (any overlap + label) matching.
Predictions overlapping gold IGNORE spans are discarded.
"""
import argparse
import json
from collections import defaultdict

from tqdm import tqdm

from common import IGNORE, ROOT, load_yaml
from data import load_split
from predict import PIIPredictor


def overlaps(a, b):
    return a["start"] < b["end"] and b["start"] < a["end"]


def score_doc(gold, pred, counts, keys, lenient):
    matched = set()
    for p in pred:
        hit = None
        for gi, g in enumerate(gold):
            if gi in matched or g["label"] != p["label"]:
                continue
            ok = overlaps(g, p) if lenient else (g["start"], g["end"]) == (p["start"], p["end"])
            if ok:
                hit = gi
                break
        for k in keys(p["label"]):
            counts[k]["tp" if hit is not None else "fp"] += 1
        if hit is not None:
            matched.add(hit)
    for gi, g in enumerate(gold):
        if gi not in matched:
            for k in keys(g["label"]):
                counts[k]["fn"] += 1


def prf(c):
    p = c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else 0.0
    r = c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def main():
    cfg = load_yaml("train.yaml")
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default=str(ROOT / cfg["output_dir"]))
    ap.add_argument("--split", default="test", help="processed split to score (test / stress)")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out", help="write metrics json here")
    args = ap.parse_args()

    t = cfg["tokenize"]
    predictor = PIIPredictor(args.model_dir, t["max_length"], t["stride"], t["label_all_tokens"])
    ds = load_split(args.split, args.limit)
    strict = defaultdict(lambda: defaultdict(int))
    lenient = defaultdict(lambda: defaultdict(int))

    for row in tqdm(ds, desc="eval"):
        ignore = [s for s in row["spans"] if s["label"] == IGNORE]
        gold = [s for s in row["spans"] if s["label"] != IGNORE]
        pred = [p for p in predictor.predict(row["text"])
                if not any(overlaps(p, g) for g in ignore)]
        src = row["source"]
        keys = lambda lab: ["ALL", f"ent/{lab}", f"src/{src}", f"src/{src}/{lab}"]
        score_doc(gold, pred, strict, keys, lenient=False)
        score_doc(gold, pred, lenient, keys, lenient=True)

    results = {}
    print(f"\n{'key':32s} {'P':>6s} {'R':>6s} {'F1':>6s} {'F1-len':>7s} {'support':>8s}")
    for k in sorted(strict, key=lambda k: (k != "ALL", k.count("/"), k)):
        if k.count("/") > 1:
            continue
        p, r, f = prf(strict[k])
        fl = prf(lenient[k])[2]
        sup = strict[k]["tp"] + strict[k]["fn"]
        results[k] = {"precision": p, "recall": r, "f1": f, "f1_lenient": fl, "support": sup}
        print(f"{k:32s} {p:6.3f} {r:6.3f} {f:6.3f} {fl:7.3f} {sup:8d}")
    per_src_ent = {k: dict(zip(["precision", "recall", "f1"], prf(v)))
                   for k, v in strict.items() if k.count("/") > 1}
    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"summary": results, "per_source_entity": per_src_ent}, fh, indent=2)


if __name__ == "__main__":
    main()
