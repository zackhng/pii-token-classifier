"""How well the model rejects look-alikes, on data/processed/<split>.jsonl + negatives.jsonl.

- look-alikes (amounts, currencies, ordinary dates, PIN/CVV/SWIFT codes): % of gold spans that
  any prediction overlaps, and which label it got
- DOB recall, and DOB predictions that land on an ordinary date
- PERSON <-> BUSINESS swaps: gold spans overlapped by a prediction of the other label
- ACCOUNT recall
"""
import argparse
import json
from collections import Counter, defaultdict

from tqdm import tqdm

from common import IGNORE, ROOT, load_yaml
from evaluate import overlaps
from predict import PIIPredictor


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def main():
    cfg = load_yaml("train.yaml")
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default=str(ROOT / cfg["output_dir"]))
    ap.add_argument("--split", default="test", help="test or stress")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out", help="write metrics json here")
    args = ap.parse_args()

    pdir = ROOT / cfg["data"]["processed_dir"]
    rows = read_jsonl(pdir / f"{args.split}.jsonl")[:args.limit]
    negs = {r["id"]: r["negs"] for r in read_jsonl(pdir / "negatives.jsonl")}
    t = cfg["tokenize"]
    predictor = PIIPredictor(args.model_dir, t["max_length"], t["stride"], t["label_all_tokens"])

    neg_hit = defaultdict(Counter)          # group -> {"n", "flagged", <pred label>...}
    gold_hit = defaultdict(Counter)         # label -> {"n", "found", "swap"}
    swap_of = {"PERSON": "BUSINESS", "BUSINESS": "PERSON"}
    for row in tqdm(rows, desc=f"confusion/{args.split}"):
        pred = predictor.predict(row["text"])
        for n in negs.get(row["id"], []):
            c = neg_hit[n["label"]]
            c["n"] += 1
            hits = [p["label"] for p in pred if overlaps(p, n)]
            if hits:
                c["flagged"] += 1
                c[hits[0]] += 1
        for g in row["spans"]:
            if g["label"] == IGNORE:
                continue
            labs = {p["label"] for p in pred if overlaps(p, g)}
            exact = any(p["label"] == g["label"] and (p["start"], p["end"]) == (g["start"], g["end"])
                        for p in pred)
            keys = [g["label"]]
            if g["label"] == "ADDRESS":   # address breakdown: layout and (address records) country
                keys.append("ADDRESS/multi-line" if "\n" in row["text"][g["start"]:g["end"]]
                            else "ADDRESS/single-line")
                if row["id"].startswith("addr-"):
                    keys.append(f"ADDRESS/{row['id'].split('-')[-2]}")
            for k in keys:
                c = gold_hit[k]
                c["n"] += 1
                c["found"] += g["label"] in labs
                c["exact"] += exact
                c["swap"] += g["label"] not in labs and swap_of.get(g["label"]) in labs

    out = {"lookalikes": {}, "gold": {}}
    print(f"\nLook-alikes tagged as PII ({args.split}) - lower is better")
    print(f"{'group':10s} {'n':>7s} {'tagged':>8s}   as")
    for grp, c in sorted(neg_hit.items()):
        rate = c["flagged"] / c["n"] if c["n"] else 0.0
        as_ = {k: v for k, v in c.items() if k not in ("n", "flagged")}
        out["lookalikes"][grp] = {"n": c["n"], "tagged_rate": rate, "as": as_}
        print(f"{grp:10s} {c['n']:7d} {rate:8.1%}   {dict(Counter(as_).most_common(4))}")

    print(f"\nGold entities: found (any overlap, right label) / exact boundaries / "
          f"swapped PERSON<->BUSINESS")
    print(f"{'label':22s} {'n':>7s} {'found':>8s} {'exact':>8s} {'swapped':>8s}")
    for lab, c in sorted(gold_hit.items()):
        found, exact, swap = c["found"] / c["n"], c["exact"] / c["n"], c["swap"] / c["n"]
        out["gold"][lab] = {"n": c["n"], "found_rate": found, "exact_rate": exact, "swap_rate": swap}
        print(f"{lab:22s} {c['n']:7d} {found:8.1%} {exact:8.1%} {swap:8.1%}")
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(out, fh, indent=2)


if __name__ == "__main__":
    main()
