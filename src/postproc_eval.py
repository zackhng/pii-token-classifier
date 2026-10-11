"""v5 no-retraining experiments on SAVED predictions (no model runs):

  per-label thresholds   a predicted span is kept only if its score >= the threshold of its label. Thresholds
                         are tuned on ml_wm_dev (best strict F1 per label, grid 0.30-0.95) and then applied
                         unchanged to the other sets, so ml_wm_dev itself is not a fair test of them.
  checksum audit         how many gold ACCOUNT spans that look like an IBAN pass the ISO 13616 mod-97 check:
                         a checksum post-filter is only testable if the gold values are valid.

    python postproc_eval.py --tags A-v5data SPAN-v5data
Scores use the punctuation trim (v5.0 as released) and merged-address gold for ml_real / ml_mixed.
"""
import argparse
import json
import re
from collections import defaultdict

from common import IGNORE, ROOT
from evaluate import overlaps, prf, score_doc
from address_merge import merge_address_spans
from predict import trim_span

SETS = ["ml_wm_dev", "ml_struct_dev", "ml_synth", "ml_real", "ml_mixed", "ml_kiii_test"]
MERGE = {"ml_real", "ml_mixed"}
GRID = [round(0.30 + 0.05 * i, 2) for i in range(14)]


def load(split, tag):
    gold = [json.loads(l) for l in open(ROOT / "data/processed" / f"{split}.jsonl", encoding="utf-8")]
    P = {}
    for l in open(ROOT / "outputs" / f"{tag}_{split}_predictions.jsonl", encoding="utf-8"):
        x = json.loads(l)
        P[x["id"]] = x["default"]
    docs = []
    for r in gold:
        spans = merge_address_spans(r["text"], r["spans"])[0] if split in MERGE else r["spans"]
        ignore = [s for s in spans if s["label"] == IGNORE]
        pred = [t for t in (trim_span(r["text"], p) for p in P[r["id"]]) if t]
        pred = [p for p in pred if not any(overlaps(p, g) for g in ignore)]
        docs.append(([s for s in spans if s["label"] not in (IGNORE, "O")], pred))
    return docs


def score(docs, thr=None):
    c = defaultdict(lambda: defaultdict(int))
    for gold, pred in docs:
        if thr:
            pred = [p for p in pred if p.get("score", 1.0) >= thr.get(p["label"], 0.0)]
        score_doc(gold, pred, c, lambda lab: ["ALL", f"ent/{lab}"], lenient=False)
    return {k: prf(v) for k, v in c.items()}


def tune(docs, labels):
    thr = {}
    for lab in labels:
        best = max(GRID, key=lambda t: (score(docs, {lab: t}).get(f"ent/{lab}", (0, 0, 0))[2], -t))
        thr[lab] = best
    return thr


IBAN = re.compile(r"^[A-Z]{2}\d{2}[A-Z0-9]{10,30}$")


def mod97_ok(s):
    s = re.sub(r"[\s-]", "", s).upper()
    if not IBAN.match(s):
        return None
    r = s[4:] + s[:4]
    return int("".join(str(int(ch, 36)) for ch in r)) % 97 == 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=["A-v5data", "SPAN-v5data"])
    ap.add_argument("--out", default=str(ROOT / "results/v5_postproc"))
    args = ap.parse_args()
    labels = ["PERSON", "ADDRESS", "ACCOUNT", "BUSINESS", "DOB", "PHONE", "EMAIL", "TIN"]
    res = {}
    for tag in args.tags:
        data = {s: load(s, tag) for s in SETS}
        thr = tune(data["ml_wm_dev"], labels)
        res[tag] = {"thresholds": thr, "base": {s: score(d)["ALL"][2] for s, d in data.items()},
                    "thr": {s: score(d, thr)["ALL"][2] for s, d in data.items()},
                    "per_entity_struct": {lab: (score(data["ml_struct_dev"]).get(f"ent/{lab}", (0, 0, 0))[2],
                                                score(data["ml_struct_dev"], thr).get(f"ent/{lab}", (0, 0, 0))[2])
                                          for lab in labels}}
    # checksum audit over gold accounts
    audit = {}
    for s in SETS:
        n = ok = 0
        for l in open(ROOT / "data/processed" / f"{s}.jsonl", encoding="utf-8"):
            r = json.loads(l)
            for sp in r["spans"]:
                if sp["label"] == "ACCOUNT":
                    v = mod97_ok(r["text"][sp["start"]:sp["end"]])
                    if v is not None:
                        n += 1
                        ok += v
        audit[s] = {"iban_shaped": n, "mod97_valid": ok}
    res["checksum_audit"] = audit
    json.dump(res, open(args.out + ".json", "w", encoding="utf-8"), indent=1)
    for tag in args.tags:
        r = res[tag]
        print(f"\n{tag}  thresholds (tuned on ml_wm_dev): {r['thresholds']}")
        for s in SETS:
            print(f"  {s:14s} base {r['base'][s]:.4f}  thr {r['thr'][s]:.4f}  ({r['thr'][s] - r['base'][s]:+.4f})")
    print("\nchecksum audit (gold ACCOUNT spans that look like an IBAN):", audit)


if __name__ == "__main__":
    main()
