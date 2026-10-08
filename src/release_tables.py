"""Release tables for the v5 model card: strict / lenient F1 per benchmark and per entity, from saved
predictions (no model runs). Multilingual splits are re-scored from outputs/<tag>_<split>_predictions.jsonl;
English test / stress come from results/<tag>_en_<split>[_mergedgold].json (evaluate.py).

    python release_tables.py --models A A-v5data --out ../results/v5_release_tables
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

from address_merge import merge_address_spans
from common import IGNORE, ROOT, load_yaml
from evaluate import overlaps, prf, score_doc

ENTITIES = load_yaml("label_map.yaml")["entities"]
ML_SPLITS = ["ml_struct_test", "ml_wm_test", "ml_struct_dev", "ml_wm_dev", "ml_synth", "ml_real", "ml_mixed", "ml_kiii_test"]
MERGE_VARIANTS = {"ml_real", "ml_mixed"}          # AI4Privacy-derived gold: also scored with merged addresses


def score_split(split: str, tag: str, merge: bool) -> dict:
    preds_path = ROOT / "outputs" / f"{tag}_{split}_predictions.jsonl"
    if not preds_path.exists():
        return {}
    P = {}
    for line in open(preds_path, encoding="utf-8"):
        x = json.loads(line)
        P[x["id"]] = x["default"]
    strict, lenient = defaultdict(lambda: defaultdict(int)), defaultdict(lambda: defaultdict(int))
    for line in open(ROOT / "data/processed" / f"{split}.jsonl", encoding="utf-8"):
        r = json.loads(line)
        spans = merge_address_spans(r["text"], r["spans"])[0] if merge else r["spans"]
        ignore = [s for s in spans if s["label"] == IGNORE]
        gold = [s for s in spans if s["label"] not in (IGNORE, "O")]
        pred = [p for p in P[r["id"]] if not any(overlaps(p, g) for g in ignore)]
        keys = lambda lab: ["ALL", f"ent/{lab}"]
        score_doc(gold, pred, strict, keys, lenient=False)
        score_doc(gold, pred, lenient, keys, lenient=True)
    out = {}
    for k in strict:
        p, rc, f = prf(strict[k])
        _, _, fl = prf(lenient[k])
        out[k] = {"strict": f, "lenient": fl, "precision": p, "recall": rc, "support": strict[k]["tp"] + strict[k]["fn"]}
    return out


def english(tag: str, split: str, merged: bool) -> dict:
    p = ROOT / "results" / f"{tag}_en_{split}{'_mergedgold' if merged else ''}.json"
    if not p.exists():
        return {}
    s = json.load(open(p, encoding="utf-8"))["summary"]
    return {k: {"strict": v["f1"], "lenient": v["f1_lenient"], "precision": v["precision"], "recall": v["recall"],
                "support": v["support"]} for k, v in s.items() if k == "ALL" or k.startswith("ent/")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["A", "A-v5data"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = {}
    for tag in a.models:
        res[tag] = {}
        for split in ML_SPLITS:
            res[tag][split] = score_split(split, tag, merge=False)
            if split in MERGE_VARIANTS:
                res[tag][split + "_merged"] = score_split(split, tag, merge=True)
        for split in ("test", "stress"):
            res[tag][f"en_{split}"] = english(tag, split, merged=False)
            res[tag][f"en_{split}_merged"] = english(tag, split, merged=True)
    Path(a.out + ".json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    base, new = a.models[0], a.models[-1]
    lines = [f"| Benchmark | {base} strict | {base} lenient | {new} strict | {new} lenient |", "|---|---|---|---|---|"]
    for b in res[new]:
        x, y = res[base].get(b, {}).get("ALL"), res[new].get(b, {}).get("ALL")
        if x and y:
            lines.append(f"| {b} | {x['strict']:.3f} | {x['lenient']:.3f} | {y['strict']:.3f} | {y['lenient']:.3f} |")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
