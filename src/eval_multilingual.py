"""Zero-shot multilingual evaluation on data/processed/ml_{real,synth}.jsonl (build_ml_eval.py).

Every document is predicted with two decodings of the same forward pass:
  default   word label = first sub-token's label (training setting, label_all_tokens=False).
            `word_starts` splits words at spaces / punctuation only, so an unspaced CJK / Thai
            run is ONE word and its first character decides the label of the whole run.
  alltok    every token keeps its own prediction (label_all_tokens=True)
The gap between the two is the decoding artifact, not what the model knows.

Per language: strict / lenient P R F1, per entity, per script class of the gold span
(latin vs nonlatin, recall only - a prediction has no gold script), detection rate (gold span
overlapped by a prediction of any label), label confusion, and look-alikes (amount / date / code)
tagged as PII. Predictions overlapping gold IGNORE spans are discarded, as in evaluate.py.

    python eval_multilingual.py --split ml_synth --model_dir ../outputs/deberta-v3-xsmall-pii-v3 --out ../results/v3_ml_synth
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from tqdm import tqdm

from common import IGNORE, ROOT, load_yaml
from eval_confusion import read_jsonl
from evaluate import overlaps, prf, score_doc
from predict import PIIPredictor

MODES = {"default": False, "alltok": True}
LANG_ORDER = ["en", "zh-Hans", "zh-Hant", "ja", "ko", "hi", "ar", "th", "vi", "ms", "id", "tl"]


def best_overlap(g, preds):
    best, size = None, 0
    for p in preds:
        o = min(g["end"], p["end"]) - max(g["start"], p["start"])
        if o > size:
            best, size = p, o
    return best


def evaluate_mode(rows, preds_by_id):
    strict = defaultdict(lambda: defaultdict(int))
    lenient = defaultdict(lambda: defaultdict(int))
    script_hits = defaultdict(Counter)       # lang/script -> {n, strict, lenient}
    detect = defaultdict(Counter)            # lang/ent -> {n, hit}
    confusion = defaultdict(Counter)         # lang/gold -> {pred label | MISS}
    negs = defaultdict(Counter)              # lang/group -> {n, flagged, <label>}
    for row in rows:
        lang = row["lang"]
        ignore = [s for s in row["spans"] if s["label"] == IGNORE]
        gold = [s for s in row["spans"] if s["label"] != IGNORE]
        pred = [p for p in preds_by_id[row["id"]] if not any(overlaps(p, g) for g in ignore)]
        keys = lambda lab: ["ALL", f"lang/{lang}", f"lang/{lang}/{lab}"]
        score_doc(gold, pred, strict, keys, lenient=False)
        score_doc(gold, pred, lenient, keys, lenient=True)
        for g in gold:
            same = [p for p in pred if p["label"] == g["label"]]
            sk = f"{lang}/{g['script']}"
            script_hits[sk]["n"] += 1
            script_hits[sk]["strict"] += any((p["start"], p["end"]) == (g["start"], g["end"]) for p in same)
            script_hits[sk]["lenient"] += any(overlaps(p, g) for p in same)
            b = best_overlap(g, pred)
            for k in (f"{lang}/{g['label']}", f"{lang}/ALL"):
                detect[k]["n"] += 1
                detect[k]["hit"] += b is not None
            confusion[f"{lang}/{g['label']}"][b["label"] if b else "MISS"] += 1
        for ng in row.get("negs", []):
            k = f"{lang}/{ng['label']}"
            negs[k]["n"] += 1
            b = best_overlap(ng, pred)
            if b:
                negs[k]["flagged"] += 1
                negs[k][b["label"]] += 1

    def table(counts):
        out = {}
        for k, c in counts.items():
            p, r, f = prf(c)
            out[k] = {"precision": p, "recall": r, "f1": f, "support": c["tp"] + c["fn"]}
        return out

    s, l = table(strict), table(lenient)
    for k in s:
        s[k]["f1_lenient"] = l[k]["f1"]
    return {
        "strict": s,
        "script_recall": {k: {"n": c["n"], "strict": c["strict"] / c["n"], "lenient": c["lenient"] / c["n"]}
                          for k, c in script_hits.items()},
        "detection": {k: c["hit"] / c["n"] for k, c in detect.items()},
        "confusion": {k: dict(c) for k, c in confusion.items()},
        "negatives": {k: {"n": c["n"], "flagged": c["flagged"] / c["n"],
                          "as": {lab: v for lab, v in c.items() if lab not in ("n", "flagged")}}
                      for k, c in negs.items()},
    }


def report(res: dict, langs: list[str], entities: list[str]) -> str:
    lines = []
    for mode, r in res.items():
        s = r["strict"]
        lines.append(f"\n=== decoding: {mode}  (strict F1 = exact boundaries + label; lenient = any overlap)")
        lines.append(f"{'lang':8s} {'P':>6s} {'R':>6s} {'F1':>6s} {'F1-len':>7s} {'detect':>7s} "
                     f"{'R-latin':>8s} {'R-nonlat':>9s} {'neg-FP':>7s} {'support':>8s}")
        for lang in ["ALL"] + langs:
            k = "ALL" if lang == "ALL" else f"lang/{lang}"
            if k not in s:
                continue
            m = s[k]
            if lang == "ALL":
                det = lat = non = neg = ""
            else:
                det = f"{r['detection'].get(f'{lang}/ALL', 0):.3f}"
                lat = r["script_recall"].get(f"{lang}/latin")
                non = r["script_recall"].get(f"{lang}/nonlatin")
                lat = f"{lat['strict']:.3f}" if lat else "-"
                non = f"{non['strict']:.3f}" if non else "-"
                ng = [v for kk, v in r["negatives"].items() if kk.startswith(lang + "/")]
                n = sum(v["n"] for v in ng)
                neg = f"{sum(v['flagged'] * v['n'] for v in ng) / n:.3f}" if n else "-"
            lines.append(f"{lang:8s} {m['precision']:6.3f} {m['recall']:6.3f} {m['f1']:6.3f} {m['f1_lenient']:7.3f} "
                         f"{det:>7s} {lat:>8s} {non:>9s} {neg:>7s} {m['support']:8d}")
        lines.append(f"\nstrict F1 per entity ({mode})")
        lines.append(f"{'lang':8s} " + " ".join(f"{e[:8]:>8s}" for e in entities))
        for lang in langs:
            cells = []
            for e in entities:
                m = s.get(f"lang/{lang}/{e}")
                cells.append(f"{m['f1']:8.3f}" if m and m["support"] else f"{'-':>8s}")
            lines.append(f"{lang:8s} " + " ".join(cells))
    r = res["alltok"]
    lines.append("\nwhat the gold spans were predicted as (alltok, largest-overlap prediction)")
    for lang in langs:
        for e in entities:
            c = r["confusion"].get(f"{lang}/{e}")
            if c:
                n = sum(c.values())
                top = ", ".join(f"{k} {v / n:.0%}" for k, v in sorted(c.items(), key=lambda x: -x[1])[:4])
                lines.append(f"  {lang:8s} {e:9s} n={n:5d}  {top}")
    return "\n".join(lines)


def main():
    cfg = load_yaml("train.yaml")
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default=str(ROOT / "outputs/deberta-v3-xsmall-pii-v3"))
    ap.add_argument("--split", default="ml_synth", help="ml_synth or ml_real")
    ap.add_argument("--limit", type=int, help="docs per language")
    ap.add_argument("--out", help="write <out>.json / <out>.txt")
    args = ap.parse_args()

    rows = read_jsonl(ROOT / cfg["data"]["processed_dir"] / f"{args.split}.jsonl")
    if args.limit:
        seen = Counter()
        rows = [r for r in rows if (seen.update([r["lang"]]) or seen[r["lang"]] <= args.limit)]
    langs = [l for l in LANG_ORDER if any(r["lang"] == l for r in rows)]
    entities = load_yaml("label_map.yaml")["entities"]

    t = cfg["tokenize"]
    predictor = PIIPredictor(args.model_dir, t["max_length"], t["stride"], False)
    preds = {m: {} for m in MODES}
    for row in tqdm(rows, desc=f"predict {args.split}"):
        for mode, alltok in MODES.items():   # same weights; only the word-level decoding differs
            predictor.label_all_tokens = alltok
            preds[mode][row["id"]] = predictor.predict(row["text"])

    res = {mode: evaluate_mode(rows, preds[mode]) for mode in MODES}
    txt = f"model: {args.model_dir}\nsplit: {args.split} ({len(rows)} docs)\n" + report(res, langs, entities)
    print(txt)
    if args.out:
        with open(args.out + ".json", "w", encoding="utf-8") as f:
            json.dump({"model": args.model_dir, "split": args.split, "n_docs": len(rows), **res}, f,
                      indent=1, ensure_ascii=False)
        with open(args.out + ".txt", "w", encoding="utf-8") as f:
            f.write(txt + "\n")
        # raw predictions (large, untracked) for qualitative examples
        pred_path = ROOT / "outputs" / (Path(args.out).name + "_predictions.jsonl")
        with open(pred_path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps({"id": row["id"], "lang": row["lang"],
                                    "default": preds["default"][row["id"]], "alltok": preds["alltok"][row["id"]]},
                                   ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
