"""v5 candidate breakdown from SAVED predictions (no model runs): strict F1 and coverage per dataset, entity
and language.

  strict F1  exact span and label (v5 scoring: punctuation trim on, merged-address gold for ml_real / ml_mixed)
  coverage   share of gold PII spans that overlap ANY predicted span, whatever its label: the privacy view
             (would this text have been masked?)

    python breakdown.py --tags A-v5data SPAN-v5data ... --out ../results/v5_breakdown
English test / stress are scored by evaluate.py (no saved predictions): strict F1 per entity only.
"""
import argparse
import json
from collections import defaultdict

from address_merge import merge_address_spans
from common import IGNORE, ROOT
from results_layout import find, result_path
from evaluate import overlaps, prf, score_doc
from predict import trim_span

SETS = ["ml_struct_dev", "ml_wm_dev", "ml_synth", "ml_real", "ml_mixed", "ml_kiii_test"]
MERGE = {"ml_real", "ml_mixed"}
ENTS = ["PERSON", "ADDRESS", "ACCOUNT", "BUSINESS", "DOB", "PHONE", "EMAIL", "TIN"]
LANGS = ["en", "zh-Hans", "zh-Hant", "ja", "ko", "hi", "ar", "th", "vi", "ms", "id", "tl"]


def counts_for(tag):
    """{(set, key): {"tp","fp","fn","cov","n"}} with key ALL / ent/X / lang/L."""
    c = defaultdict(lambda: defaultdict(int))
    for split in SETS:
        path = ROOT / "outputs" / f"{tag}_{split}_predictions.jsonl"
        if not path.exists():
            continue
        P = {}
        for l in open(path, encoding="utf-8"):
            x = json.loads(l)
            P[x["id"]] = x["default"]
        for l in open(ROOT / "data/processed" / f"{split}.jsonl", encoding="utf-8"):
            r = json.loads(l)
            spans = merge_address_spans(r["text"], r["spans"])[0] if split in MERGE else r["spans"]
            ignore = [s for s in spans if s["label"] == IGNORE]
            gold = [s for s in spans if s["label"] not in (IGNORE, "O")]
            pred = [t for t in (trim_span(r["text"], p) for p in P[r["id"]]) if t]
            pred = [p for p in pred if not any(overlaps(p, g) for g in ignore)]
            lang = r.get("lang", "?")
            keys = lambda lab: [(split, "ALL"), (split, f"ent/{lab}"), (split, f"lang/{lang}"),
                                ("ALLSETS", "ALL"), ("ALLSETS", f"ent/{lab}"), ("ALLSETS", f"lang/{lang}")]
            tmp = defaultdict(lambda: defaultdict(int))
            score_doc(gold, pred, tmp, keys, lenient=False)
            for k, v in tmp.items():
                for kk, vv in v.items():
                    c[k][kk] += vv
            for g in gold:
                hit = any(overlaps(g, p) for p in pred)
                for k in keys(g["label"]):
                    c[k]["n"] += 1
                    c[k]["cov"] += hit
    return c


def cell(c, key):
    v = c.get(key)
    if not v or not v["n"]:
        return None
    return prf(v)[2], v["cov"] / v["n"], v["n"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--out", default=str(ROOT / "results/v5_breakdown"))
    args = ap.parse_args()
    if args.out:
        args.out = result_path(args.out)   # flat results/ path -> its group folder
    C = {t: counts_for(t) for t in args.tags}
    names = [t.replace("-v5data", "") for t in args.tags]
    lines = ["# v5 candidates: strict F1 / coverage", "",
             "Strict F1 = exact span and label (trim on; merged-address gold for ml_real / ml_mixed). "
             "Coverage = share of gold PII spans overlapped by any predicted span (any label). "
             "Each cell: `strict F1 / coverage`.", ""]

    def table(title, rows, setname):
        lines.extend([f"## {title}", "", "| | n | " + " | ".join(names) + " |", "|---|---|" + "---|" * len(names)])
        for label, key in rows:
            vals = [cell(C[t], (setname, key)) for t in args.tags]
            n = next((v[2] for v in vals if v), 0)
            if not n:
                continue
            best = max(v[0] for v in vals if v)
            fmt = lambda v: "–" if not v else (("**%.3f**" if v[0] == best else "%.3f") % v[0]) + " / %.3f" % v[1]
            lines.append(f"| {label} | {n} | " + " | ".join(fmt(v) for v in vals) + " |")
        lines.append("")

    table("Per dataset", [(s, "ALL") for s in SETS], None) if False else None
    lines.extend(["## Per dataset", "", "| Dataset | n | " + " | ".join(names) + " |", "|---|---|" + "---|" * len(names)])
    for s in SETS + ["ALLSETS"]:
        vals = [cell(C[t], (s, "ALL")) for t in args.tags]
        n = next((v[2] for v in vals if v), 0)
        if not n:
            continue
        best = max(v[0] for v in vals if v)
        lines.append(f"| {'all six sets' if s == 'ALLSETS' else s} | {n} | " + " | ".join(
            "–" if not v else (("**%.3f**" if v[0] == best else "%.3f") % v[0]) + " / %.3f" % v[1] for v in vals) + " |")
    lines.append("")
    table("Per entity (all six sets pooled)", [(e, f"ent/{e}") for e in ENTS], "ALLSETS")
    table("Per language (all six sets pooled)", [(l, f"lang/{l}") for l in LANGS], "ALLSETS")
    for s in SETS:
        table(f"{s}: per entity", [(e, f"ent/{e}") for e in ENTS], s)
        table(f"{s}: per language", [(l, f"lang/{l}") for l in LANGS], s)
    # English (evaluate.py, trimmed runs where available): strict F1 per entity
    lines.extend(["## English test / stress (strict F1 per entity, trim on; coverage not available)", ""])
    for split in ("test", "stress"):
        lines.extend([f"### en_{split}", "", "| Entity | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)])
        D = {}
        for t in args.tags:
            for suf in ("_trim", ""):
                p = find(f"{t}_en_{split}{suf}.json")
                if p.exists():
                    D[t] = json.load(open(p, encoding="utf-8"))["summary"]
                    break
        for k in ["ALL"] + [f"ent/{e}" for e in ENTS]:
            v = [D.get(t, {}).get(k, {}).get("f1") for t in args.tags]
            if all(x is None for x in v):
                continue
            best = max(x for x in v if x is not None)
            lines.append(f"| {k.replace('ent/', '')} | " + " | ".join(
                "–" if x is None else ("**%.3f**" if x == best else "%.3f") % x for x in v) + " |")
        lines.append("")
    pathlib_out = args.out + ".md"
    open(pathlib_out, "w", encoding="utf-8").write("\n".join(lines))
    print("\n".join(lines[:60]))


if __name__ == "__main__":
    main()
