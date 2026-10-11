"""README experiment log (v5 series): every experiment with strict F1 and coverage, rebuilt from saved
predictions and logs (no model runs). Rewrites the block between the markers in README.md:

    <!-- v5-experiments:start --> ... <!-- v5-experiments:end -->

    python experiments_table.py

Scoring for the multilingual sets: strict F1 = exact span + label with the punctuation trim on (v5.0 as
released) and merged-address gold for ml_real / ml_mixed; coverage = share of gold PII spans overlapped by
any predicted span (any label). English test / stress: strict F1 from evaluate.py, untrimmed (the only run
every model has). "–" = not run, not scored on that set, or not recoverable (predictions not saved).
"""
import json
import re
from collections import defaultdict
from functools import lru_cache

from address_merge import merge_address_spans
from common import IGNORE, ROOT
from evaluate import overlaps, prf, score_doc
from predict import trim_span

SETS = ["ml_struct_dev", "ml_wm_dev", "ml_synth", "ml_real", "ml_mixed", "ml_kiii_test"]
SHORT = {"ml_struct_dev": "struct", "ml_wm_dev": "wm", "ml_synth": "synth", "ml_real": "real",
         "ml_mixed": "mixed", "ml_kiii_test": "Kiii"}
MERGE = {"ml_real", "ml_mixed"}
START, END = "<!-- v5-experiments:start -->", "<!-- v5-experiments:end -->"

# (group, id, prediction/result tag, train-log name, one change vs parent, parent, status)
DONE, RUN, QUEUED, PLAN, STOP = "done", "running", "queued", "planned", "dropped"
EXPS = [
    ("Baselines", "v4.1", "A", "ml_A", "v4 release (Model A): XLM-R base, v4 data", "–", DONE),
    ("Baselines", "v4.2", "B64", "ml_B", "v4 release (Model B): per-language adapters", "–", DONE),
    ("Baselines", "v4 MiniLM", "A-minilm", "ml_minilm", "v4 data, encoder Multilingual-MiniLM-L12", "v4.1", DONE),
    ("Stage 1: data", "pilot r1 0% frames", "pilot_f00", "pilot_f00", "50k pilot, old frames, share 0%", "–", DONE),
    ("Stage 1: data", "pilot r1 10%", "pilot_f10", "pilot_f10", "frame share 10%", "pilot r1 0%", DONE),
    ("Stage 1: data", "pilot r1 25%", "pilot_f25", "pilot_f25", "frame share 25%", "pilot r1 0%", DONE),
    ("Stage 1: data", "pilot r1 40%", "pilot_f40", "pilot_f40", "frame share 40%", "pilot r1 0%", DONE),
    ("Stage 1: data", "pilot r2 0% frames", "pilot2_f00", "pilot2_f00", "50k pilot, frames + wm grid, share 0%", "–", DONE),
    ("Stage 1: data", "pilot r2 10%", "pilot2_f10", "pilot2_f10", "frame share 10%", "pilot r2 0%", DONE),
    ("Stage 1: data", "pilot r2 25% (chosen)", "pilot2_f25", "pilot2_f25", "frame share 25%", "pilot r2 0%", DONE),
    ("Stage 1: data", "pilot r2 40%", "pilot2_f40", "pilot2_f40", "frame share 40%", "pilot r2 0%", DONE),
    ("Stage 1: data", "**v5.0** (A-v5data)", "A-v5data", "A_v5data", "data: ml_train_v5 (v4.1 recipe)", "v4.1", DONE),
    ("Stage 2: what trains", "H1", "H1-v5data", "H1-v5data", "encoder frozen, head only (LR 3e-3)", "v5.0", DONE),
    ("Stage 2: what trains", "P1", "P1-v5data", "P1-v5data", "top 4 layers + head (LR 1e-4)", "v5.0", DONE),
    ("Stage 2: what trains", "S1", "S1-v5data", "S1-v5data", "frozen encoder + one shared adapter (LR 3e-4)", "v5.0", DONE),
    ("Stage 2: what trains", "B1", "B1-v5data", "B1_v5data", "frozen encoder + per-language adapters (LR 1e-4)", "v5.0", DONE),
    ("Stage 2: what trains", "B1 lr 3e-4", "B1-v5data-lr3e-4", "B1-v5data-lr3e-4", "LR 1e-4 → 3e-4", "B1", DONE),
    ("Stage 2: what trains", "B′", "Bprime-v5data", "Bprime_v5data", "per-language adapters on v5.0's encoder", "B1", DONE),
    ("Stage 2: what trains", "B′ fresh head", "–", "–", "B′ with the classifier re-initialised", "B′", "stopped at 28% (B series dropped)"),
    ("Stage 2: what trains", "B2 (DAPT + adapters)", "–", "–", "per-language adapters on a financial-DAPT encoder", "B1", "dropped (B series)"),
    ("Stage 2: what trains", "DAPT row (H2 / P2 / A2)", "–", "–", "multilingual financial DAPT before fine-tuning", "v5.0", "planned, not built"),
    ("Deep head", "AL2", "AL2-v5data", "AL2-v5data", "+2 transformer layers before the classifier (head LR 1e-4)", "v5.0", DONE),
    ("Deep head", "AL4", "AL4-v5data", "AL4-v5data", "+4 layers", "v5.0", DONE),
    ("Deep head", "AL8", "AL8-v5data", "AL8-v5data", "+8 layers", "v5.0", DONE),
    ("Deep head", "AL16", "–", "–", "+16 layers", "v5.0", "cancelled by user"),
    ("Deep head", "HL2 / HL4 / HL8", "–", "–", "frozen encoder + 2/4/8 layers", "H1", "dropped (head-only not promising)"),
    ("Heads", "CRF", "CRF-v5data", "CRF-v5data", "CRF on the classifier (transitions LR 5e-3)", "v5.0", DONE),
    ("Heads", "**SPAN**", "SPAN-v5data", "SPAN-v5data", "GlobalPointer span head (head LR 1e-4)", "v5.0", DONE),
    ("Heads", "SPAN-CE", "SPAN-CE-v5data", "SPAN-CE-v5data", "span loss → per-span softmax CE", "SPAN", DONE),
    ("Heads", "SPAN-BS", "SPAN-BS-v5data", "SPAN-BS-v5data", "boundary smoothing ε 0.1, D 1", "SPAN-CE", DONE),
    ("Encoders (softmax head)", "mmBERT-small", "MMBS-v5data", "MMBS-v5data", "encoder → mmBERT-small", "v5.0", DONE),
    ("Encoders (softmax head)", "mmBERT-base", "MMBB-v5data", "MMBB-v5data", "encoder → mmBERT-base", "v5.0", DONE),
    ("Encoders (softmax head)", "EuroBERT-210m", "–", "–", "encoder → EuroBERT-210m", "v5.0", "dropped by user"),
    ("Encoders (softmax head)", "Qwen3-Embedding-0.6B", "–", "–", "encoder → Qwen3-Embedding-0.6B", "v5.0", "dropped by user"),
    ("Encoders (softmax head)", "MiniLM-L12", "MML12-v5data", "MML12-v5data", "encoder → mMiniLMv2-L12-H384", "v5.0", DONE),
    ("Encoders (softmax head)", "MiniLM-L6", "MML6-v5data", "MML6-v5data", "encoder → mMiniLMv2-L6-H384", "v5.0", DONE),
    ("Encoders + span", "mmBERT-small + span", "MMBS-SPAN-v5data", "MMBS-SPAN-v5data", "span head", "mmBERT-small", DONE),
    ("Encoders + span", "mmBERT-base + span", "–", "–", "span head", "mmBERT-base", "cancelled by user"),
    ("Encoders + span", "MiniLM-L12 + span", "MML12-SPAN-v5data", "MML12-SPAN-v5data", "span head", "MiniLM-L12", DONE),
    ("Encoders + span", "MiniLM-L6 + span", "MML6-SPAN-v5data", "MML6-SPAN-v5data", "span head", "MiniLM-L6", DONE),
    ("Encoders + boundary-smoothed span", "MiniLM-L12 + span-BS", "MML12-BS-v5data", "MML12-BS-v5data", "head → SPAN-BS head (CE loss + boundary smoothing)", "MiniLM-L12 + span", DONE),
    ("Encoders + boundary-smoothed span", "MiniLM-L6 + span-BS", "MML6-BS-v5data", "MML6-BS-v5data", "head → SPAN-BS head (CE loss + boundary smoothing)", "MiniLM-L6 + span", DONE),
    ("Encoders + boundary-smoothed span", "mDeBERTa-v3-base + span-BS", "MDEB-BS-v5data", "MDEB-BS-v5data", "encoder → mDeBERTa-v3-base", "SPAN-BS", QUEUED),
    ("Encoders + boundary-smoothed span", "InfoXLM-base + span-BS", "INFO-BS-v5data", "INFO-BS-v5data", "encoder → InfoXLM-base", "SPAN-BS", DONE),
    ("Encoders + boundary-smoothed span", "XLM-V-base + span-BS", "XLMV-BS-v5data", "XLMV-BS-v5data", "encoder → XLM-V-base", "SPAN-BS", RUN),
    ("Encoders + boundary-smoothed span", "XLM-R large + span-BS", "XLMRL-BS-v5data", "XLMRL-BS-v5data", "encoder → XLM-R large", "SPAN-BS", QUEUED),
    ("Inference only", "v5.0 merge central", "A-v5data_merge-central", "–", "overlapping windows: most central window", "v5.0", DONE),
    ("Inference only", "v5.0 merge avg", "A-v5data_merge-avg", "–", "overlapping windows: mean probabilities", "v5.0", DONE),
    ("Inference only", "v5.0 overlap 64", "A-v5data_stride64", "–", "window overlap 96 → 64", "v5.0", DONE),
    ("Inference only", "SPAN merge central", "SPAN-v5data_merge-central", "–", "most central window", "SPAN", DONE),
    ("Inference only", "SPAN merge avg", "SPAN-v5data_merge-avg", "–", "span scores averaged over windows", "SPAN", DONE),
    ("Inference only", "SPAN overlap 64", "SPAN-v5data_stride64", "–", "window overlap 96 → 64", "SPAN", DONE),
    ("Inference only", "v5.0 + label thresholds", "thr:A-v5data", "–", "per-label score thresholds (tuned on wm)", "v5.0", DONE),
    ("Inference only", "SPAN + label thresholds", "thr:SPAN-v5data", "–", "per-label score thresholds (tuned on wm)", "SPAN", DONE),
    ("Inference only", "checksum post-filter", "–", "–", "drop IBAN-shaped accounts failing mod-97", "SPAN", "not testable (synthetic IBANs invalid)"),
    ("Inference only", "int8 (all Linear)", "–", "–", "dynamic int8 encoder, CPU (300-doc sample: F1 ≈ 0.50)", "v5.0", "paused by user"),
    ("Validation", "seed runs (v5.0, SPAN × 2 seeds)", "–", "–", "seed 42 → 2 more seeds", "v5.0 / SPAN", "proposed, not queued"),
]


@lru_cache(None)
def gold(split):
    out = {}
    for l in open(ROOT / "data/processed" / f"{split}.jsonl", encoding="utf-8"):
        r = json.loads(l)
        spans = merge_address_spans(r["text"], r["spans"])[0] if split in MERGE else r["spans"]
        out[r["id"]] = (r["text"], [s for s in spans if s["label"] == IGNORE],
                        [s for s in spans if s["label"] not in (IGNORE, "O")])
    return out


# ml_struct was renamed ml_struct_dev (same 3,084 documents: v4.1's predictions are identical under both names)
LEGACY = {"ml_struct_dev": "ml_struct"}


def score(tag, split, thr=None):
    """(strict F1, coverage); thr = per-label minimum span score (threshold rows)."""
    path = ROOT / "outputs" / f"{tag}_{split}_predictions.jsonl"
    if not path.exists() and split in LEGACY:
        path = ROOT / "outputs" / f"{tag}_{LEGACY[split]}_predictions.jsonl"
    if not path.exists() or not (ROOT / "data/processed" / f"{split}.jsonl").exists():
        return None
    G = gold(split)
    c = defaultdict(lambda: defaultdict(int))
    cov = n = 0
    for l in open(path, encoding="utf-8"):
        x = json.loads(l)
        if x["id"] not in G:
            return None
        text, ignore, g = G[x["id"]]
        pred = [t for t in (trim_span(text, p) for p in x["default"]) if t]
        pred = [p for p in pred if not any(overlaps(p, i) for i in ignore)]
        if thr:
            pred = [p for p in pred if p.get("score", 1.0) >= thr.get(p["label"], 0.0)]
        score_doc(g, pred, c, lambda lab: ["ALL"], lenient=False)
        n += len(g)
        cov += sum(any(overlaps(s, p) for p in pred) for s in g)
    return prf(c["ALL"])[2], (cov / n if n else None)


def val_f1(log):
    p = ROOT / "outputs" / f"train_{log}.log"
    if log == "–" or not p.exists():
        return None
    v = [float(x) for x in re.findall(r"'eval_macro_f1': '?([0-9.]+)", p.read_text(encoding="utf-8", errors="ignore"))]
    return max(v) if v else None


def english(tag, split):
    p = ROOT / "results" / f"{tag}_en_{split}.txt"
    if not p.exists():
        return None
    m = re.search(r"^ALL\s+\S+\s+\S+\s+(\S+)", p.read_text(encoding="utf-8", errors="ignore"), re.M)
    return float(m.group(1)) if m else None


def main():
    thr = {}
    pp = ROOT / "results/v5_postproc.json"
    if pp.exists():
        thr = json.load(open(pp, encoding="utf-8"))
    f = lambda v: "–" if v is None else f"{v:.3f}"
    head = ["Group", "Experiment", "One change vs parent", "Parent", "Status"]
    strict = ["| " + " | ".join(head + [SHORT[s] for s in SETS] + ["EN test", "EN stress", "val macro"]) + " |",
              "|" + "---|" * (len(head) + len(SETS) + 3)]
    cover = ["| " + " | ".join(["Group", "Experiment", "Status"] + [SHORT[s] for s in SETS]) + " |",
             "|" + "---|" * (3 + len(SETS))]
    notrun = ["| Group | Experiment | Change vs parent | Parent | Why not run |", "|---|---|---|---|---|"]
    for grp, name, tag, log, change, parent, status in EXPS:
        if status not in (DONE, RUN, QUEUED):          # dropped / cancelled / on hold / proposed: own table
            notrun.append(f"| {grp} | {name} | {change} | {parent} | {status} |")
            continue
        if tag.startswith("thr:"):
            t = tag[4:]
            r = [score(t, s, thr.get(t, {}).get("thresholds")) for s in SETS]
            st = [x[0] if x else None for x in r]
            cv = [x[1] if x else None for x in r]
            en = [None, None]
        elif tag == "–":
            st, cv, en = [None] * len(SETS), [None] * len(SETS), [None, None]
        else:
            r = [score(tag, s) for s in SETS]
            st = [x[0] if x else None for x in r]
            cv = [x[1] if x else None for x in r]
            en = [english(tag, "test"), english(tag, "stress")]
        strict.append("| " + " | ".join([grp, name, change, parent, status] + [f(v) for v in st + en]
                                       + [f(val_f1(log))]) + " |")
        cover.append("| " + " | ".join([grp, name, status] + [f(v) for v in cv]) + " |")
    block = "\n".join([
        START,
        "## v5 experiment log",
        "",
        "Every v5 experiment that has run, is running or is queued; ideas that were dropped, cancelled or not yet "
        "queued are in a separate table at the end. Each row changes **one thing** relative to its "
        "parent. Choices were made on the dev sets only; the held-out test sets (ml_struct_test, ml_wm_test) "
        "were used once, for v5.0. Single seed (42) throughout. Regenerate with `cd src && python experiments_table.py`.",
        "",
        "- **struct** = ml_struct_dev (unseen sentence structures), **wm** = ml_wm_dev (wealth-management grid, 12 "
        "languages + English–X), **synth** = ml_synth, **real** = ml_real (public data), **mixed** = ml_mixed "
        "(code-switched public data), **Kiii** = Korean financial documents.",
        "- Multilingual sets: punctuation trim on, merged-address gold for real / mixed, scored from saved predictions. "
        "English test / stress: evaluate.py, **untrimmed** (trim adds about +0.015 on English test for XLM-R-tokenizer "
        "models, e.g. v5.0 .894 → .911, SPAN .907 → .923). val macro = best ml_val_v5 macro-F1 during training "
        "(pilots / v4 rows: their own validation set).",
        "- Pilot rows train on a 50k-document pilot set; LR-sweep pilots are not listed. Older runs scored on "
        "`ml_struct` (the original name of ml_struct_dev, same 3,084 documents) are re-scored against today's gold.",
        "- Threshold rows: the per-label thresholds tuned on wm (results/v5_postproc.json) applied to the saved "
        "predictions; their wm column is the tuning set, so not a fair test. English needs a model run, so –.",
        "- Model names (AL, SPAN, SPAN-BS, …): see **Model names** in the suggestions section above.",
        "- **–** = not run, not scored on that set, or not recoverable (predictions not saved).",
        "",
        "### Strict F1",
        "",
        *strict,
        "",
        "### Coverage (share of gold PII spans overlapped by any prediction)",
        "",
        *cover,
        "",
        "### Not run (dropped, cancelled, on hold or proposed)",
        "",
        *notrun,
        END,
    ])
    readme = ROOT / "README.md"
    text = readme.read_text(encoding="utf-8")
    if START in text:
        text = text[:text.index(START)] + block + text[text.index(END) + len(END):]
    else:
        anchor = "\n## Labels"
        text = text.replace(anchor, "\n" + block + "\n" + anchor, 1)
    readme.write_text(text, encoding="utf-8")
    print(block)
    from recommend import write_readme      # suggestions block, built from the same scores
    write_readme()


if __name__ == "__main__":
    main()
