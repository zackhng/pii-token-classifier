"""Entity-vs-context difficulty of the training data (v5, plan 4e).

Implements the measurement of Ma et al., "Towards Building More Robust NER Datasets: An Empirical
Study on NER Dataset Bias from a Dataset Difficulty View" (EMNLP 2023), whose code is no longer
online, on top of V-information / PVI (Ethayarajh et al., 2022):

  entity-only model   input = the mention text alone                         -> PVI_E
  context-only model  input = +-CONTEXT chars around it, mention -> <mask>  -> PVI_C
  null model          input = nothing; it converges to the class prior, so p(y | null) = prior
  PVI(x -> y) = -log2 p_null(y) + log2 p_model(y | x);   CEIM = PVI_E - PVI_C

Classes: the 8 entity types + LOOKALIKE (explicit O spans: amounts, codes, dates, postcodes ...).
PVI is cross-fitted (2 folds): each mention is scored by a model trained on the other fold.
CEIM bands follow the paper: low < -0.5 (context easier), |CEIM| <= 0.5 near-zero, high > 0.5.
Only low-CEIM mentions taught robustness in the paper; their share is the v5 frames knob.

    python difficulty.py --split ml_train --per_cell 1000 --out ../results/difficulty_ml_train
"""
import argparse
import json
import math
import random
from collections import Counter, defaultdict

import numpy as np
import torch
from datasets import Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                          Trainer, TrainingArguments)

from common import IGNORE, ROOT, load_yaml

CONTEXT = 160
LOOKALIKE = "LOOKALIKE"


def mentions(split: str, per_cell: int, seed: int) -> list[dict]:
    """Stratified sample: up to per_cell mentions per (lang, label)."""
    cells = defaultdict(list)
    for line in open(ROOT / "data/processed" / f"{split}.jsonl", encoding="utf-8"):
        r = json.loads(line)
        for s in r["spans"]:
            if s["label"] == IGNORE:
                continue
            lab = LOOKALIKE if s["label"] == "O" else s["label"]
            t = r["text"]
            a, b = max(0, s["start"] - CONTEXT), min(len(t), s["end"] + CONTEXT)
            cells[(r["lang"], lab)].append({
                "lang": r["lang"], "label": lab, "source": r["source"], "doc": r["id"],
                "entity": t[s["start"]:s["end"]],
                "context": t[a:s["start"]] + "<mask>" + t[s["end"]:b]})
    rng = random.Random(seed)
    out = []
    for k in sorted(cells):
        v = cells[k]
        rng.shuffle(v)
        out += v[:per_cell]
    return out


def fit_predict(train_rows, test_rows, field, labels, model_name, out_dir, epochs, max_len):
    """Train a classifier on train_rows[field] and return p(gold | x) for test_rows."""
    tok = AutoTokenizer.from_pretrained(model_name)
    l2i = {l: i for i, l in enumerate(labels)}
    enc = lambda rows: Dataset.from_dict({"text": [r[field] for r in rows], "label": [l2i[r["label"]] for r in rows]}).map(
        lambda b: tok(b["text"], truncation=True, max_length=max_len), batched=True, remove_columns=["text"])
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(labels), dtype=torch.float32)
    args = TrainingArguments(output_dir=str(out_dir), num_train_epochs=epochs, learning_rate=3e-5,
                             per_device_train_batch_size=64, per_device_eval_batch_size=256, warmup_steps=0.1,   # 5.x: <1 = ratio
                             weight_decay=0.01, bf16=torch.cuda.is_available(), save_strategy="no", logging_steps=200,
                             report_to=[], seed=42)
    tr = Trainer(model=model, args=args, train_dataset=enc(train_rows), data_collator=DataCollatorWithPadding(tok))
    tr.train()
    logits = tr.predict(enc(test_rows)).predictions
    p = torch.softmax(torch.tensor(logits, dtype=torch.float32), -1).numpy()
    return p[np.arange(len(test_rows)), [l2i[r["label"]] for r in test_rows]]


def summarise(rows: list[dict]) -> dict:
    def agg(group):
        g = defaultdict(list)
        for r in rows:
            g[group(r)].append(r)
        out = {}
        for k, v in sorted(g.items(), key=lambda x: str(x[0])):
            ce = [r["ceim"] for r in v]
            out["/".join(k) if isinstance(k, tuple) else k] = {
                "n": len(v), "V_entity": float(np.mean([r["pvi_e"] for r in v])),
                "V_context": float(np.mean([r["pvi_c"] for r in v])), "CEIM": float(np.mean(ce)),
                "low": sum(c < -0.5 for c in ce) / len(v), "near_zero": sum(abs(c) <= 0.5 for c in ce) / len(v),
                "high": sum(c > 0.5 for c in ce) / len(v)}
        return out
    return {"all": agg(lambda r: "ALL")["ALL"], "by_label": agg(lambda r: r["label"]),
            "by_lang": agg(lambda r: r["lang"]), "by_lang_label": agg(lambda r: (r["lang"], r["label"])),
            "by_source": agg(lambda r: r["source"])}


def report(s: dict) -> str:
    lines = [f"{'group':28s} {'n':>6s} {'V_ent':>7s} {'V_ctx':>7s} {'CEIM':>6s} {'low':>6s} {'nz':>6s} {'high':>6s}"]
    for sec in ("all", "by_label", "by_lang", "by_source"):
        items = {"ALL": s["all"]} if sec == "all" else s[sec]
        lines.append(f"-- {sec}")
        for k, m in items.items():
            lines.append(f"{k[:28]:28s} {m['n']:6d} {m['V_entity']:7.2f} {m['V_context']:7.2f} {m['CEIM']:6.2f} "
                         f"{m['low']:6.1%} {m['near_zero']:6.1%} {m['high']:6.1%}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="ml_train")
    ap.add_argument("--per_cell", type=int, default=1000, help="mentions per (language, label)")
    ap.add_argument("--model", default="xlm-roberta-base")
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    seed = load_yaml("train.yaml")["seed"]

    rows = mentions(args.split, args.per_cell, seed)
    labels = sorted({r["label"] for r in rows})
    prior = Counter(r["label"] for r in rows)
    print(f"{len(rows)} mentions, classes {dict(prior)}")
    # 2-fold cross-fitting, folds by document so a doc's mentions never straddle train / score
    docs = sorted({r["doc"] for r in rows})
    random.Random(seed).shuffle(docs)
    fold_of = {d: i % 2 for i, d in enumerate(docs)}
    for field, key, max_len in (("entity", "p_e", 48), ("context", "p_c", 160)):
        for k in (0, 1):
            tr = [r for r in rows if fold_of[r["doc"]] != k]
            te = [r for r in rows if fold_of[r["doc"]] == k]
            p = fit_predict(tr, te, field, labels, args.model, ROOT / f"outputs/difficulty_tmp_{field}_{k}",
                            args.epochs, max_len)
            for r, v in zip(te, p):
                r[key] = float(v)
    n = len(rows)
    for r in rows:
        null = -math.log2(prior[r["label"]] / n)
        r["pvi_e"] = null + math.log2(max(r["p_e"], 1e-12))
        r["pvi_c"] = null + math.log2(max(r["p_c"], 1e-12))
        r["ceim"] = r["pvi_e"] - r["pvi_c"]
    s = summarise(rows)
    txt = f"split {args.split}: {n} mentions ({args.per_cell} per language x label)\n" + report(s)
    print(txt)
    with open(args.out + ".json", "w", encoding="utf-8") as f:
        json.dump({"split": args.split, "per_cell": args.per_cell, "model": args.model, **s}, f, indent=1, ensure_ascii=False)
    with open(args.out + ".txt", "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    with open(ROOT / "outputs" / (args.out.split("/")[-1] + "_mentions.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
