"""Fine-tune DeBERTa-v3-xsmall for BIO PII token classification."""
import argparse

import torch
from seqeval.metrics import f1_score, precision_score, recall_score
from transformers import (AutoModelForTokenClassification, AutoTokenizer,
                          DataCollatorForTokenClassification, Trainer, TrainingArguments,
                          set_seed)

from common import ROOT, label_maps, load_yaml
from data import load_ml_split, load_split, tokenize


def build_compute_metrics(id2label, window_langs=None):
    """seqeval P/R/F1; with `window_langs` (one per eval window, in order) also F1 per language
    and their macro average ("macro_f1"), so no single language dominates checkpoint choice."""
    def compute(pred):
        logits, labels = pred
        preds = logits.argmax(-1)
        y_true, y_pred = [], []
        for p_row, l_row in zip(preds, labels):
            keep = l_row != -100
            y_true.append([id2label[int(l)] for l in l_row[keep]])
            y_pred.append([id2label[int(p)] for p in p_row[keep]])
        out = {
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred),
        }
        if window_langs is not None:
            per = {}
            for lang in sorted(set(window_langs)):
                idx = [i for i, l in enumerate(window_langs) if l == lang]
                per[lang] = f1_score([y_true[i] for i in idx], [y_pred[i] for i in idx])
                out[f"f1_{lang}"] = per[lang]
            out["macro_f1"] = sum(per.values()) / len(per)
        return out
    return compute


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_train", type=int, help="limit train docs (smoke test)")
    ap.add_argument("--max_eval", type=int, help="limit val docs")
    ap.add_argument("--max_steps", type=int, default=-1)
    ap.add_argument("--output_dir", help="override config output_dir")
    ap.add_argument("--model_name", help="override config model_name, e.g. a DAPT checkpoint dir")
    ap.add_argument("--eval_steps", type=int, help="override config eval/save steps")
    ap.add_argument("--config", default="train.yaml", help="configs/<name>: train.yaml (English DeBERTa) "
                    "or train_ml.yaml (multilingual)")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    entities = load_yaml("label_map.yaml")["entities"]
    labels, label2id, id2label = label_maps(entities)
    set_seed(cfg["seed"])
    out_dir = str(ROOT / (args.output_dir or cfg["output_dir"]))
    if args.model_name:
        local = ROOT / args.model_name
        cfg["model_name"] = str(local) if local.exists() else args.model_name
    print(f"base model: {cfg['model_name']}")

    tok = AutoTokenizer.from_pretrained(cfg["model_name"])
    tcfg = cfg["tokenize"]
    dcfg = cfg["data"]
    ml = dcfg.get("multilingual", False)
    load = (lambda n, lim: load_ml_split(n, lim, dcfg["processed_dir"])) if ml else load_split
    train_ds = tokenize(load(dcfg.get("train_split", "train"), args.max_train), tok, label2id,
                        tcfg["max_length"], tcfg["stride"], tcfg["label_all_tokens"],
                        show_breaks=tcfg["visible_breaks"],
                        script_boundaries=tcfg.get("script_boundaries", False))
    val_ds = tokenize(load(dcfg.get("val_split", "val"), args.max_eval), tok, label2id,
                      tcfg["max_length"], tcfg["stride"], tcfg["label_all_tokens"],
                      show_breaks=tcfg["visible_breaks"],
                      script_boundaries=tcfg.get("script_boundaries", False))
    print(f"train windows={len(train_ds)}  val windows={len(val_ds)}")
    window_langs = val_ds["lang"] if "lang" in val_ds.column_names else None
    train_ds = train_ds.remove_columns([c for c in ["lang"] if c in train_ds.column_names])
    val_ds = val_ds.remove_columns([c for c in ["lang"] if c in val_ds.column_names])

    # transformers 5.x loads in the checkpoint dtype (fp16 for deberta-v3-xsmall); fp16 master
    # weights make AdamW diverge to NaN, so load fp32 and let bf16 autocast do mixed precision.
    model = AutoModelForTokenClassification.from_pretrained(
        cfg["model_name"], num_labels=len(labels), id2label=id2label, label2id=label2id,
        dtype=torch.float32)
    model.config.visible_breaks = tcfg["visible_breaks"]   # read by PIIPredictor
    model.config.script_boundaries = tcfg.get("script_boundaries", False)

    t = cfg["train"]
    targs = TrainingArguments(
        output_dir=out_dir,
        learning_rate=t["learning_rate"],
        num_train_epochs=t["num_train_epochs"],
        max_steps=args.max_steps,
        per_device_train_batch_size=t["per_device_train_batch_size"],
        per_device_eval_batch_size=t["per_device_eval_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        warmup_steps=t["warmup_ratio"],  # transformers 5.x: float < 1 is a ratio
        weight_decay=t["weight_decay"],
        bf16=t["bf16"],
        eval_strategy="steps",
        eval_steps=args.eval_steps or t["eval_steps"],
        save_strategy="steps",
        save_steps=args.eval_steps or t["save_steps"],
        save_total_limit=t["save_total_limit"],
        logging_steps=t["logging_steps"],
        load_best_model_at_end=True,
        metric_for_best_model=t.get("metric_for_best_model", "f1"),
        greater_is_better=True,
        dataloader_num_workers=2,
        train_sampling_strategy="group_by_length",
        report_to="none",
        seed=cfg["seed"],
    )
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorForTokenClassification(tok, pad_to_multiple_of=8),
        processing_class=tok,
        compute_metrics=build_compute_metrics(id2label, window_langs),
    )
    trainer.train()
    print(trainer.evaluate())
    trainer.save_model(out_dir)
    tok.save_pretrained(out_dir)
    print(f"saved -> {out_dir}")


if __name__ == "__main__":
    main()
