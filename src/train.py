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


def apply_freeze(model, fcfg: dict):
    """v5 grid cells H (mode head_only: only the classifier trains) and P (mode top_k: the classifier
    and the top k encoder layers train). Embeddings and lower layers stay frozen."""
    mode, k = fcfg["mode"], fcfg.get("k", 0)
    n_layers = model.config.num_hidden_layers
    top = {f"encoder.layer.{i}." for i in range(n_layers - k, n_layers)} if mode == "top_k" else set()
    assert mode in ("head_only", "top_k"), mode
    for name, p in model.named_parameters():
        p.requires_grad = name.startswith("classifier.") or any(t in name for t in top)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"freeze: {mode}{f' k={k}' if k else ''}  trainable={n_train / 1e6:.3f}M "
          f"of {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_train", type=int, help="limit train docs (smoke test)")
    ap.add_argument("--max_eval", type=int, help="limit val docs")
    ap.add_argument("--max_steps", type=int, default=-1)
    ap.add_argument("--output_dir", help="override config output_dir")
    ap.add_argument("--model_name", help="override config model_name, e.g. a DAPT checkpoint dir")
    ap.add_argument("--eval_steps", type=int, help="override config eval/save steps")
    ap.add_argument("--resume", nargs="?", const="last", default=None,
                    help="continue an interrupted run: from the newest checkpoint in output_dir, or a given checkpoint dir")
    ap.add_argument("--cpu", action="store_true", help="run on CPU without bf16 (smoke tests while the GPU is busy)")
    ap.add_argument("--config", default="train.yaml", help="configs/<name>: train.yaml (English DeBERTa) "
                    "or train_ml.yaml (multilingual)")
    ap.add_argument("--train_split", help="override data.train_split, e.g. ml_train_v5 (v5 runs, frame pilots)")
    ap.add_argument("--val_split", help="override data.val_split, e.g. ml_val_v5")
    ap.add_argument("--learning_rate", type=float, help="override train.learning_rate (v5 per-method LR sweep)")
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
    if args.train_split:
        dcfg["train_split"] = args.train_split
    if args.val_split:
        dcfg["val_split"] = args.val_split
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
    acfg = cfg.get("adapters")
    if acfg:   # Model B: each window is routed to its language's adapter (gold language in training)
        from adapters import LANG_IDS
        to_ids = lambda b: {"lang_ids": [LANG_IDS[l] for l in b["lang"]]}
        train_ds = train_ds.map(to_ids, batched=True, remove_columns=["lang"])
        val_ds = val_ds.map(to_ids, batched=True, remove_columns=["lang"])
    else:
        train_ds = train_ds.remove_columns([c for c in ["lang"] if c in train_ds.column_names])
        val_ds = val_ds.remove_columns([c for c in ["lang"] if c in val_ds.column_names])

    # transformers 5.x loads in the checkpoint dtype (fp16 for deberta-v3-xsmall); fp16 master
    # weights make AdamW diverge to NaN, so load fp32 and let bf16 autocast do mixed precision.
    base_name = cfg["model_name"]
    if acfg and acfg.get("init_from"):     # B': start from a fine-tuned encoder (Model A)
        local = ROOT / acfg["init_from"]
        base_name = str(local) if local.exists() else acfg["init_from"]
    model = AutoModelForTokenClassification.from_pretrained(
        base_name, num_labels=len(labels), id2label=id2label, label2id=label2id,
        dtype=torch.float32)
    model.config.visible_breaks = tcfg["visible_breaks"]   # read by PIIPredictor
    model.config.script_boundaries = tcfg.get("script_boundaries", False)
    if acfg:
        from adapters import AdapterTokenClassifier
        model = AdapterTokenClassifier(model, acfg["bottleneck"], freeze_base=acfg.get("freeze_base", True),
                                       shared=acfg.get("shared", False))
        if acfg.get("reset_head"):       # B' with a fresh head: only the encoder is inherited
            model.base.classifier.reset_parameters()
            print("classifier head re-initialised (reset_head)")
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        per = model.adapter_params()
        print(f"adapters: bottleneck={acfg['bottleneck']}  shared={acfg.get('shared', False)}  trainable={n_train / 1e6:.2f}M  "
              f"per adapter set={next(iter(per.values())) / 1e6:.3f}M x {len(per)}")
    elif cfg["train"].get("freeze"):
        apply_freeze(model, cfg["train"]["freeze"])

    t = cfg["train"]
    if args.learning_rate:
        t["learning_rate"] = args.learning_rate
        print(f"learning rate override: {args.learning_rate}")
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
        bf16=t["bf16"] and not args.cpu,
        use_cpu=args.cpu,
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
    trainer.train(resume_from_checkpoint=True if args.resume == "last" else args.resume)
    print(trainer.evaluate())
    if acfg:
        # frozen encoder: store only adapters + head and point at the base; else save everything
        frozen = acfg.get("freeze_base", True)
        model.save_pretrained(out_dir, base_dir=base_name if frozen else None)
        router = ROOT / acfg.get("router", "outputs/router") / "router.pkl"
        if router.exists():
            import shutil
            shutil.copy(router, out_dir)
    else:
        trainer.save_model(out_dir)
    tok.save_pretrained(out_dir)
    print(f"saved -> {out_dir}")


if __name__ == "__main__":
    main()
