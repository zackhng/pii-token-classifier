"""Load processed jsonl into HF datasets and tokenize."""
from datasets import Features, Sequence, Value, load_dataset

from common import ROOT, load_yaml
from tokenize_bio import make_tokenize_fn

FEATURES = Features({
    "id": Value("string"),
    "source": Value("string"),
    "text": Value("string"),
    "spans": [{"start": Value("int64"), "end": Value("int64"), "label": Value("string")}],
})


def load_split(name: str, limit: int | None = None):
    cfg = load_yaml("train.yaml")
    path = ROOT / cfg["data"]["processed_dir"] / f"{name}.jsonl"
    ds = load_dataset("json", data_files=str(path), features=FEATURES, split="train")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    return ds


def tokenize(ds, tokenizer, label2id, max_length, stride, label_all_tokens=False, num_proc=4,
             show_breaks=True, script_boundaries=False):
    fn = make_tokenize_fn(tokenizer, label2id, max_length, stride, label_all_tokens, show_breaks,
                          script_boundaries)
    return ds.map(fn, batched=True, remove_columns=ds.column_names, num_proc=num_proc,
                  desc="tokenize")


ML_FIELDS = ["id", "source", "lang", "partial", "text", "spans"]


def load_ml_split(name: str, limit: int | None = None, processed_dir: str = "data/processed"):
    """Multilingual jsonl (build_ml_train.py) -> HF dataset with lang / partial kept for
    tokenization; spans keep only start / end / label (label may be "O" or IGNORE)."""
    import json
    from datasets import Dataset
    rows = []
    with open(ROOT / processed_dir / f"{name}.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            rows.append({"id": r["id"], "source": r["source"], "lang": r["lang"], "partial": r.get("partial", ""),
                         "text": r["text"],
                         "spans": [{"start": s["start"], "end": s["end"], "label": s["label"]} for s in r["spans"]]})
            if limit and len(rows) >= limit:
                break
    return Dataset.from_list(rows)
