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
             show_breaks=True):
    fn = make_tokenize_fn(tokenizer, label2id, max_length, stride, label_all_tokens, show_breaks)
    return ds.map(fn, batched=True, remove_columns=ds.column_names, num_proc=num_proc,
                  desc="tokenize")
