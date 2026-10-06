"""Download all source datasets (all splits) to data/raw/<name>.

`ai4privacy500k` (Hindi for the multilingual zero-shot test, build_ml_eval.py) is a single
validation file: data/raw/ai4privacy500k/validation.jsonl.
"""
import argparse
import shutil

from datasets import load_dataset
from huggingface_hub import hf_hub_download

from common import DATASETS, ROOT, load_yaml

# evaluation-only files: name -> (hub id, file in the repo)
EVAL_FILES = {"ai4privacy500k": ("ai4privacy/open-pii-masking-500k-ai4privacy", "data/validation/test.jsonl")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", choices=list(DATASETS) + list(EVAL_FILES), help="subset of sources")
    args = ap.parse_args()

    raw_dir = ROOT / load_yaml("train.yaml")["data"]["raw_dir"]
    for name, hub_id in DATASETS.items():
        if args.only and name not in args.only:
            continue
        out = raw_dir / name
        if out.exists():
            print(f"[skip] {name} already at {out}")
            continue
        print(f"[download] {hub_id}")
        ds = load_dataset(hub_id)
        print(ds)
        ds.save_to_disk(str(out))
    for name, (hub_id, path) in EVAL_FILES.items():
        if args.only and name not in args.only:
            continue
        out = raw_dir / name / "validation.jsonl"
        if out.exists():
            print(f"[skip] {name} already at {out}")
            continue
        print(f"[download] {hub_id}/{path}")
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(hf_hub_download(hub_id, path, repo_type="dataset"), out)


if __name__ == "__main__":
    main()
