"""Download all three source datasets (all splits) to data/raw/<name>."""
import argparse

from datasets import load_dataset

from common import DATASETS, ROOT, load_yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", choices=list(DATASETS), help="subset of sources")
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


if __name__ == "__main__":
    main()
