"""Frame-share pilot sets (v5, plan 4e): choose the share of v5 frames in ml_train_v5.

Ma et al. (EMNLP 2023) found robustness rises then falls as the share of "context decides"
instances grows (best 10-30% on CoNLL03, 20-40% on ACE05) while i.i.d. F1 slowly drops. We build
equal-size training sets with 0 / 10 / 25 / 40% frame documents from ml_train_v5. The public-source
documents are nested (a higher share uses a prefix of the same shuffled list), so the sets differ
only in how many public documents frames replace. Each is trained with v4.1's recipe and scored
on ml_val_v5 (i.i.d.) and ml_struct (unseen structures).

    python pilot_frames.py --total 50000      ->  data/processed/ml_pilot_f{00,10,25,40}.jsonl
"""
import argparse
import json
import random

from build_dataset import write_jsonl
from common import ROOT

SHARES = [0.0, 0.10, 0.25, 0.40]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--total", type=int, default=50_000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    pub, frm = [], []
    for line in open(ROOT / "data/processed/ml_train_v5.jsonl", encoding="utf-8"):
        r = json.loads(line)
        (frm if r["source"] in ("frames", "frames_wm") else pub).append(r)
    rng = random.Random(a.seed)
    rng.shuffle(pub)
    rng.shuffle(frm)
    for s in SHARES:
        n_f = round(a.total * s)
        rows = pub[:a.total - n_f] + frm[:n_f]
        random.Random(a.seed).shuffle(rows)
        name = f"ml_pilot_f{round(s * 100):02d}"
        write_jsonl(ROOT / "data/processed" / f"{name}.jsonl", rows)
        print(f"{name}: {len(rows)} docs, {n_f} frame docs ({s:.0%})")


if __name__ == "__main__":
    main()
