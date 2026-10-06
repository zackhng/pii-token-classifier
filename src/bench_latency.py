"""Latency of PIIPredictor.predict on one 100,000-character document (LATENCY.md method).

    python bench_latency.py --model_dir ../outputs/xlmr-base-pii-ml-A --doc en|ml [--device cuda|cpu]

Documents are real test documents joined by blank lines and cut to exactly 100,000 characters:
  en  data/processed/test.jsonl (Nemotron-PII, Gretel, AI4Privacy test splits), as in LATENCY.md
  ml  data/processed/ml_real.jsonl, all languages interleaved
Reports median / min / max of warm runs plus tokens, windows and spans found, and a breakdown
(tokenize + window, forward passes, decoding).
"""
import argparse
import json
import random
import statistics
import time

import torch

from common import ROOT
from predict import PIIPredictor
from tokenize_bio import window_encode

N_CHARS = 100_000


def build_doc(kind: str, seed: int = 0) -> str:
    path = ROOT / "data/processed" / ("test.jsonl" if kind == "en" else "ml_real.jsonl")
    rows = [json.loads(l)["text"] for l in open(path, encoding="utf-8")]
    random.Random(seed).shuffle(rows)
    parts, n = [], 0
    for t in rows:
        parts.append(t)
        n += len(t) + 2
        if n >= N_CHARS:
            break
    return "\n\n".join(parts)[:N_CHARS]


def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--doc", choices=["en", "ml"], default="en")
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--threads", type=int, default=6)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    text = build_doc(args.doc)
    t0 = time.perf_counter()
    p = PIIPredictor(args.model_dir, device=args.device)
    load = time.perf_counter() - t0

    t0 = time.perf_counter()
    spans = p.predict(text)          # warm-up (CUDA init, kernel selection)
    sync(args.device)
    first = time.perf_counter() - t0

    times = []
    for _ in range(args.runs):
        t0 = time.perf_counter()
        spans = p.predict(text)
        sync(args.device)
        times.append(time.perf_counter() - t0)

    # breakdown of one run: tokenize + windows / forward / everything else (decode, spans)
    t0 = time.perf_counter()
    wins = window_encode(p.tok, text, p.max_length, p.stride, p.show_breaks, blank_lone_space=p.script_boundaries)
    t_tok = time.perf_counter() - t0
    n_tokens = sum(len(w["input_ids"]) - 2 for w in wins[:1]) + sum(len(w["input_ids"]) - 2 - p.stride for w in wins[1:])
    pad = p.tok.pad_token_id
    with torch.no_grad():
        sync(args.device)
        t0 = time.perf_counter()
        for b in range(0, len(wins), p.batch_windows):
            chunk = wins[b:b + p.batch_windows]
            width = max(len(w["input_ids"]) for w in chunk)
            ids = torch.tensor([w["input_ids"] + [pad] * (width - len(w["input_ids"])) for w in chunk], device=args.device)
            mask = (ids != pad).long()
            extra = {"lang_ids": torch.zeros(len(chunk), dtype=torch.long, device=args.device)} if p.adapter else {}
            p.model(input_ids=ids, attention_mask=mask, **extra)
        sync(args.device)
        t_fwd = time.perf_counter() - t0
    med = statistics.median(times)
    res = {"model": args.model_dir, "doc": args.doc, "device": args.device, "chars": len(text),
           "tokens": n_tokens, "windows": len(wins), "spans": len(spans), "load_s": round(load, 2),
           "first_call_s": round(first, 3), "median_s": round(med, 3), "min_s": round(min(times), 3),
           "max_s": round(max(times), 3), "runs": args.runs,
           "breakdown_s": {"tokenize_windows": round(t_tok, 3), "forward": round(t_fwd, 3),
                           "decode_other": round(max(med - t_tok - t_fwd, 0), 3)},
           "params_M": round(sum(x.numel() for x in p.model.parameters()) / 1e6, 1)}
    print(json.dumps(res))


if __name__ == "__main__":
    main()
