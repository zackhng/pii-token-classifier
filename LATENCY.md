# Inference Latency

Latency of `PIIPredictor.predict` (in `src/predict.py`) on a **100,000-character document**.

## Setup

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 5060 Ti 16 GB (driver 591.86) |
| CPU | AMD Ryzen 5 8400F, 6 cores / 12 threads (PyTorch uses 6) |
| RAM | 32 GB |
| Software | Windows 11, Python 3.10, torch 2.8.0+cu128, transformers 5.18.0 |
| Model | `deberta-v3-xsmall-pii` v1.0, fp32 weights, no quantization |
| Inference | windows of 384 tokens with a 96-token overlap, batch of 16 windows |

**Test document.** It is built from real test-set documents (Nemotron-PII, Gretel and AI4Privacy) joined together and cut to exactly 100,000 characters. It has 23,073 tokens, which become 81 windows. The model finds 322 PII spans in it.

The code was benchmarked as a fresh clone of this repo with the release weights. Warm numbers are medians over 20 runs on GPU and 3 runs on CPU, taken after the model is loaded and has run once.

## Results: 100k-character document

| Device | Median | p90 | Min | Max |
|---|---|---|---|---|
| **GPU** (RTX 5060 Ti) | **494 ms** | 503 ms | 486 ms | 582 ms |
| **CPU** (Ryzen 5 8400F) | **8.05 s** | – | 8.00 s | 8.12 s |

One-time costs: loading the model takes ~1.2 s, and the first `predict` call on GPU takes ~0.8 s for CUDA warm-up.

### Where the GPU time goes

| Stage | Time |
|---|---|
| Tokenize + split into windows | 29 ms |
| Model forward pass (81 windows) | 453 ms |
| Post-processing (merge windows, decode spans) | ~12 ms |

The forward pass takes ~92% of the time. Batch size barely matters: 4, 16 and 64 windows per batch gave 496, 501 and 491 ms.

### Latency vs. document length

| Characters | GPU | CPU |
|---|---|---|
| 1,000 | 16 ms | 62 ms |
| 10,000 | 41 ms | 711 ms |
| 50,000 | 232 ms | – |
| 100,000 | 508 ms | 8.05 s |

Above a few thousand characters, latency grows roughly linearly with length: about **5 ms per 1k characters on GPU** and **80 ms per 1k characters on CPU**.

## Change in v1.0 → `c1ef3c7`

The first version of `predict.py` took **~1.10 s** on GPU for the same document. More than half of that was Python post-processing. It pulled each token's prediction off the GPU one at a time, and it recomputed each span's confidence by scanning every token. Doing the argmax in one tensor operation and computing span scores in a single pass cut it to **~0.50 s**. The output is identical to v1.0 on 301 documents.

## Why CPU is ~16× slower

- **Most parameters cost nothing to compute.** The "xsmall" model has 70M parameters, but most are the 128k-word vocabulary table, which is just a lookup. The actual compute is a 12-layer, 384-wide encoder.
- **DeBERTa attention is heavier than BERT's.** Its disentangled attention adds content-to-position and position-to-content terms, so each token costs roughly 2–3× more than in BERT.
- **The windows overlap.** With a 96-token overlap, 23k tokens become ~31k tokens actually processed.
- **The CPU path is unoptimized:** plain fp32 PyTorch, with no quantization, bf16 or ONNX Runtime.

Options that may speed up CPU inference have not been benchmarked yet:
- bf16 autocast (the Zen 4 CPU supports native bf16).
- int8 dynamic quantization.
- ONNX Runtime.
- A smaller window overlap.
