# PII Token Classifier

BIO token-classification models for detecting personally identifiable information in wealth-management
text, following the [OpenMed](https://huggingface.co/OpenMed) PII recipe (Hugging Face Trainer, max
length 384, first-sub-token labelling, seqeval F1):
- **v5.x / v4.x, multilingual:** 12 languages, [XLM-R base](https://huggingface.co/FacebookAI/xlm-roberta-base)
  (v5.0–v5.2, v4.1, v4.2) or [Multilingual-MiniLM](https://huggingface.co/microsoft/Multilingual-MiniLM-L12-H384) (v4.3).
  English, Simplified / Traditional Chinese, Japanese, Korean, Hindi, Arabic, Thai, Vietnamese, Malay,
  Indonesian, Tagalog.
- **v1–v3, English:** [`microsoft/deberta-v3-xsmall`](https://huggingface.co/microsoft/deberta-v3-xsmall).

**Which model** (multilingual; held-out test mean strict F1 / coverage over `ml_struct_test`, `ml_wm_test`, `ml_wm_known`):
- **v5.1** (XLM-R base + span head): the most accurate base-size model, 0.954 / 0.976; better boundaries on long
  entities (addresses, company names) than v5.0.
- **v5.2** (XLM-R base + span head with boundary smoothing): 0.951 / 0.984, level with v5.1 on accuracy with more coverage.
- **v5.0** (XLM-R base, per-token head): 0.947 / **0.992**; the most coverage of the base-size models, for mask-everything use.
- **v4.3** for multilingual text when speed matters: 2.3× faster on GPU and 3× on CPU than v4.1 / v5.0
  (faster than v3 too), but trained on v4's data.
- **v4.1** if you need v4's fragment-style ADDRESS spans (street, number and postcode separately).
- **v3.0** for English-only text: more accurate on English.
- **v4.2** is the language-adapter research variant; see [`model_card/v4.md`](model_card/v4.md).

## Releases

| Version | Weights | Highlights | Model card |
|---|---|---|---|
| **v5.2** (current, multilingual) | [`xlmr-base-pii-ml-v5.2.zip`](../../releases/tag/v5.2) | v5.0 with one change: a **span head with boundary smoothing** (scores whole spans; near-miss boundaries get partial credit in training). Dev sets, strict F1: unseen structures 0.921 → **0.945**, wm grid 0.934 → 0.948, ADDRESS 0.888 → **0.918**, BUSINESS 0.908 → 0.927; coverage 0.996 → 0.993. 2–6% slower. Held-out tests 0.948 / 0.956 / 0.951 (mean 0.951), coverage 0.984 | [`model_card/v5.2.md`](model_card/v5.2.md) |
| v5.1 (multilingual) | [`xlmr-base-pii-ml-v5.1.zip`](../../releases/tag/v5.1) | v5.0 with one change: a **span head** (GlobalPointer). Dev sets, strict F1: unseen structures 0.921 → 0.932, wm grid 0.934 → **0.951**, public data 0.975, English stress 0.976; coverage 0.996 → 0.990. Held-out tests 0.945 / 0.960 / 0.957 (mean **0.954**, best base-size model), coverage 0.976 | [`model_card/v5.1.md`](model_card/v5.1.md) |
| **v5.0** (multilingual) | [`xlmr-base-pii-ml-v5.0.zip`](../../releases/tag/v5.0) | v4.1's model and recipe on new training data built for **unseen sentence structures**: hand-written and generated wealth-management frames in 12 languages and English + X code-switching, real bank / custodian / broker account formats, look-alike contrasts, whole-address labels. Held-out unseen-structure tests **0.557 → 0.948** and **0.725 → 0.949** (strict F1); accounts no longer read as tax IDs (account recall 0.54 → 0.97, incl. IBAN / IBKR / Fidelity / Pershing formats); look-alikes tagged as PII 24.8% → 0.8%; public data and English test level or better on the same gold; punctuation trim on by default | [`model_card/v5.md`](model_card/v5.md) |
| v4.1 (multilingual) | [`xlmr-base-pii-ml-v4.1.zip`](../../releases/tag/v4.1) | **12 languages**, XLM-R base fully fine-tuned on 337k docs from 14 public sources (+ generated data only where none exists). Multilingual real-data F1 0.675 → **0.942**, wealth-management docs 0.523 → **0.873** (v3 with the new CJK / Thai word boundaries); English stress 0.936 → 0.955, English test 0.925 → 0.910 | [`model_card/v4.md`](model_card/v4.md) |
| v4.3 (fast, multilingual) | [`minilm-pii-ml-v4.3.zip`](../../releases/tag/v4.3) | v4.1's recipe on Multilingual-MiniLM-L12-H384 (21.5M transformer parameters, 117.5M total). Multilingual real-data F1 0.939, wealth-management docs 0.860, English test 0.903. 100k-character document in **0.38 s on GPU / 6.7 s on CPU** (v4.1 0.87 s / 19.8 s, v3 0.60 s / 9.3 s) | [`model_card/v4.md`](model_card/v4.md) |
| v4.2 (adapters) | [`xlmr-base-pii-adapters-v4.2.zip`](../../releases/tag/v4.2) | Frozen XLM-R + one 1.2M-parameter adapter per language + language router (13.3M trained parameters). Within 0.4–1.2 F1 of v4.1 on public / mixed-language / Korean financial data, but 0.690 on unseen financial formats and 0.821 on English test | [`model_card/v4.md`](model_card/v4.md) |
| **v3.0** (English) | [`deberta-v3-xsmall-pii-v3.zip`](../../releases/tag/v3.0) | v2.1's fine-tuning on top of **domain-adaptive pretraining** (178M tokens of public financial / regulatory / RM-communication text): stress F1 0.921 → 0.936, ADDRESS 0.792 → 0.848, multi-line addresses exact 59% → 70% | [`model_card/v3.md`](model_card/v3.md) |
| v2.1 | [`deberta-v3-xsmall-pii-v2.1.zip`](../../releases/tag/v2.1) | ADDRESS: model sees line breaks / tabs (multi-line addresses), 20,848 real public addresses from SG, IN, UAE, UK, HK, ASEAN & Asia in letters / KYC forms / statements / signatures, bare place names no longer ADDRESS | [`model_card/v2.1.md`](model_card/v2.1.md) |
| v2.0 | [`deberta-v3-xsmall-pii-v2.zip`](../../releases/tag/v2.0) | Rejects look-alikes: DOB vs ordinary dates, account numbers vs amounts / codes; label-map fixes; DOB format augmentation; form-style records | [`model_card/v2.md`](model_card/v2.md) |
| v1.0 | [`deberta-v3-xsmall-pii.zip`](../../releases/tag/v1.0) | First release | [`model_card/v1.md`](model_card/v1.md) |

<!-- v5-suggestions:start -->
## Which v5 model to use (suggestions)

Built from the experiment log below by `src/recommend.py` (re-run by `experiments_table.py`), so it changes as queued runs finish. **Accuracy** = mean strict F1 over the six multilingual sets (struct, wm, synth, real, mixed, Kiii); **coverage** = mean share of gold PII spans overlapped by any prediction. Latency: one document, ml_real text, CPU = 6 threads fp32, GPU = RTX 5060 Ti, median of warm runs. Single seed: gaps under 0.005 are within noise, so near-ties are listed. Dev sets only; nothing here has been checked on the held-out test sets.

**Model names**

| Name | What it is |
|---|---|
| v5.0 (A-v5data) | XLM-R base, fully fine-tuned, softmax BIO head (one label per token), trained on ml_train_v5. The released v5 model and the parent of every other row. |
| H1 | encoder frozen, only the classifier head trains |
| P1 | only the top 4 encoder layers + head train (partial unfreeze) |
| S1 | encoder frozen + one shared bottleneck adapter for all languages |
| B1 / B′ | encoder frozen + one adapter per language, routed by the window's main language (B1 on pretrained XLM-R, B′ on v5.0's encoder). Dropped: switching adapters adds latency on long mixed-language documents |
| AL2 / AL4 / AL8 | "Additional Layers": v5.0 with 2 / 4 / 8 extra transformer layers between the encoder and the classifier |
| CRF | v5.0 with a linear-chain CRF on the classifier: decodes the best label sequence per window instead of each token alone, so B/I tags stay consistent |
| SPAN | v5.0 with a span head (GlobalPointer) instead of per-token labels: scores every (start word, end word, type) candidate up to 160 tokens long and keeps the best non-overlapping spans. Better boundaries on long spans (ADDRESS, BUSINESS) |
| SPAN-CE | SPAN with a different loss: each candidate span is a softmax over {not PII + 8 types} (the span-classification loss of Zhu & Li, ACL 2022) |
| SPAN-BS | SPAN-CE + boundary smoothing (Zhu & Li 2022): 10% of each gold span's target is spread over spans whose start or end is one word off, so near-miss boundaries are not punished as hard |
| mmBERT-small / -base | encoder swapped to mmBERT (ModernBERT architecture, 1,800+ languages); 140M / 307M parameters |
| MiniLM-L12 / -L6 | encoder swapped to Multilingual-MiniLMv2, 12 or 6 layers of width 384, distilled from XLM-R large (same tokenizer as XLM-R); about 117M / 107M parameters, mostly the embedding table, so far fewer compute per token |
| … + span / + span-BS | that encoder with the SPAN or SPAN-BS head instead of the softmax head |
| InfoXLM-base | XLM-R base further pretrained with cross-lingual contrastive objectives (Microsoft) |
| XLM-V-base | XLM-R base architecture with a 1M-token vocabulary (fewer, better tokens for many languages) |
| mDeBERTa-v3-base | multilingual DeBERTa v3 (disentangled attention, ELECTRA-style pretraining) |
| XLM-R large | 24-layer, 560M-parameter XLM-R |
| merge central / avg, overlap 64 | inference-only changes: how predictions from overlapping 384-token windows are combined, and a 64-token instead of 96-token window overlap |

| Use | Model | Accuracy | Coverage | CPU 1k chars | CPU 100k chars | GPU 100k chars | Why |
|---|---|---|---|---|---|---|---|
| Most accurate | **XLM-R large + span-BS** | 0.950 | 0.995 | 0.59 s | 86.69 s | 3.32 s | highest mean strict F1 |
| Most cost-efficient (CPU) | **XLM-V-base + span-BS** | 0.940 | 0.993 | 0.15 s | 22.56 s | 1.00 s | fastest model within 0.010 strict F1 of the most accurate; 1.1× faster than v5.0 on CPU |
| Fastest acceptable | **MiniLM-L6 + span** | 0.928 | 0.985 | 33 ms | 4.80 s | 0.38 s | fastest model within 0.010 strict F1 of v5.0 |
| Widest coverage (masking) | **AL8** | 0.939 | 0.996 | – | – | – | highest mean coverage; within noise: AL4, v5.0 (A-v5data), B′, 14 more |
| Reference | **v5.0 (A-v5data)** | 0.935 | 0.996 | 0.16 s | 24.38 s | 1.07 s | v5.0 as released |

### Accuracy vs CPU latency (models that nothing beats on both)

| Model | Accuracy | Coverage | CPU 100k chars | Speed vs v5.0 |
|---|---|---|---|---|
| MiniLM-L6 | 0.912 | 0.992 | 4.25 s | 5.7× |
| MiniLM-L6 + span | 0.928 | 0.985 | 4.80 s | 5.1× |
| MiniLM-L12 + span | 0.936 | 0.990 | 8.79 s | 2.8× |
| mmBERT-small + span | 0.937 | 0.987 | 20.29 s | 1.2× |
| XLM-V-base + span-BS | 0.940 | 0.993 | 22.56 s | 1.1× |
| SPAN | 0.943 | 0.990 | 24.81 s | 1.0× |
| SPAN-BS | 0.943 | 0.993 | 24.98 s | 1.0× |
| XLM-R large + span-BS | 0.950 | 0.995 | 86.69 s | 0.3× |

### Best model per entity (six sets pooled)

Best strict F1, best coverage, and the best strict F1 among CPU-fast models (≤ 10 s per 100k characters: MiniLM-L12, MiniLM-L6, MiniLM-L12 + span, MiniLM-L6 + span, MiniLM-L12 + span-BS, MiniLM-L6 + span-BS). Last columns: strict F1 / coverage of the overall picks for comparison.

| Entity | n | Best strict F1 | Best coverage | Best CPU-fast strict F1 | XLM-R large + span-BS | XLM-V-base + span-BS |
|---|---|---|---|---|---|---|
| PERSON | 31474 | XLM-R large + span-BS 0.940 (ties: SPAN, XLM-V-base + span-BS, InfoXLM-base + span-BS …) | mmBERT-base 0.999 (ties: CRF, v5.0 (A-v5data), B′ …) | MiniLM-L12 + span 0.932 (ties: MiniLM-L12 + span-BS, MiniLM-L6 + span, MiniLM-L6 + span-BS) | 0.940 / 0.998 | 0.938 / 0.997 |
| ADDRESS | 11214 | XLM-R large + span-BS 0.944 | mmBERT-base 0.997 (ties: B′, v5.0 (A-v5data), AL4 …) | MiniLM-L12 + span 0.898 (ties: MiniLM-L12 + span-BS) | 0.944 / 0.992 | 0.901 / 0.986 |
| ACCOUNT | 16669 | XLM-R large + span-BS 0.946 (ties: mmBERT-small + span, InfoXLM-base + span-BS, SPAN …) | v5.0 (A-v5data) 0.998 (ties: B′, CRF, P1 …) | MiniLM-L12 + span 0.937 (ties: MiniLM-L12 + span-BS) | 0.946 / 0.998 | 0.936 / 0.998 |
| BUSINESS | 6685 | XLM-R large + span-BS 0.954 | AL8 0.997 (ties: MiniLM-L6, S1, AL2 …) | MiniLM-L12 + span-BS 0.915 (ties: MiniLM-L12 + span, MiniLM-L6 + span-BS, MiniLM-L6 + span) | 0.954 / 0.995 | 0.940 / 0.988 |
| DOB | 6788 | mDeBERTa-v3-base + span-BS 0.934 (ties: AL4, v5.0 (A-v5data), AL8 …) | MiniLM-L12 0.994 (ties: AL4, mmBERT-base, P1 …) | MiniLM-L12 0.923 (ties: MiniLM-L12 + span, MiniLM-L12 + span-BS) | 0.932 / 0.984 | 0.928 / 0.985 |
| PHONE | 11922 | XLM-R large + span-BS 0.976 (ties: mmBERT-base, SPAN, mDeBERTa-v3-base + span-BS …) | mmBERT-base 1.000 (ties: mDeBERTa-v3-base + span-BS, XLM-R large + span-BS, B′ …) | MiniLM-L12 + span-BS 0.972 (ties: MiniLM-L12 + span) | 0.976 / 1.000 | 0.975 / 0.999 |
| EMAIL | 12290 | SPAN 0.994 (ties: SPAN-CE, XLM-V-base + span-BS, MiniLM-L12 + span …) | v5.0 (A-v5data) 1.000 (ties: H1, P1, S1 …) | MiniLM-L12 + span 0.993 (ties: MiniLM-L12 + span-BS, MiniLM-L6 + span, MiniLM-L6 + span-BS …) | 0.993 / 0.997 | 0.993 / 0.997 |
| TIN | 8321 | XLM-R large + span-BS 0.970 (ties: mDeBERTa-v3-base + span-BS) | P1 0.993 (ties: AL4, MiniLM-L6, SPAN-CE …) | MiniLM-L12 + span-BS 0.956 (ties: MiniLM-L12 + span) | 0.970 / 0.983 | 0.960 / 0.982 |

### Best model per language (six sets pooled)

Best strict F1, best coverage, and the best strict F1 among CPU-fast models (≤ 10 s per 100k characters: MiniLM-L12, MiniLM-L6, MiniLM-L12 + span, MiniLM-L6 + span, MiniLM-L12 + span-BS, MiniLM-L6 + span-BS). Last columns: strict F1 / coverage of the overall picks for comparison.

| Language | n | Best strict F1 | Best coverage | Best CPU-fast strict F1 | XLM-R large + span-BS | XLM-V-base + span-BS |
|---|---|---|---|---|---|---|
| en | 7088 | CRF 0.984 (ties: SPAN, SPAN-CE, mmBERT-small + span …) | v5.0 (A-v5data) 0.996 (ties: CRF, B′, XLM-V-base + span-BS …) | MiniLM-L12 + span 0.975 (ties: MiniLM-L12 + span-BS, MiniLM-L6 + span, MiniLM-L6 + span-BS) | 0.981 / 0.994 | 0.977 / 0.995 |
| zh-Hans | 7382 | XLM-R large + span-BS 0.988 (ties: mDeBERTa-v3-base + span-BS, InfoXLM-base + span-BS) | AL4 0.999 (ties: AL2, AL8, mmBERT-small …) | MiniLM-L12 + span-BS 0.978 (ties: MiniLM-L12 + span) | 0.988 / 0.996 | 0.979 / 0.996 |
| zh-Hant | 7332 | XLM-R large + span-BS 0.986 (ties: SPAN-BS, SPAN-CE, mDeBERTa-v3-base + span-BS …) | AL8 0.994 (ties: AL4, AL2, S1 …) | MiniLM-L12 + span 0.982 (ties: MiniLM-L12 + span-BS) | 0.986 / 0.988 | 0.979 / 0.990 |
| ja | 7460 | XLM-R large + span-BS 0.973 | AL4 0.999 (ties: mmBERT-base, B′, P1 …) | MiniLM-L12 + span-BS 0.930 (ties: MiniLM-L12 + span) | 0.973 / 0.995 | 0.938 / 0.991 |
| ko | 23782 | XLM-R large + span-BS 0.860 | v5.0 (A-v5data) 0.998 (ties: B′, AL4, AL2 …) | MiniLM-L12 + span 0.849 (ties: MiniLM-L12 + span-BS) | 0.860 / 0.996 | 0.855 / 0.994 |
| hi | 5666 | InfoXLM-base + span-BS 0.983 (ties: XLM-R large + span-BS, SPAN, SPAN-CE …) | AL2 0.996 (ties: AL4, P1, S1 …) | MiniLM-L12 + span 0.976 (ties: MiniLM-L12 + span-BS) | 0.983 / 0.991 | 0.975 / 0.986 |
| ar | 4158 | XLM-R large + span-BS 0.972 | P1 0.999 (ties: MiniLM-L12, mmBERT-base, MiniLM-L6 …) | MiniLM-L12 + span-BS 0.954 (ties: MiniLM-L12 + span) | 0.972 / 0.986 | 0.961 / 0.981 |
| th | 4134 | XLM-R large + span-BS 0.987 (ties: SPAN, mmBERT-base, AL4 …) | S1 1.000 (ties: AL4, SPAN-CE, SPAN-BS …) | MiniLM-L12 + span-BS 0.979 (ties: MiniLM-L12 + span, MiniLM-L6 + span, MiniLM-L6 + span-BS) | 0.987 / 0.998 | 0.981 / 0.996 |
| vi | 7391 | XLM-R large + span-BS 0.984 | S1 0.996 (ties: AL8, AL4, AL2 …) | MiniLM-L12 + span 0.964 (ties: MiniLM-L12 + span-BS) | 0.984 / 0.995 | 0.972 / 0.992 |
| ms | 7370 | XLM-R large + span-BS 0.974 (ties: mmBERT-small + span) | mmBERT-base 0.999 (ties: SPAN-CE, S1, XLM-R large + span-BS …) | MiniLM-L12 + span-BS 0.952 | 0.974 / 0.998 | 0.961 / 0.995 |
| id | 7367 | XLM-R large + span-BS 0.990 (ties: mDeBERTa-v3-base + span-BS) | AL2 0.999 (ties: AL8, XLM-R large + span-BS, mmBERT-base …) | MiniLM-L12 + span 0.979 (ties: MiniLM-L6 + span, MiniLM-L6 + span-BS, MiniLM-L12 + span-BS) | 0.990 / 0.998 | 0.984 / 0.997 |
| tl | 7554 | XLM-V-base + span-BS 0.988 (ties: XLM-R large + span-BS, mDeBERTa-v3-base + span-BS) | XLM-R large + span-BS 0.999 (ties: mmBERT-base, AL2, AL4 …) | MiniLM-L12 + span 0.978 (ties: MiniLM-L12 + span-BS, MiniLM-L6 + span-BS, MiniLM-L6 + span) | 0.987 / 0.999 | 0.988 / 0.998 |

<!-- v5-suggestions:end -->

<!-- v5-experiments:start -->
## v5 experiment log

Every v5 experiment that has run, is running or is queued; ideas that were dropped, cancelled or not yet queued are in a separate table at the end. Each row changes **one thing** relative to its parent. Choices were made on the dev sets only; the held-out test sets (ml_struct_test, ml_wm_test) were used once, for v5.0. Single seed (42) throughout. Regenerate with `cd src && python experiments_table.py`.

- **struct** = ml_struct_dev (unseen sentence structures), **wm** = ml_wm_dev (wealth-management grid, 12 languages + English–X), **synth** = ml_synth, **real** = ml_real (public data), **mixed** = ml_mixed (code-switched public data), **Kiii** = Korean financial documents.
- Multilingual sets: punctuation trim on, merged-address gold for real / mixed, scored from saved predictions. English test / stress: evaluate.py, **untrimmed** (trim adds about +0.015 on English test for XLM-R-tokenizer models, e.g. v5.0 .894 → .911, SPAN .907 → .923). val macro = best ml_val_v5 macro-F1 during training (pilots / v4 rows: their own validation set).
- Pilot rows train on a 50k-document pilot set; LR-sweep pilots are not listed. Older runs scored on `ml_struct` (the original name of ml_struct_dev, same 3,084 documents) are re-scored against today's gold.
- Threshold rows: the per-label thresholds tuned on wm (results/v5/summary/v5_postproc.json) applied to the saved predictions; their wm column is the tuning set, so not a fair test. English needs a model run, so –.
- Model names (AL, SPAN, SPAN-BS, …): see **Model names** in the suggestions section above.
- **–** = not run, not scored on that set, or not recoverable (predictions not saved).

### Strict F1

| Group | Experiment | One change vs parent | Parent | Status | struct | wm | synth | real | mixed | Kiii | EN test | EN stress | val macro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Baselines | v4.1 | v4 release (Model A): XLM-R base, v4 data | – | done | 0.541 | 0.739 | 0.896 | 0.905 | 0.912 | 0.852 | 0.910 | 0.955 | 0.946 |
| Baselines | v4.2 | v4 release (Model B): per-language adapters | – | done | 0.474 | 0.606 | 0.705 | 0.894 | 0.901 | 0.848 | 0.821 | 0.907 | 0.937 |
| Baselines | v4 MiniLM | v4 data, encoder Multilingual-MiniLM-L12 | v4.1 | done | 0.534 | 0.720 | 0.883 | 0.902 | 0.909 | 0.850 | 0.903 | 0.932 | 0.939 |
| Stage 1: data | pilot r1 0% frames | 50k pilot, old frames, share 0% | – | done | 0.697 | – | – | 0.962 | – | – | – | – | 0.917 |
| Stage 1: data | pilot r1 10% | frame share 10% | pilot r1 0% | done | 0.919 | – | – | 0.959 | – | – | – | – | 0.912 |
| Stage 1: data | pilot r1 25% | frame share 25% | pilot r1 0% | done | 0.916 | – | – | 0.956 | – | – | – | – | 0.911 |
| Stage 1: data | pilot r1 40% | frame share 40% | pilot r1 0% | done | 0.916 | – | – | 0.956 | – | – | – | – | 0.907 |
| Stage 1: data | pilot r2 0% frames | 50k pilot, frames + wm grid, share 0% | – | done | 0.614 | 0.788 | – | 0.961 | – | – | – | – | 0.916 |
| Stage 1: data | pilot r2 10% | frame share 10% | pilot r2 0% | done | 0.913 | 0.917 | – | 0.960 | – | – | – | – | 0.916 |
| Stage 1: data | pilot r2 25% (chosen) | frame share 25% | pilot r2 0% | done | 0.930 | 0.935 | – | 0.959 | – | – | – | – | 0.910 |
| Stage 1: data | pilot r2 40% | frame share 40% | pilot r2 0% | done | 0.945 | 0.937 | – | 0.953 | – | – | – | – | 0.908 |
| Stage 1: data | **v5.0** (A-v5data) | data: ml_train_v5 (v4.1 recipe) | v4.1 | done | 0.921 | 0.934 | 0.960 | 0.973 | 0.974 | 0.851 | 0.894 | 0.963 | 0.945 |
| Stage 2: what trains | H1 | encoder frozen, head only (LR 3e-3) | v5.0 | done | 0.637 | 0.618 | 0.659 | 0.729 | 0.723 | 0.486 | 0.613 | 0.563 | 0.642 |
| Stage 2: what trains | P1 | top 4 layers + head (LR 1e-4) | v5.0 | done | 0.884 | 0.926 | 0.950 | 0.968 | 0.967 | 0.848 | 0.884 | 0.941 | 0.933 |
| Stage 2: what trains | S1 | frozen encoder + one shared adapter (LR 3e-4) | v5.0 | done | 0.911 | 0.932 | 0.962 | 0.966 | 0.966 | 0.844 | 0.844 | 0.935 | 0.926 |
| Stage 2: what trains | B1 | frozen encoder + per-language adapters (LR 1e-4) | v5.0 | done | 0.865 | 0.889 | 0.904 | 0.960 | 0.956 | 0.848 | 0.818 | 0.926 | 0.934 |
| Stage 2: what trains | B1 lr 3e-4 | LR 1e-4 → 3e-4 | B1 | done | 0.865 | 0.885 | 0.900 | 0.962 | 0.961 | 0.850 | 0.817 | 0.936 | 0.935 |
| Stage 2: what trains | B′ | per-language adapters on v5.0's encoder | B1 | done | 0.923 | 0.934 | 0.957 | 0.972 | 0.974 | 0.851 | 0.870 | 0.965 | 0.946 |
| Deep head | AL2 | +2 transformer layers before the classifier (head LR 1e-4) | v5.0 | done | 0.923 | 0.935 | 0.966 | 0.973 | 0.974 | 0.851 | 0.895 | 0.963 | 0.945 |
| Deep head | AL4 | +4 layers | v5.0 | done | 0.917 | 0.939 | 0.958 | 0.973 | 0.975 | 0.852 | 0.896 | 0.964 | 0.944 |
| Deep head | AL8 | +8 layers | v5.0 | done | 0.930 | 0.940 | 0.966 | 0.973 | 0.975 | 0.853 | 0.897 | 0.974 | 0.945 |
| Heads | CRF | CRF on the classifier (transitions LR 5e-3) | v5.0 | done | 0.933 | 0.942 | 0.964 | 0.974 | 0.975 | 0.848 | 0.902 | 0.970 | 0.949 |
| Heads | **SPAN** | GlobalPointer span head (head LR 1e-4) | v5.0 | done | 0.932 | 0.951 | 0.970 | 0.975 | 0.976 | 0.854 | 0.907 | 0.976 | 0.949 |
| Heads | SPAN-CE | span loss → per-span softmax CE | SPAN | done | 0.935 | 0.947 | 0.964 | 0.974 | 0.976 | 0.853 | 0.906 | 0.969 | 0.951 |
| Heads | SPAN-BS | boundary smoothing ε 0.1, D 1 | SPAN-CE | done | 0.945 | 0.948 | 0.967 | 0.974 | 0.976 | 0.850 | 0.906 | 0.968 | 0.949 |
| Encoders (softmax head) | mmBERT-small | encoder → mmBERT-small | v5.0 | done | 0.889 | 0.926 | 0.955 | 0.973 | 0.977 | 0.853 | 0.905 | 0.961 | 0.939 |
| Encoders (softmax head) | mmBERT-base | encoder → mmBERT-base | v5.0 | done | 0.888 | 0.934 | 0.953 | 0.975 | 0.979 | 0.854 | 0.911 | 0.972 | 0.945 |
| Encoders (softmax head) | MiniLM-L12 | encoder → mMiniLMv2-L12-H384 | v5.0 | done | 0.881 | 0.926 | 0.946 | 0.964 | 0.963 | 0.840 | 0.876 | 0.919 | 0.929 |
| Encoders (softmax head) | MiniLM-L6 | encoder → mMiniLMv2-L6-H384 | v5.0 | done | 0.879 | 0.911 | 0.938 | 0.957 | 0.953 | 0.834 | 0.863 | 0.894 | 0.914 |
| Encoders + span | mmBERT-small + span | span head | mmBERT-small | done | 0.914 | 0.938 | 0.961 | 0.976 | 0.979 | 0.855 | 0.921 | 0.966 | 0.947 |
| Encoders + span | MiniLM-L12 + span | span head | MiniLM-L12 | done | 0.919 | 0.940 | 0.963 | 0.970 | 0.971 | 0.851 | 0.901 | 0.945 | 0.941 |
| Encoders + span | MiniLM-L6 + span | span head | MiniLM-L6 | done | 0.919 | 0.926 | 0.952 | 0.964 | 0.963 | 0.846 | 0.890 | 0.916 | 0.929 |
| Encoders + boundary-smoothed span | MiniLM-L12 + span-BS | head → SPAN-BS head (CE loss + boundary smoothing) | MiniLM-L12 + span | done | 0.921 | 0.940 | 0.961 | 0.969 | 0.968 | 0.849 | 0.900 | 0.945 | 0.941 |
| Encoders + boundary-smoothed span | MiniLM-L6 + span-BS | head → SPAN-BS head (CE loss + boundary smoothing) | MiniLM-L6 + span | done | 0.917 | 0.927 | 0.953 | 0.964 | 0.963 | 0.844 | 0.891 | 0.920 | 0.929 |
| Encoders + boundary-smoothed span | mDeBERTa-v3-base + span-BS | encoder → mDeBERTa-v3-base | SPAN-BS | done | 0.924 | 0.946 | 0.959 | 0.973 | 0.975 | 0.851 | 0.884 | 0.981 | 0.953 |
| Encoders + boundary-smoothed span | InfoXLM-base + span-BS | encoder → InfoXLM-base | SPAN-BS | done | 0.928 | 0.944 | 0.966 | 0.974 | 0.975 | 0.850 | 0.905 | 0.968 | 0.949 |
| Encoders + boundary-smoothed span | XLM-V-base + span-BS | encoder → XLM-V-base (resumed from step 32,000 after a crash) | SPAN-BS | done | 0.927 | 0.950 | 0.972 | 0.971 | 0.971 | 0.852 | 0.872 | 0.955 | 0.952 |
| Encoders + boundary-smoothed span | XLM-R large + span-BS | encoder → XLM-R large | SPAN-BS | done | 0.948 | 0.963 | 0.985 | 0.976 | 0.977 | 0.853 | 0.911 | 0.983 | 0.958 |
| Inference only | v5.0 merge central | overlapping windows: most central window | v5.0 | done | 0.921 | 0.934 | 0.960 | 0.973 | 0.974 | 0.852 | 0.894 | 0.963 | – |
| Inference only | v5.0 merge avg | overlapping windows: mean probabilities | v5.0 | done | 0.921 | 0.934 | 0.960 | 0.973 | 0.974 | 0.851 | 0.894 | 0.963 | – |
| Inference only | v5.0 overlap 64 | window overlap 96 → 64 | v5.0 | done | 0.921 | 0.934 | 0.960 | 0.973 | 0.974 | 0.851 | 0.895 | 0.963 | – |
| Inference only | SPAN merge central | most central window | SPAN | done | 0.932 | 0.951 | 0.970 | 0.975 | 0.976 | 0.855 | 0.908 | 0.976 | – |
| Inference only | SPAN merge avg | span scores averaged over windows | SPAN | done | 0.932 | 0.951 | 0.970 | 0.975 | 0.976 | 0.853 | 0.908 | 0.976 | – |
| Inference only | SPAN overlap 64 | window overlap 96 → 64 | SPAN | done | 0.932 | 0.951 | 0.970 | 0.975 | 0.976 | 0.853 | 0.907 | 0.976 | – |
| Inference only | v5.0 + label thresholds | per-label score thresholds (tuned on wm) | v5.0 | done | 0.928 | 0.941 | 0.962 | 0.973 | 0.975 | 0.852 | – | – | – |
| Inference only | SPAN + label thresholds | per-label score thresholds (tuned on wm) | SPAN | done | 0.924 | 0.953 | 0.971 | 0.971 | 0.973 | 0.836 | – | – | – |

### Coverage (share of gold PII spans overlapped by any prediction)

| Group | Experiment | Status | struct | wm | synth | real | mixed | Kiii |
|---|---|---|---|---|---|---|---|---|
| Baselines | v4.1 | done | 0.919 | 0.908 | 0.970 | 0.998 | 0.997 | 0.999 |
| Baselines | v4.2 | done | 0.895 | 0.896 | 0.945 | 0.996 | 0.993 | 0.998 |
| Baselines | v4 MiniLM | done | 0.922 | 0.916 | 0.975 | 0.997 | 0.995 | 0.999 |
| Stage 1: data | pilot r1 0% frames | done | 0.932 | – | – | 0.995 | – | – |
| Stage 1: data | pilot r1 10% | done | 0.987 | – | – | 0.994 | – | – |
| Stage 1: data | pilot r1 25% | done | 0.993 | – | – | 0.993 | – | – |
| Stage 1: data | pilot r1 40% | done | 0.996 | – | – | 0.993 | – | – |
| Stage 1: data | pilot r2 0% frames | done | 0.933 | 0.957 | – | 0.995 | – | – |
| Stage 1: data | pilot r2 10% | done | 0.996 | 0.991 | – | 0.995 | – | – |
| Stage 1: data | pilot r2 25% (chosen) | done | 0.996 | 0.994 | – | 0.995 | – | – |
| Stage 1: data | pilot r2 40% | done | 0.997 | 0.993 | – | 0.993 | – | – |
| Stage 1: data | **v5.0** (A-v5data) | done | 0.993 | 0.993 | 0.995 | 0.998 | 0.996 | 0.999 |
| Stage 2: what trains | H1 | done | 0.936 | 0.959 | 0.985 | 0.973 | 0.967 | 0.968 |
| Stage 2: what trains | P1 | done | 0.990 | 0.996 | 0.995 | 0.996 | 0.994 | 0.996 |
| Stage 2: what trains | S1 | done | 0.986 | 0.995 | 0.999 | 0.996 | 0.993 | 0.997 |
| Stage 2: what trains | B1 | done | 0.976 | 0.986 | 0.990 | 0.995 | 0.992 | 0.998 |
| Stage 2: what trains | B1 lr 3e-4 | done | 0.974 | 0.985 | 0.992 | 0.995 | 0.993 | 0.998 |
| Stage 2: what trains | B′ | done | 0.992 | 0.993 | 0.995 | 0.998 | 0.997 | 0.999 |
| Deep head | AL2 | done | 0.987 | 0.995 | 0.999 | 0.997 | 0.995 | 0.998 |
| Deep head | AL4 | done | 0.990 | 0.996 | 0.998 | 0.997 | 0.997 | 0.998 |
| Deep head | AL8 | done | 0.991 | 0.995 | 0.999 | 0.997 | 0.996 | 0.998 |
| Heads | CRF | done | 0.992 | 0.994 | 0.993 | 0.995 | 0.994 | 0.999 |
| Heads | **SPAN** | done | 0.978 | 0.979 | 0.993 | 0.996 | 0.995 | 0.999 |
| Heads | SPAN-CE | done | 0.981 | 0.986 | 0.993 | 0.996 | 0.994 | 0.999 |
| Heads | SPAN-BS | done | 0.990 | 0.985 | 0.994 | 0.996 | 0.994 | 0.999 |
| Encoders (softmax head) | mmBERT-small | done | 0.974 | 0.991 | 0.996 | 0.997 | 0.996 | 0.999 |
| Encoders (softmax head) | mmBERT-base | done | 0.980 | 0.993 | 0.997 | 0.998 | 0.998 | 0.999 |
| Encoders (softmax head) | MiniLM-L12 | done | 0.990 | 0.995 | 0.998 | 0.994 | 0.990 | 0.996 |
| Encoders (softmax head) | MiniLM-L6 | done | 0.985 | 0.994 | 0.997 | 0.994 | 0.989 | 0.994 |
| Encoders + span | mmBERT-small + span | done | 0.966 | 0.979 | 0.988 | 0.996 | 0.995 | 0.998 |
| Encoders + span | MiniLM-L12 + span | done | 0.983 | 0.982 | 0.991 | 0.993 | 0.992 | 0.998 |
| Encoders + span | MiniLM-L6 + span | done | 0.973 | 0.977 | 0.986 | 0.990 | 0.985 | 0.997 |
| Encoders + boundary-smoothed span | MiniLM-L12 + span-BS | done | 0.980 | 0.986 | 0.989 | 0.993 | 0.991 | 0.997 |
| Encoders + boundary-smoothed span | MiniLM-L6 + span-BS | done | 0.982 | 0.980 | 0.989 | 0.991 | 0.987 | 0.996 |
| Encoders + boundary-smoothed span | mDeBERTa-v3-base + span-BS | done | 0.987 | 0.986 | 0.990 | 0.997 | 0.997 | 0.999 |
| Encoders + boundary-smoothed span | InfoXLM-base + span-BS | done | 0.989 | 0.986 | 0.995 | 0.996 | 0.993 | 0.998 |
| Encoders + boundary-smoothed span | XLM-V-base + span-BS | done | 0.986 | 0.986 | 0.996 | 0.996 | 0.996 | 0.998 |
| Encoders + boundary-smoothed span | XLM-R large + span-BS | done | 0.989 | 0.988 | 0.999 | 0.997 | 0.997 | 0.998 |
| Inference only | v5.0 merge central | done | 0.993 | 0.993 | 0.995 | 0.998 | 0.996 | 0.999 |
| Inference only | v5.0 merge avg | done | 0.993 | 0.993 | 0.995 | 0.998 | 0.996 | 0.999 |
| Inference only | v5.0 overlap 64 | done | 0.993 | 0.993 | 0.995 | 0.998 | 0.996 | 0.999 |
| Inference only | SPAN merge central | done | 0.978 | 0.979 | 0.993 | 0.996 | 0.995 | 0.998 |
| Inference only | SPAN merge avg | done | 0.978 | 0.979 | 0.993 | 0.996 | 0.995 | 0.999 |
| Inference only | SPAN overlap 64 | done | 0.978 | 0.979 | 0.993 | 0.996 | 0.995 | 0.999 |
| Inference only | v5.0 + label thresholds | done | 0.961 | 0.980 | 0.979 | 0.992 | 0.989 | 0.993 |
| Inference only | SPAN + label thresholds | done | 0.936 | 0.973 | 0.987 | 0.986 | 0.986 | 0.967 |

### Held-out test sets

Scored once per model, after every v5 choice was made on the dev sets (struct = ml_struct_test, wm = ml_wm_test, wm_known = ml_wm_known, the same grid with accounts in known institution formats such as IBAN, IBKR, Fidelity). Same scoring as above.

| Group | Experiment | Status | struct_test strict | wm_test strict | wm_known strict | struct_test coverage | wm_test coverage | wm_known coverage |
|---|---|---|---|---|---|---|---|---|
| Baselines | v4.1 | done | 0.601 | 0.733 | 0.734 | 0.869 | 0.911 | 0.904 |
| Baselines | v4.2 | done | 0.574 | 0.606 | – | 0.891 | 0.917 | – |
| Baselines | v4 MiniLM | done | 0.592 | 0.718 | – | 0.865 | 0.919 | – |
| Stage 1: data | **v5.0** (A-v5data) | done | 0.948 | 0.949 | 0.946 | 0.985 | 0.996 | 0.995 |
| Stage 2: what trains | H1 | done | 0.775 | 0.615 | 0.608 | 0.962 | 0.959 | 0.953 |
| Stage 2: what trains | P1 | done | 0.941 | 0.938 | 0.938 | 0.976 | 0.993 | 0.994 |
| Stage 2: what trains | S1 | done | 0.940 | 0.943 | 0.944 | 0.980 | 0.995 | 0.995 |
| Stage 2: what trains | B1 | done | 0.919 | 0.890 | 0.889 | 0.973 | 0.987 | 0.985 |
| Stage 2: what trains | B1 lr 3e-4 | done | 0.913 | 0.888 | 0.885 | 0.963 | 0.987 | 0.987 |
| Stage 2: what trains | B′ | done | 0.942 | 0.948 | 0.946 | 0.982 | 0.996 | 0.995 |
| Deep head | AL2 | done | 0.934 | 0.946 | 0.942 | 0.987 | 0.995 | 0.995 |
| Deep head | AL4 | done | 0.935 | 0.950 | 0.947 | 0.983 | 0.996 | 0.995 |
| Deep head | AL8 | done | 0.952 | 0.951 | 0.948 | 0.989 | 0.995 | 0.994 |
| Heads | CRF | done | 0.957 | 0.951 | 0.949 | 0.987 | 0.996 | 0.995 |
| Heads | **SPAN** | done | 0.945 | 0.960 | 0.957 | 0.966 | 0.983 | 0.981 |
| Heads | SPAN-CE | done | 0.948 | 0.956 | 0.952 | 0.969 | 0.988 | 0.986 |
| Heads | SPAN-BS | done | 0.948 | 0.956 | 0.951 | 0.977 | 0.989 | 0.987 |
| Encoders (softmax head) | mmBERT-small | done | 0.923 | 0.932 | 0.927 | 0.965 | 0.993 | 0.991 |
| Encoders (softmax head) | mmBERT-base | done | 0.941 | 0.947 | 0.945 | 0.967 | 0.993 | 0.993 |
| Encoders (softmax head) | MiniLM-L12 | done | 0.918 | 0.938 | 0.935 | 0.988 | 0.995 | 0.994 |
| Encoders (softmax head) | MiniLM-L6 | done | 0.905 | 0.919 | 0.918 | 0.972 | 0.992 | 0.991 |
| Encoders + span | mmBERT-small + span | done | 0.931 | 0.945 | 0.941 | 0.943 | 0.979 | 0.977 |
| Encoders + span | MiniLM-L12 + span | done | 0.933 | 0.947 | 0.945 | 0.958 | 0.984 | 0.982 |
| Encoders + span | MiniLM-L6 + span | done | 0.929 | 0.931 | 0.931 | 0.960 | 0.977 | 0.975 |
| Encoders + boundary-smoothed span | MiniLM-L12 + span-BS | done | 0.912 | 0.945 | 0.943 | 0.962 | 0.987 | 0.984 |
| Encoders + boundary-smoothed span | MiniLM-L6 + span-BS | done | 0.933 | 0.932 | 0.932 | 0.971 | 0.981 | 0.977 |
| Encoders + boundary-smoothed span | mDeBERTa-v3-base + span-BS | done | 0.953 | 0.949 | 0.948 | 0.992 | 0.989 | 0.987 |
| Encoders + boundary-smoothed span | InfoXLM-base + span-BS | done | 0.946 | 0.947 | 0.946 | 0.975 | 0.990 | 0.987 |
| Encoders + boundary-smoothed span | XLM-V-base + span-BS | done | 0.942 | 0.957 | 0.958 | 0.964 | 0.990 | 0.989 |
| Encoders + boundary-smoothed span | XLM-R large + span-BS | done | 0.970 | 0.969 | 0.967 | 0.987 | 0.994 | 0.992 |

### Not run (dropped, cancelled, on hold or proposed)

| Group | Experiment | Change vs parent | Parent | Why not run |
|---|---|---|---|---|
| Stage 2: what trains | B′ fresh head | B′ with the classifier re-initialised | B′ | stopped at 28% (B series dropped) |
| Stage 2: what trains | B2 (DAPT + adapters) | per-language adapters on a financial-DAPT encoder | B1 | dropped (B series) |
| Stage 2: what trains | DAPT row (H2 / P2 / A2) | multilingual financial DAPT before fine-tuning | v5.0 | planned, not built |
| Deep head | AL16 | +16 layers | v5.0 | cancelled by user |
| Deep head | HL2 / HL4 / HL8 | frozen encoder + 2/4/8 layers | H1 | dropped (head-only not promising) |
| Encoders (softmax head) | EuroBERT-210m | encoder → EuroBERT-210m | v5.0 | dropped by user |
| Encoders (softmax head) | Qwen3-Embedding-0.6B | encoder → Qwen3-Embedding-0.6B | v5.0 | dropped by user |
| Encoders + span | mmBERT-base + span | span head | mmBERT-base | cancelled by user |
| Inference only | checksum post-filter | drop IBAN-shaped accounts failing mod-97 | SPAN | not testable (synthetic IBANs invalid) |
| Inference only | int8 (all Linear) | dynamic int8 encoder, CPU (300-doc sample: F1 ≈ 0.50) | v5.0 | paused by user |
| Validation | seed runs (v5.0, SPAN × 2 seeds) | seed 42 → 2 more seeds | v5.0 / SPAN | proposed, not queued |
<!-- v5-experiments:end -->

## Repository layout

| Path | Contents |
|---|---|
| `src/` | data builders, training, prediction, evaluation |
| `configs/` | data sources, training recipes, account-format registry |
| `scripts/english/`, `scripts/v4/`, `scripts/v5/` | training and evaluation queues per series (run from anywhere; they `cd` to the repo) |
| `results/english/<v1 \| v2 \| v2.1 \| v3>/` | English-only models |
| `results/v4/<A \| B64 \| A-minilm \| A-pilot>/` | v4.1 / v4.2 / v4.3 and the v4 pilot; `train_set/`, `latency/`, `router/` |
| `results/v5/runs/<run>/` | one folder per v5 model: every benchmark, inference-only variants, latency |
| `results/v5/{pilots,sweeps,train_set,summary}/` | frame-share pilots, LR sweeps, training-data coverage, cross-run tables |
| `results/v5/latency/{gpu,cpu}_latency/` | latency by document size (`summary.tsv`) |
| `model_card/`, `reports/` | model cards per release, research reports |

Evaluation scripts given an output path directly in `results/` write it into its group folder
(`src/results_layout.py`); `python results_layout.py` files anything written there by a shell redirect.

## Labels

8 entity types, BIO-tagged (17 labels incl. `O`):

| Label | Covers |
|---|---|
| `PERSON` | personal names |
| `BUSINESS` | company / organization names |
| `ADDRESS` | residential and office addresses and their parts (street, building, unit, city, postcode, state, country), single- or multi-line. A bare place name in running text ("expanding into France") is **not** an address (v2.1+) |
| `DOB` | dates of birth (not other dates) |
| `ACCOUNT` | bank / card / IBAN / customer / policy account identifiers (not amounts, PINs, CVVs, SWIFT / routing codes) |
| `PHONE` | telephone numbers |
| `EMAIL` | email addresses |
| `TIN` | tax identification numbers (incl. SSN) |

Source-label mapping lives in [`configs/label_map.yaml`](configs/label_map.yaml): look-alikes such as
amounts, PINs, CVVs, SWIFT codes and ordinary dates are deliberately labelled `O` so the model learns
to reject them.

## Results (v3.0)

v3 = domain-adaptive pretraining of the base model (replaced-token detection, DeBERTa-v3's own
objective) on 178M tokens of public text — PII task documents, regulation from India / UAE / UK / EU /
US plus central-bank statements, 10-K risk and MD&A sections, business e-mail and investing Q&A —
then exactly v2.1's fine-tuning. Same data and labels as v2.1:

| | v2.1 | **v3** |
|---|---|---|
| Test F1 (6k docs) | 0.925 | **0.925** |
| Test ADDRESS / ACCOUNT / TIN | 0.905 / 0.954 / 0.963 | **0.914 / 0.961 / 0.969** |
| Test BUSINESS | **0.833** | 0.821 |
| Stress F1 (1.8k unseen-layout records) | 0.921 | **0.936** |
| Stress ADDRESS / PHONE | 0.792 / 0.883 | **0.848 / 0.955** |
| Stress TIN | **0.885** | 0.849 |
| Addresses with exact boundaries (stress): multi-line / India | 58.9% / 59.5% | **69.7% / 78.5%** |
| Look-alike codes tagged as PII (stress) | **4.6%** | 6.6% |

Pretraining halved held-out masked-LM loss on every domain (e.g. regulation 3.53 → 1.38); the
PII-task gain is modest and concentrated in addresses and phones. Details, corpus and
limitations: [`model_card/v3.md`](model_card/v3.md).

## Results (v2.1)

Strict = exact character boundaries + label. v2 is re-scored on the same v2.1-labelled data.

**Test set** — 6,000 documents from the datasets' official test/validation splits:

| | v2 F1 | **v2.1 F1** | v2.1 Precision | v2.1 Recall | Support |
|---|---|---|---|---|---|
| **All** | 0.898 | **0.925** | 0.911 | 0.940 | 21,974 |
| ADDRESS | 0.767 | **0.905** | 0.895 | 0.914 | 3,514 |
| DOB | 0.975 | 0.980 | 0.966 | 0.995 | 770 |
| ACCOUNT | 0.947 | 0.954 | 0.933 | 0.976 | 1,862 |
| TIN | 0.962 | 0.963 | 0.942 | 0.984 | 876 |
| EMAIL | 0.975 | 0.973 | 0.961 | 0.985 | 2,807 |
| PHONE | 0.970 | 0.973 | 0.962 | 0.984 | 1,669 |
| PERSON | 0.939 | 0.936 | 0.934 | 0.937 | 6,908 |
| BUSINESS | 0.828 | 0.833 | 0.793 | 0.876 | 3,568 |

(v2's ADDRESS score here is mostly the new definition: v2 tags 95% of bare place names as ADDRESS, v2.1 2.7%.)

**Stress set** — 1,800 short records with layouts not used in training (form records, e-mail
signatures, upper-case envelopes, CSV rows) and real addresses held out from training:
F1 **0.921** (v2 0.816); ADDRESS **0.792** (v2 0.519), PERSON 0.982 (v2 0.852), BUSINESS 0.962 (v2 0.796).

**Addresses** (stress set): found 99.0% (v2 86.2%); exact boundaries single-line **89.4%** (v2 68.6%),
multi-line **58.9%** (v2 11.1%); per country exact: UK 84.6%, Malaysia 85.7%, Singapore 73.1%,
Hong Kong 73.0%, UAE 64.9%, India 59.5%.

**Look-alikes wrongly tagged as PII** (stress set): amounts 1.0%, PIN / CVV / SWIFT / expiry codes 4.6%,
ordinary dates 0.0%; bare place names on the test set 2.7%.

Full numbers: [`results/`](results) (`*_v21data.*`) and [`model_card/v2.1.md`](model_card/v2.1.md).

**Known limitations.** Trained on synthetic documents plus real addresses placed in templates; not yet
measured on real documents. Multi-line address boundaries are the weakest part (58.9% exact);
BUSINESS remains the weakest entity (0.833). UAE and East-Asian addresses are under-represented in the
public data; Asian IDs (PAN, MyKad, NPWP) and company-name forms (Sdn Bhd, Pte Ltd) are rare.
**Regression vs v2:** a hyphenated account number after a bare `A/C` cue in a very short input
(`A/C 0123-456789-0`) can be missed or clipped; `Account No: …` and sentence contexts are fine. **v3 is English only**: zero-shot on 11 Asian / Middle-Eastern languages it scores 0.28–0.76 F1 and misses most native-script names, companies and addresses (Thai 0%); see [`model_card/multilingual_zero_shot.md`](model_card/multilingual_zero_shot.md). Use v4.1 for those languages.

## Quick start (inference only)

```bash
git clone https://github.com/zackhng/pii-token-classifier.git
cd pii-token-classifier

# weights -> outputs/deberta-v3-xsmall-pii-v3/
curl -L -o model.zip https://github.com/zackhng/pii-token-classifier/releases/download/v3.0/deberta-v3-xsmall-pii-v3.zip
mkdir -p outputs/deberta-v3-xsmall-pii-v3 && unzip model.zip -d outputs/deberta-v3-xsmall-pii-v3
# Windows PowerShell: Expand-Archive model.zip -DestinationPath outputs\deberta-v3-xsmall-pii-v3

python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128   # NVIDIA GPU
# pip install torch==2.8.0                                                    # CPU-only / macOS
pip install -r requirements.txt

cd src
python predict.py --model_dir ../outputs/deberta-v3-xsmall-pii-v3 --text "Maria Gonzalez
Level 12, 8 Marina Boulevard
Singapore 018981
DOB 04/12/1987
Account No: 0123-456789-0
Balance: SGD 12,450.00"
```

Multilingual (v5.0):

```bash
curl -L -o model.zip https://github.com/zackhng/pii-token-classifier/releases/download/v5.0/xlmr-base-pii-ml-v5.0.zip
mkdir -p outputs/xlmr-base-pii-ml-v5.0 && unzip model.zip -d outputs/xlmr-base-pii-ml-v5.0
cd src
python predict.py --model_dir ../outputs/xlmr-base-pii-ml-v5.0 --text "Hi Tan Wei Ming (陈伟明), 已经 transfer SGD 25,000 到 account 123-456789-001, ref TT240918。คุณสมชาย ใจดี เลขที่บัญชี 123-4-56789-0"
# -> PERSON 'Tan Wei Ming', PERSON '陈伟明', ACCOUNT '123-456789-001', PERSON 'สมชาย ใจดี', ACCOUNT '123-4-56789-0'
#    (the amount and the reference stay unlabelled)
```

v4.1 (`xlmr-base-pii-ml-v4.1.zip`) and v4.3 (`minilm-pii-ml-v4.3.zip`) unzip and load the same way.
v4.2 unzips the same way into `outputs/xlmr-base-pii-adapters-v4.2`. It downloads `xlm-roberta-base`
(1.1 GB) from the Hugging Face Hub on first use and routes each blank-line block of the input to that
language's adapter (`predict(text, lang="th")` forces one).

## Usage

```python
from predict import PIIPredictor  # run from src/

p = PIIPredictor("../outputs/deberta-v3-xsmall-pii-v3")   # or -v2.1, -v2
p.predict("Hi, I'm Maria Gonzalez, reach me at maria.g@outlook.com or (415) 555-0199.")
# [{'start': 8, 'end': 22, 'label': 'PERSON', 'score': 1.0, 'text': 'Maria Gonzalez'}, ...]
```

`PIIPredictor` handles sliding windows for long documents and maps predictions back to character spans.
From v2.1 the model is shown line breaks and tabs as marker tokens; the setting is stored in the model's
`config.json` (`"visible_breaks": true`) and applied automatically, so v1/v2 weights keep their original
tokenization. If loading the model directly with transformers ≥ 5, pass `dtype=torch.float32` and use
`tokenize_bio.window_encode` for the same input handling.

### Latency

One 100,000-character document (≈23k tokens → 81 windows of 384 tokens, 322 PII spans found),
`PIIPredictor.predict`, fp32, model already loaded (same architecture for all versions):

| Device | Latency |
|---|---|
| RTX 5060 Ti 16 GB | ~0.5 s |
| CPU (6 threads) | ~8 s |

Model load adds ~1 s once. Latency scales roughly linearly with document length. Full breakdown: [`LATENCY.md`](LATENCY.md).

## Training data (v2.1)

74,036 English training documents over 3 epochs (88,911 windows of 384 tokens):

| Source | Train docs | License |
|---|---|---|
| [nvidia/Nemotron-PII](https://huggingface.co/datasets/nvidia/Nemotron-PII) | 29,528 (20,000 + 9,528 extra docs with a date of birth) | CC-BY-4.0 |
| [gretelai/synthetic_pii_finance_multilingual](https://huggingface.co/datasets/gretelai/synthetic_pii_finance_multilingual) (English) | 15,509 | Apache-2.0 |
| [ai4privacy/pii-masking-openpii-1.5m](https://huggingface.co/datasets/ai4privacy/pii-masking-openpii-1.5m) (English) | 14,999 | CC-BY-4.0 |
| Form-style records ([`src/snippets.py`](src/snippets.py)), built from train-split entity values | 6,000 | — |
| Address records ([`src/snippets.py`](src/snippets.py)): letters, KYC forms, statement headers, sign-offs, attention blocks, sentences | 8,000 | — |

Real addresses for the address records ([`src/address_sources.py`](src/address_sources.py)), 20,848 in total
(90% train / 10% stress only):

| Source | Addresses | License |
|---|---|---|
| [ellenhp/libpostal](https://huggingface.co/datasets/ellenhp/libpostal) — OpenStreetMap addresses | 17,350 | ODbL |
| [ellenhp/libpostal](https://huggingface.co/datasets/ellenhp/libpostal) — UK OpenAddresses | 1,499 | OpenAddresses |
| [gagan1985/indian-addresses-raw](https://huggingface.co/datasets/gagan1985/indian-addresses-raw) — registered offices, bank branches | 1,999 | Apache-2.0 |

India 3,499 · UK 3,000 · Singapore 2,990 · Hong Kong 2,500 · Indonesia 1,500 · Malaysia 1,500 ·
Philippines 1,000 · UAE 820 · Kuwait / Saudi Arabia 600 each · Thailand 560 · Bangladesh, Sri Lanka,
Pakistan 500 each · China 355 · Qatar, Vietnam, Oman, Korea, Taiwan, Japan, Bahrain < 150 each.

Processing ([`src/build_dataset.py`](src/build_dataset.py)):
- Adjacent same-label spans are merged (first + last name → one `PERSON`; street + city + zip → one `ADDRESS`).
- A place-only span (city / state / country, no street or postcode) is `ADDRESS` only after an address
  field cue ("City:", "Address:"); otherwise `O`. Gretel's unlabelled continuation after a street
  address is masked.
- AI4Privacy `DATE` becomes `DOB` when a birth cue ("DOB", "born", "date of birth") directly precedes it, otherwise `O`.
- Train only: in half the documents, birth dates *and* ordinary dates are rewritten in the same random mix
  of ~25 formats and birth cues are varied ([`src/augment.py`](src/augment.py)).
- Validation (3,000) and test (6,000) documents are never augmented; train docs identical to a val/test doc are dropped.

## Evaluation

```bash
cd src
python evaluate.py                    # test set P/R/F1 (default model: outputs/deberta-v3-xsmall-pii-v2.1)
python evaluate.py --split stress     # stress set
python eval_confusion.py --split stress
```

`eval_confusion.py` reports how often look-alikes (amounts, currencies, ordinary dates, PIN / CVV /
SWIFT codes, bare place names) are tagged as PII, per-entity recall and exact-boundary rates,
PERSON↔BUSINESS swaps, and an ADDRESS breakdown by single- / multi-line and country.

## Reproduce

```bash
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

cd src
python download.py          # ~5 GB into data/raw
python address_sources.py   # real addresses -> data/addresses (~2.3 GB of libpostal / Indian data)
python build_dataset.py     # 74k train / 3k val / 6k test / 1.8k stress jsonl
python train.py             # ~22 min on an RTX 5060 Ti -> outputs/deberta-v3-xsmall-pii-v2.1
python evaluate.py
pytest ../tests             # span <-> BIO round-trip, multi-line spans, augmentation and record checks
```

Hyperparameters are in [`configs/train.yaml`](configs/train.yaml): lr 5e-5, batch 32, 3 epochs,
10% warmup, weight decay 0.01, bf16 mixed precision, seed 42, `visible_breaks: true`.

### v3: domain-adaptive pretraining, then the same fine-tuning

```bash
cd src
python build_pretrain_corpus.py   # 178M-token corpus -> data/pretrain (downloads EDGAR, Pile-of-Law, ...)
python pretrain_rtd.py            # ~2.4 h on an RTX 5060 Ti; checkpoints every 50M tokens -> outputs/dapt-v3
python train.py --model_name outputs/dapt-v3/tokens-178M --output_dir outputs/deberta-v3-xsmall-pii-v3
```

Settings in [`configs/pretrain.yaml`](configs/pretrain.yaml). The domain-adapted base model (before
PII fine-tuning) is attached to the v3.0 release as `deberta-v3-xsmall-dapt-wealth.zip`.
