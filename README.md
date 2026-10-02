# PII Token Classifier — DeBERTa-v3-xsmall

A BIO token-classification model for detecting personally identifiable information in English text,
fine-tuned from [`microsoft/deberta-v3-xsmall`](https://huggingface.co/microsoft/deberta-v3-xsmall)
following the [OpenMed](https://huggingface.co/OpenMed) PII recipe (Hugging Face Trainer, max length 384,
first-sub-token labelling, seqeval F1).

## Releases

| Version | Weights | Highlights | Model card |
|---|---|---|---|
| **v3.0** (current) | [`deberta-v3-xsmall-pii-v3.zip`](../../releases/tag/v3.0) | v2.1's fine-tuning on top of **domain-adaptive pretraining** (178M tokens of public financial / regulatory / RM-communication text): stress F1 0.921 → 0.936, ADDRESS 0.792 → 0.848, multi-line addresses exact 59% → 70% | [`model_card/v3.md`](model_card/v3.md) |
| v2.1 | [`deberta-v3-xsmall-pii-v2.1.zip`](../../releases/tag/v2.1) | ADDRESS: model sees line breaks / tabs (multi-line addresses), 20,848 real public addresses from SG, IN, UAE, UK, HK, ASEAN & Asia in letters / KYC forms / statements / signatures, bare place names no longer ADDRESS | [`model_card/v2.1.md`](model_card/v2.1.md) |
| v2.0 | [`deberta-v3-xsmall-pii-v2.zip`](../../releases/tag/v2.0) | Rejects look-alikes: DOB vs ordinary dates, account numbers vs amounts / codes; label-map fixes; DOB format augmentation; form-style records | [`model_card/v2.md`](model_card/v2.md) |
| v1.0 | [`deberta-v3-xsmall-pii.zip`](../../releases/tag/v1.0) | First release | [`model_card/v1.md`](model_card/v1.md) |

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
PII-task gain is modest and concentrated in addresses and phones. Details, corpus, curve and
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
(`A/C 0123-456789-0`) can be missed or clipped; `Account No: …` and sentence contexts are fine. English only.

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
