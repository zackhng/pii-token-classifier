# PII Token Classifier — DeBERTa-v3-xsmall

A BIO token-classification model for detecting personally identifiable information in English text,
fine-tuned from [`microsoft/deberta-v3-xsmall`](https://huggingface.co/microsoft/deberta-v3-xsmall)
following the [OpenMed](https://huggingface.co/OpenMed) PII recipe (Hugging Face Trainer, max length 384,
first-sub-token labelling, seqeval F1).

## Releases

| Version | Weights | Highlights | Model card |
|---|---|---|---|
| **v2.0** (current) | [`deberta-v3-xsmall-pii-v2.zip`](../../releases/tag/v2.0) | Rejects look-alikes: DOB vs ordinary dates, account numbers vs amounts / codes; label-map fixes; DOB format augmentation; form-style records | [`model_card/v2.md`](model_card/v2.md) |
| v1.0 | [`deberta-v3-xsmall-pii.zip`](../../releases/tag/v1.0) | First release | [`model_card/v1.md`](model_card/v1.md) |

## Labels

8 entity types, BIO-tagged (17 labels incl. `O`):

| Label | Covers |
|---|---|
| `PERSON` | personal names |
| `BUSINESS` | company / organization names |
| `ADDRESS` | street & postal addresses **and** geographic locations (city, state, country, …) |
| `DOB` | dates of birth (not other dates) |
| `ACCOUNT` | bank / card / IBAN / customer / policy account identifiers (not amounts, PINs, CVVs, SWIFT / routing codes) |
| `PHONE` | telephone numbers |
| `EMAIL` | email addresses |
| `TIN` | tax identification numbers (incl. SSN) |

Source-label mapping lives in [`configs/label_map.yaml`](configs/label_map.yaml): look-alikes such as
amounts, PINs, CVVs, SWIFT codes and ordinary dates are deliberately labelled `O` so the model learns
to reject them.

## Results (v2.0)

Strict = exact character boundaries + label. v1 is re-scored on the same v2-labelled data.

**Test set** — 6,000 documents from the datasets' official test/validation splits:

| | v1 F1 | **v2 F1** | v2 Precision | v2 Recall | Support |
|---|---|---|---|---|---|
| **All** | 0.916 | **0.925** | 0.909 | 0.941 | 23,348 |
| DOB | 0.888 | **0.975** | 0.954 | 0.997 | 770 |
| ACCOUNT | 0.885 | **0.947** | 0.928 | 0.967 | 1,862 |
| TIN | 0.951 | **0.962** | 0.937 | 0.989 | 876 |
| EMAIL | 0.972 | 0.975 | 0.967 | 0.984 | 2,807 |
| PHONE | 0.968 | 0.970 | 0.956 | 0.983 | 1,669 |
| PERSON | 0.935 | 0.939 | 0.944 | 0.933 | 6,908 |
| ADDRESS | 0.914 | 0.913 | 0.899 | 0.928 | 4,888 |
| BUSINESS | 0.829 | 0.828 | 0.779 | 0.885 | 3,568 |

**Stress set** — 1,000 short form-style records (key–value, tables, sentences) with keys and layouts
not used in training: F1 **0.893** (v1 0.729); DOB 0.995 (v1 0.530), ACCOUNT 0.890 (v1 0.606).

**Look-alikes wrongly tagged as PII** (stress set, lower is better): amounts / balances
**1.3%** (v1 10.9%), PIN / CVV / SWIFT / expiry codes **5.0%** (v1 10.7%), ordinary dates **0.1%** (v1 3.6%).

Full numbers: [`results/`](results) (`v2_test_*`, `v2_stress_*`) and [`model_card/v2.md`](model_card/v2.md).

**Known limitations.** Trained only on synthetic documents and not yet measured on real ones.
The tokenizer cannot see line breaks or tabs, so multi-line addresses can be missed or cut short;
ADDRESS (0.913) and BUSINESS (0.828) are the weakest entities. Few Asian-format IDs, company
names and addresses. English only. Details in the model card.

## Quick start (inference only)

```bash
git clone https://github.com/zackhng/pii-token-classifier.git
cd pii-token-classifier

# weights -> outputs/deberta-v3-xsmall-pii-v2/ (the default path used by the scripts)
curl -L -o model.zip https://github.com/zackhng/pii-token-classifier/releases/download/v2.0/deberta-v3-xsmall-pii-v2.zip
mkdir -p outputs/deberta-v3-xsmall-pii-v2 && unzip model.zip -d outputs/deberta-v3-xsmall-pii-v2
# Windows PowerShell: Expand-Archive model.zip -DestinationPath outputs\deberta-v3-xsmall-pii-v2

python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128   # NVIDIA GPU
# pip install torch==2.8.0                                                    # CPU-only / macOS
pip install -r requirements.txt

cd src
python predict.py --text "Hi, I'm Maria Gonzalez, DOB 04/12/1987, account 0123-456789-0, balance SGD 12,450.00"
```

## Usage

```python
from predict import PIIPredictor  # run from src/

p = PIIPredictor("../outputs/deberta-v3-xsmall-pii-v2")
p.predict("Hi, I'm Maria Gonzalez, reach me at maria.g@outlook.com or (415) 555-0199.")
# [{'start': 8, 'end': 22, 'label': 'PERSON', 'score': 1.0, 'text': 'Maria Gonzalez'}, ...]
```

`PIIPredictor` handles sliding windows for long documents and maps predictions back to character spans.
If loading the model directly with transformers ≥ 5, pass `dtype=torch.float32`.

### Latency

One 100,000-character document (≈23k tokens → 81 windows of 384 tokens, 322 PII spans found),
`PIIPredictor.predict`, fp32, model already loaded (same architecture for v1 and v2):

| Device | Latency |
|---|---|
| RTX 5060 Ti 16 GB | ~0.5 s |
| CPU (6 threads) | ~8 s |

Model load adds ~1 s once. Latency scales roughly linearly with document length. Full breakdown: [`LATENCY.md`](LATENCY.md).

## Training data (v2.0)

66,036 English training documents over 3 epochs (77,812 windows of 384 tokens):

| Source | Train docs | License |
|---|---|---|
| [nvidia/Nemotron-PII](https://huggingface.co/datasets/nvidia/Nemotron-PII) | 29,528 (20,000 + 9,528 extra docs with a date of birth) | CC-BY-4.0 |
| [gretelai/synthetic_pii_finance_multilingual](https://huggingface.co/datasets/gretelai/synthetic_pii_finance_multilingual) (English) | 15,509 | Apache-2.0 |
| [ai4privacy/pii-masking-openpii-1.5m](https://huggingface.co/datasets/ai4privacy/pii-masking-openpii-1.5m) (English) | 14,999 | CC-BY-4.0 |
| Form-style records ([`src/snippets.py`](src/snippets.py)), built from train-split entity values | 6,000 | — |

Processing ([`src/build_dataset.py`](src/build_dataset.py)):
- Adjacent same-label spans are merged (first + last name → one `PERSON`; street + city + zip → one `ADDRESS`).
- AI4Privacy `DATE` becomes `DOB` when a birth cue ("DOB", "born", "date of birth") directly precedes it, otherwise `O`.
- Train only: in half the documents, birth dates *and* ordinary dates are rewritten in the same random mix
  of ~25 formats (`22/05/1987`, `22 May 1987`, `22-05-87`, `05/87`, …) and birth cues are varied, so
  date format is no longer a shortcut for DOB ([`src/augment.py`](src/augment.py)).
- Validation (3,000) and test (6,000) documents are never augmented; train docs identical to a val/test doc are dropped.

## Evaluation

```bash
cd src
python evaluate.py --model_dir ../outputs/deberta-v3-xsmall-pii-v2                  # test set P/R/F1
python evaluate.py --model_dir ../outputs/deberta-v3-xsmall-pii-v2 --split stress    # form-style stress set
python eval_confusion.py --model_dir ../outputs/deberta-v3-xsmall-pii-v2 --split stress
```

`eval_confusion.py` reports how often look-alikes (amounts, currencies, ordinary dates, PIN / CVV /
SWIFT codes) are tagged as PII, plus per-entity recall and PERSON↔BUSINESS swaps.

## Reproduce

```bash
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

cd src
python download.py        # ~5 GB into data/raw
python build_dataset.py   # 66k train / 3k val / 6k test / 1k stress jsonl
python train.py           # ~20 min on an RTX 5060 Ti -> outputs/deberta-v3-xsmall-pii-v2
python evaluate.py
pytest ../tests           # span <-> BIO round-trip, augmentation and record checks
```

Hyperparameters are in [`configs/train.yaml`](configs/train.yaml): lr 5e-5, batch 32, 3 epochs,
10% warmup, weight decay 0.01, bf16 mixed precision, seed 42.
