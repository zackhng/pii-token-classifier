# PII Token Classifier — DeBERTa-v3-xsmall

A BIO token-classification model for detecting personally identifiable information in English text,
fine-tuned from [`microsoft/deberta-v3-xsmall`](https://huggingface.co/microsoft/deberta-v3-xsmall)
following the [OpenMed](https://huggingface.co/OpenMed) PII recipe (Hugging Face Trainer, max length 384,
first-sub-token labelling, seqeval F1).

**Weights:** download `deberta-v3-xsmall-pii.zip` from the [v1.0 release](../../releases/tag/v1.0).

## Labels

8 entity types, BIO-tagged (17 labels incl. `O`):

| Label | Covers |
|---|---|
| `PERSON` | personal names |
| `BUSINESS` | company / organization names |
| `ADDRESS` | street & postal addresses **and** geographic locations (city, state, country, …) |
| `DOB` | dates of birth |
| `ACCOUNT` | bank / card / customer / other account identifiers |
| `PHONE` | telephone numbers |
| `EMAIL` | email addresses |
| `TIN` | tax identification numbers (incl. SSN) |

Source-label mapping lives in [`configs/label_map.yaml`](configs/label_map.yaml).

## Results

Held-out test set: 6,000 documents (2,000 from each dataset's official test/validation split).
Strict = exact character boundaries + label; lenient = any overlap + label.

| | Precision | Recall | F1 (strict) | F1 (lenient) | Support |
|---|---|---|---|---|---|
| **All** | 0.902 | 0.942 | **0.922** | 0.943 | 22,692 |
| EMAIL | 0.962 | 0.981 | 0.972 | 0.987 | 2,807 |
| DOB | 0.948 | 0.981 | 0.964 | 0.966 | 467 |
| PHONE | 0.941 | 0.982 | 0.961 | 0.967 | 1,669 |
| TIN | 0.930 | 0.967 | 0.948 | 0.953 | 853 |
| ACCOUNT | 0.920 | 0.972 | 0.945 | 0.958 | 1,615 |
| PERSON | 0.931 | 0.943 | 0.937 | 0.955 | 6,825 |
| ADDRESS | 0.897 | 0.931 | 0.914 | 0.945 | 4,888 |
| BUSINESS | 0.781 | 0.883 | 0.829 | 0.864 | 3,568 |

Per source: Nemotron 0.968 · AI4Privacy 0.972 · Gretel 0.840. Full numbers: [`results/test_metrics.json`](results/test_metrics.json).

**Known limitations.** Trained only on long-form synthetic documents; on short, terse inputs it can miss
entities (e.g. `DOB 04/12/1987`, a bare account number, or a well-known bank name) and can clip
boundaries on hyphenated IDs. English only.

## Quick start (inference only)

```bash
git clone https://github.com/zackhng/pii-token-classifier.git
cd pii-token-classifier

# weights -> outputs/deberta-v3-xsmall-pii/ (the default path used by the scripts)
curl -L -o model.zip https://github.com/zackhng/pii-token-classifier/releases/download/v1.0/deberta-v3-xsmall-pii.zip
mkdir -p outputs/deberta-v3-xsmall-pii && unzip model.zip -d outputs/deberta-v3-xsmall-pii
# Windows PowerShell: Expand-Archive model.zip -DestinationPath outputs\deberta-v3-xsmall-pii

python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128   # NVIDIA GPU
# pip install torch==2.8.0                                                    # CPU-only / macOS
pip install -r requirements.txt

cd src
python predict.py --text "Hi, I'm Maria Gonzalez, reach me at maria.g@outlook.com"
```

## Usage

```python
from predict import PIIPredictor  # run from src/

p = PIIPredictor("../outputs/deberta-v3-xsmall-pii")
p.predict("Hi, I'm Maria Gonzalez, reach me at maria.g@outlook.com or (415) 555-0199.")
# [{'start': 8, 'end': 22, 'label': 'PERSON', 'score': 1.0, 'text': 'Maria Gonzalez'}, ...]
```

`PIIPredictor` handles sliding windows for long documents and maps predictions back to character spans.
If loading the model directly with transformers ≥ 5, pass `dtype=torch.float32`.

### Latency

One 100,000-character document (≈23k tokens → 81 windows of 384 tokens, 322 PII spans found),
`PIIPredictor.predict`, fp32, model already loaded:

| Device | Latency |
|---|---|
| RTX 5060 Ti 16 GB | ~0.5 s |
| CPU (6 threads) | ~8 s |

Model load adds ~1 s once. Latency scales roughly linearly with document length.

## Training data

50,000 English training documents over 3 epochs:

| Dataset | Train docs | License |
|---|---|---|
| [nvidia/Nemotron-PII](https://huggingface.co/datasets/nvidia/Nemotron-PII) | 20,000 | CC-BY-4.0 |
| [gretelai/synthetic_pii_finance_multilingual](https://huggingface.co/datasets/gretelai/synthetic_pii_finance_multilingual) (English) | 15,000 | Apache-2.0 |
| [ai4privacy/pii-masking-openpii-1.5m](https://huggingface.co/datasets/ai4privacy/pii-masking-openpii-1.5m) (English) | 15,000 | CC-BY-4.0 |

Adjacent same-label spans are merged (first + last name → one `PERSON`; street + city + zip → one `ADDRESS`).
AI4Privacy `DATE` spans are loss-masked because they don't say whether a date is a birth date.

## Reproduce

```bash
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

cd src
python download.py        # ~5 GB into data/raw
python build_dataset.py   # 50k train / 3k val / 6k test jsonl
python train.py           # ~16 min on an RTX 5060 Ti
python evaluate.py
pytest ../tests           # span <-> BIO round-trip checks
```

Hyperparameters are in [`configs/train.yaml`](configs/train.yaml): lr 5e-5, batch 32, 3 epochs,
10% warmup, weight decay 0.01, bf16 mixed precision.
