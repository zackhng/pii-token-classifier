"""Shared helpers: config loading and the BIO label schema."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent

DATASETS = {
    "nemotron": "nvidia/Nemotron-PII",
    "gretel": "gretelai/synthetic_pii_finance_multilingual",
    "ai4privacy": "ai4privacy/pii-masking-openpii-1.5m",
    "pfi": "ai4privacy/pii-masking-financial-pfi-400k",   # gated: needs HF login + approval
}

IGNORE = "IGNORE"
DATE_CUE = "DATE_CUE"   # label-map target: DOB if a birth cue precedes the date, else O


def load_yaml(name: str) -> dict:
    with open(ROOT / "configs" / name, encoding="utf-8") as f:
        return yaml.safe_load(f)


def bio_labels(entities: list[str]) -> list[str]:
    labels = ["O"]
    for e in entities:
        labels += [f"B-{e}", f"I-{e}"]
    return labels


def label_maps(entities: list[str]):
    labels = bio_labels(entities)
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}
    return labels, label2id, id2label
