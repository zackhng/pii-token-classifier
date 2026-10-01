"""Normalize raw sources to {id, source, text, spans} jsonl with the target label set.

Pipeline per source: English filter -> label map -> overlap cleanup -> merge adjacent
same-label spans -> seeded subsample (50k train by default) -> train/val/test jsonl in data/processed.
"""
import json
import random
import re
from collections import Counter

from datasets import load_from_disk

from common import IGNORE, ROOT, load_yaml

# Gap text allowed between two same-label spans for them to be merged into one.
# Addresses span commas/newlines ("12 Main St,\nSpringfield, IL"); other labels only spaces.
MERGE_GAP = {"ADDRESS": re.compile(r"[\s,]*"), "DEFAULT": re.compile(r"[ \t]*")}

# (text column, spans column, official eval split) per source
SOURCE_FIELDS = {
    "nemotron": ("text", "spans", "test"),
    "gretel": ("generated_text", "pii_spans", "test"),
    "ai4privacy": ("source_text", "privacy_mask", "validation"),
}


def is_english(source: str, row: dict) -> bool:
    if source == "nemotron":
        return True
    lang = str(row.get("language", "")).strip().lower()
    return lang == "en" or lang.startswith("english")


def parse_spans(raw) -> list[dict]:
    if raw is None:
        return []
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return []
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            import ast
            raw = ast.literal_eval(raw)
    return list(raw)


def clean_spans(text: str, spans: list[dict], mapping: dict, stats: Counter) -> list[dict]:
    mapped = []
    for s in spans:
        target = mapping.get(s["label"])
        if target is None:
            continue
        start, end = int(s["start"]), int(s["end"])
        if not (0 <= start < end <= len(text)) or not text[start:end].strip():
            stats["invalid"] += 1
            continue
        # trim whitespace so token alignment starts/ends on real characters
        while text[start].isspace():
            start += 1
        while text[end - 1].isspace():
            end -= 1
        mapped.append({"start": start, "end": end, "label": target})

    # overlaps: keep the longer span
    mapped.sort(key=lambda s: (s["start"], -(s["end"] - s["start"])))
    kept: list[dict] = []
    for s in mapped:
        if kept and s["start"] < kept[-1]["end"]:
            stats["overlap_dropped"] += 1
            if (s["end"] - s["start"]) > (kept[-1]["end"] - kept[-1]["start"]):
                kept[-1] = s
            continue
        kept.append(s)

    # merge adjacent same-label spans (first+last name, street+city+zip, ...)
    merged: list[dict] = []
    for s in kept:
        if merged and s["label"] == merged[-1]["label"] and s["label"] != IGNORE:
            prev = merged[-1]
            gap_re = MERGE_GAP.get(s["label"], MERGE_GAP["DEFAULT"])
            if gap_re.fullmatch(text[prev["end"]:s["start"]]):
                prev["end"] = s["end"]
                stats["merged"] += 1
                continue
        merged.append(dict(s))
    return merged


def normalize(source: str, split, mapping: dict, stats: Counter) -> list[dict]:
    text_col, span_col, _ = SOURCE_FIELDS[source]
    out = []
    for i, row in enumerate(split):
        if not is_english(source, row):
            continue
        text = row[text_col]
        if not text or not text.strip():
            continue
        spans = clean_spans(text, parse_spans(row[span_col]), mapping, stats)
        rid = row.get("uid", row.get("index", i))
        out.append({"id": f"{source}-{rid}", "source": source, "text": text, "spans": spans})
    return out


def english_only(source: str, split):
    """Cheap column-level filter before the per-row python loop (matters for 1.6M rows)."""
    if source == "nemotron" or "language" not in split.column_names:
        return split
    return split.filter(lambda b: [is_english(source, {"language": l}) for l in b["language"]],
                        batched=True, num_proc=4)


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def allocate(caps: dict, available: dict) -> dict:
    """Per-source train counts; shortfall of small sources is spread over the others."""
    alloc = {k: min(caps[k], available[k]) for k in caps}
    short = sum(caps.values()) - sum(alloc.values())
    while short > 0:
        room = {k: available[k] - alloc[k] for k in alloc if available[k] > alloc[k]}
        if not room:
            break
        share = max(1, short // len(room))
        for k, r in room.items():
            add = min(share, r, short)
            alloc[k] += add
            short -= add
    return alloc


def main():
    cfg = load_yaml("train.yaml")
    label_cfg = load_yaml("label_map.yaml")
    dcfg = cfg["data"]
    seed = cfg["seed"]
    raw_dir, out_dir = ROOT / dcfg["raw_dir"], ROOT / dcfg["processed_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    n_val = dcfg["val_per_source"]

    splits = {}
    for source, (_, _, eval_split) in SOURCE_FIELDS.items():
        ds = load_from_disk(str(raw_dir / source))
        splits[source] = (english_only(source, ds["train"]), english_only(source, ds[eval_split]))
        print(f"{source}: English train={len(splits[source][0])} eval={len(splits[source][1])}")

    available = {s: len(tr) - n_val for s, (tr, _) in splits.items()}
    alloc = allocate(dcfg["train_caps"], available)
    print(f"train allocation: {alloc}  total={sum(alloc.values())}")

    train, val, test = [], [], []
    for source, (tr_split, te_split) in splits.items():
        mapping = label_cfg[source]
        stats = Counter()
        tr_split = tr_split.shuffle(seed=seed)
        val_rows = normalize(source, tr_split.select(range(n_val)), mapping, stats)
        tr_rows = normalize(source, tr_split.select(range(n_val, n_val + alloc[source])), mapping, stats)
        te_split = te_split.shuffle(seed=seed)
        te_rows = normalize(source, te_split.select(range(min(dcfg["test_per_source"], len(te_split)))),
                            mapping, stats)
        train += tr_rows
        val += val_rows
        test += te_rows

        label_counts = Counter(s["label"] for r in tr_rows for s in r["spans"])
        print(f"\n== {source}: train={len(tr_rows)} val={len(val_rows)} test={len(te_rows)}")
        print(f"   span labels: {dict(label_counts.most_common())}")
        print(f"   cleanup: {dict(stats)}")

    random.Random(seed).shuffle(train)
    for name, rows in [("train", train), ("val", val), ("test", test)]:
        write_jsonl(out_dir / f"{name}.jsonl", rows)
        print(f"wrote {name}: {len(rows)} rows")


if __name__ == "__main__":
    main()
