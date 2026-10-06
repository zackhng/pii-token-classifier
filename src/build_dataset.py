"""Normalize raw sources to {id, source, text, spans} jsonl with the target label set.

Pipeline per source: English filter -> (train only: date/cue augmentation) -> label map ->
overlap cleanup -> merge adjacent same-label spans -> seeded subsample -> train/val/test jsonl in
data/processed. Train also gets unused DOB-bearing docs (dob_boost) and form-style records
(snippets). Look-alike spans (amounts, ordinary dates, codes) of test/stress docs are written to
negatives.jsonl for eval_confusion.py.
"""
import json
import random
import re
from collections import Counter

from datasets import load_from_disk

from augment import augment_doc, has_birth_cue
from common import DATE_CUE, IGNORE, ROOT, load_yaml
from snippets import make_address_records, make_records

# Gap text allowed between two same-label spans for them to be merged into one.
# Addresses span commas/newlines ("12 Main St,\nSpringfield, IL"); other labels only spaces.
MERGE_GAP = {"ADDRESS": re.compile(r"[\s,]*"), "DEFAULT": re.compile(r"[ \t]*")}

# (text column, spans column, official eval split) per source
SOURCE_FIELDS = {
    "nemotron": ("text", "spans", "test"),
    "gretel": ("generated_text", "pii_spans", "test"),
    "ai4privacy": ("source_text", "privacy_mask", "validation"),
    "pfi": ("source_text", "privacy_mask", "validation"),
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


# ADDRESS = full residential / office addresses and their parts. A span made only of place-name
# parts (no street, building, postcode or coordinates) counts only when an address field cue
# precedes it on the same line ("City: Boston"); a bare place in running text ("expanding into
# France") becomes O and is recorded as a "place" look-alike.
PLACE_ONLY = {"city", "state", "county", "country", "CITY"}
ADDRESS_CUE = re.compile(r"(address|addr\.?|city|town|state|province|country|county|post ?code|zip|"
                         r"residen\w*|located at|domicile)\W{0,3}$", re.I)
# Gretel labels only the street part of "0567 Drake Road, Manchester, M12 4BE, UK"; the unlabelled
# continuation is masked so a complete prediction is not trained (or scored) as wrong.
STREET_TAIL = re.compile(r",[^\n.;|()]{1,80}")
ADDRESS_TAIL_SOURCES = {"gretel"}


def english_birth_cue(text: str, start: int, end: int) -> bool:
    return has_birth_cue(text, start)


def clean_spans(text: str, spans: list[dict], mapping: dict, stats: Counter,
                negs: list | None = None, neg_groups: dict | None = None,
                address_tail_ignore: bool = False, birth_cue=english_birth_cue) -> list[dict]:
    """Map raw spans to target labels. Look-alike spans whose raw label is in `neg_groups`
    (raw label -> group) are appended to `negs`; they stay O for training. `birth_cue(text, start, end)`
    decides DATE_CUE spans."""
    mapped = []
    for s in spans:
        start, end = int(s["start"]), int(s["end"])
        target = mapping.get(s["label"])
        if target == DATE_CUE:
            target = "DOB" if birth_cue(text, start, end) else None
            stats["date_cue_dob" if target else "date_cue_o"] += 1
        if target is None:
            group = (neg_groups or {}).get(s["label"])
            if negs is not None and group and 0 <= start < end <= len(text):
                negs.append({"start": start, "end": end, "label": group})
            continue
        if not (0 <= start < end <= len(text)) or not text[start:end].strip():
            stats["invalid"] += 1
            continue
        if target == "DOB" and not any(c.isdigit() for c in text[start:end]):
            target = IGNORE            # placeholders like "[Redacted]", "MM/DD/YYYY"
            stats["dob_placeholder"] += 1
        # trim whitespace so token alignment starts/ends on real characters
        while text[start].isspace():
            start += 1
        while text[end - 1].isspace():
            end -= 1
        mapped.append({"start": start, "end": end, "label": target, "raw": {s["label"]}})

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
                prev["raw"] = prev["raw"] | s["raw"]
                stats["merged"] += 1
                continue
        merged.append(dict(s))

    out = []
    for s in merged:
        raw = s.pop("raw")
        if s["label"] == "ADDRESS" and raw <= PLACE_ONLY:
            line = text[max(0, s["start"] - 40):s["start"]].rsplit("\n", 1)[-1]
            if not ADDRESS_CUE.search(line):
                stats["address_bare_place_to_O"] += 1
                if negs is not None:
                    negs.append({"start": s["start"], "end": s["end"], "label": "place"})
                continue
        out.append(s)
        if address_tail_ignore and s["label"] == "ADDRESS" and raw == {"street_address"}:
            m = STREET_TAIL.match(text, s["end"])
            nxt = min((o["start"] for o in merged if o["start"] >= s["end"]), default=len(text))
            if m and m.end() <= nxt:
                out.append({"start": s["end"], "end": m.end(), "label": IGNORE})
                stats["address_tail_ignored"] += 1
    return out


def normalize(source: str, split, mapping: dict, stats: Counter, aug: dict | None = None,
              rng: random.Random | None = None, neg_groups: dict | None = None,
              keep_negs: bool = False, lang_ok=is_english, birth_cue=english_birth_cue) -> list[dict]:
    text_col, span_col, _ = SOURCE_FIELDS[source]
    out = []
    for i, row in enumerate(split):
        if not lang_ok(source, row):
            continue
        text = row[text_col]
        if not text or not text.strip():
            continue
        raw = [{"start": int(s["start"]), "end": int(s["end"]), "label": s["label"]}
               for s in parse_spans(row[span_col])]
        if aug:
            new_text, raw = augment_doc(text, raw, rng, aug["p_date"], aug["p_cue"])
            stats["augmented"] += new_text != text
            text = new_text
        negs = [] if keep_negs else None
        spans = clean_spans(text, raw, mapping, stats, negs, neg_groups,
                            address_tail_ignore=source in ADDRESS_TAIL_SOURCES, birth_cue=birth_cue)
        rid = row.get("uid", row.get("index", i))
        rec = {"id": f"{source}-{rid}", "source": source, "text": text, "spans": spans}
        if keep_negs:
            rec["negs"] = negs
        out.append(rec)
    return out


def has_raw_label(source: str, row: dict, label: str) -> bool:
    return any(s["label"] == label for s in parse_spans(row[SOURCE_FIELDS[source][1]]))


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


def split_negs(rows: list[dict]) -> list[dict]:
    """Pop the `negs` field (not part of the training schema) into separate records."""
    return [{"id": r["id"], "negs": r.pop("negs")} for r in rows if "negs" in r]


def main():
    cfg = load_yaml("train.yaml")
    label_cfg = load_yaml("label_map.yaml")
    dcfg = cfg["data"]
    seed = cfg["seed"]
    raw_dir, out_dir = ROOT / dcfg["raw_dir"], ROOT / dcfg["processed_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    n_val = dcfg["val_per_source"]
    aug = dcfg.get("augment")
    aug_rng = random.Random(seed)
    neg_groups = {lab: g for g, labs in label_cfg["negatives"].items() for lab in labs}

    splits = {}
    for source, (_, _, eval_split) in SOURCE_FIELDS.items():
        if source not in dcfg["train_caps"]:
            continue
        if not (raw_dir / source).exists():
            print(f"[skip] {source}: not downloaded (python download.py --only {source})")
            continue
        ds = load_from_disk(str(raw_dir / source))
        splits[source] = (english_only(source, ds["train"]), english_only(source, ds[eval_split]))
        print(f"{source}: English train={len(splits[source][0])} eval={len(splits[source][1])}")

    caps = {s: dcfg["train_caps"][s] for s in splits}
    available = {s: len(tr) - n_val for s, (tr, _) in splits.items()}
    alloc = allocate(caps, available)
    print(f"train allocation: {alloc}  total={sum(alloc.values())}")

    train, val, test = [], [], []
    for source, (tr_split, te_split) in splits.items():
        mapping = label_cfg[source]
        stats = Counter()
        tr_split = tr_split.shuffle(seed=seed)
        val_rows = normalize(source, tr_split.select(range(n_val)), mapping, stats)
        used = n_val + alloc[source]
        tr_rows = normalize(source, tr_split.select(range(n_val, used)), mapping, stats,
                            aug, aug_rng, neg_groups)
        if source in dcfg.get("dob_boost", []):
            rest = tr_split.select(range(used, len(tr_split)))
            rest = rest.filter(lambda r: has_raw_label(source, r, "date_of_birth"))
            tr_rows += normalize(source, rest, mapping, stats, aug, aug_rng, neg_groups)
            print(f"{source}: dob_boost +{len(rest)} docs")
        te_split = te_split.shuffle(seed=seed)
        te_rows = normalize(source, te_split.select(range(min(dcfg["test_per_source"], len(te_split)))),
                            mapping, stats, neg_groups=neg_groups, keep_negs=True)
        train += tr_rows
        val += val_rows
        test += te_rows

        label_counts = Counter(s["label"] for r in tr_rows for s in r["spans"])
        print(f"\n== {source}: train={len(tr_rows)} val={len(val_rows)} test={len(te_rows)}")
        print(f"   span labels: {dict(label_counts.most_common())}")
        print(f"   cleanup: {dict(stats)}")

    # form-style records: train ones from train spans, stress ones (other templates) from test
    sn = dcfg.get("snippets", {})
    snips = make_records(train, sn.get("train", 0), seed, "train", "snip")
    for r in snips:
        r.pop("negs")
    train += snips
    stress = make_records(test, sn.get("stress", 0), seed + 1, "stress", "stress")
    print(f"\nsnippets: train +{len(snips)}, stress {len(stress)}")

    # real public addresses in context (src/address_sources.py must have been run)
    addr_path = ROOT / "data/addresses/addresses.jsonl"
    acfg = dcfg.get("addresses")
    if acfg and addr_path.exists():
        addrs = [json.loads(l) for l in open(addr_path, encoding="utf-8")]
        rec = acfg["records"]
        a_train = make_address_records(train, [a for a in addrs if a["split"] == "train"],
                                       rec["train"], seed + 2, "train", "addr")
        for r in a_train:
            r.pop("negs")
        train += a_train
        stress += make_address_records(test, [a for a in addrs if a["split"] == "stress"],
                                       rec["stress"], seed + 3, "stress", "addr-stress")
        print(f"address records: train +{len(a_train)}, stress +{rec['stress']} "
              f"(from {len(addrs)} real addresses)")
    elif acfg:
        print("[warn] data/addresses/addresses.jsonl missing - run address_sources.py first")

    # the sources' official splits share some identical texts: keep them out of train
    held_out = {r["text"] for r in val + test}
    n_before = len(train)
    train = [r for r in train if r["text"] not in held_out]
    print(f"dropped {n_before - len(train)} train docs identical to a val/test doc")

    random.Random(seed).shuffle(train)
    negatives = split_negs(test) + split_negs(stress)
    for name, rows in [("train", train), ("val", val), ("test", test), ("stress", stress),
                       ("negatives", negatives)]:
        write_jsonl(out_dir / f"{name}.jsonl", rows)
        print(f"wrote {name}: {len(rows)} rows")


if __name__ == "__main__":
    main()
