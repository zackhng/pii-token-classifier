"""Multilingual training / validation data from public sources (configs/ml_sources.yaml).

  data/processed/ml_train.jsonl    {id, source, lang, kind, partial, text, spans}
  data/processed/ml_val.jsonl      val_per_lang docs per language, from the sources' dev splits
  data/processed/ml_kiii_test.jsonl  Kiii held-out Korean financial docs (eval format)
  results/ml_train_coverage.{txt,json}  spans per language x entity, and by source kind

Labels are the 8 targets; IGNORE spans are loss-masked; "O" spans are known look-alikes, kept so
partially annotated sources (`partial`) still learn them as not-PII. Documents whose text is in
an evaluation set (ml_real, ml_synth, English test / stress) are dropped.

    python build_ml_train.py [--allow_gaps]
"""
import argparse
import json
import random
import re
from collections import Counter, defaultdict

from datasets import load_from_disk

from build_dataset import clean_spans, normalize, write_jsonl
from build_ml_eval import ml_birth_cue, tag_scripts
from common import IGNORE, ROOT, load_yaml
from ml_sources import READERS, ai4privacy_500k_hi
from ml_lookalikes import records as lookalike_records
from ml_train_docs import generate

LANGS = ["en", "zh-Hans", "zh-Hant", "ja", "ko", "hi", "ar", "th", "vi", "ms", "id", "tl"]
ENTITIES = load_yaml("label_map.yaml")["entities"]
PROC = ROOT / "data/processed"
# CLUENER "address" includes bare places ("北京"); only street-level ones are ADDRESS
ZH_STREET = re.compile(r"[0-9０-９]|[路街道巷弄号號室楼樓栋棟层層幢座村镇鎮乡鄉]|大厦|大廈|小区|小區|花园|花園")


def eval_texts() -> set[str]:
    out = set()
    for name in ["ml_real", "ml_synth", "test", "stress"]:
        p = PROC / f"{name}.jsonl"
        if p.exists():
            out.update(json.loads(l)["text"] for l in open(p, encoding="utf-8"))
    return out


def ai4p_rows(lang: str, split: str, n: int, seed: int, stats: Counter) -> list[dict]:
    """AI4Privacy rows in target labels (same label map / cleanup as the English training data)."""
    label_cfg = load_yaml("label_map.yaml")
    neg_groups = {lab: g for g, labs in label_cfg["negatives"].items() for lab in labs}
    if lang == "hi":
        rows = ai4privacy_500k_hi(split)
        random.Random(seed).shuffle(rows)
        rows = rows[:2 * n]
    else:
        ds = load_from_disk(str(ROOT / "data/raw/ai4privacy"))[split]
        ds = ds.filter(lambda b: [l == lang for l in b["language"]], batched=True).shuffle(seed=seed)
        rows = ds.select(range(min(len(ds), 2 * n)))
    out = normalize("ai4privacy", rows, label_cfg["ai4privacy"], stats, neg_groups=neg_groups,
                    lang_ok=lambda _s, r: r["language"] == lang, birth_cue=ml_birth_cue)
    return [r for r in out if any(s["label"] != IGNORE for s in r["spans"])][:n]


def mapped_rows(name: str, cfg: dict, split: str, stats: Counter) -> list[dict]:
    """Rows of a READERS source mapped to target labels."""
    mapping = dict(cfg.get("map", {}))
    mapping["address_place"] = IGNORE
    neg_groups = cfg.get("neg", {})
    partial = cfg.get("partial", "")
    out = []
    for r in READERS[cfg.get("reader", name)](split):
        raw = r["spans"]
        if name == "cluener":
            raw = [dict(s, label="address_place") if s["label"] == "address" and not
                   ZH_STREET.search(r["text"][s["start"]:s["end"]]) else s for s in raw]
        negs = []
        spans = clean_spans(r["text"], raw, mapping, stats, negs, neg_groups, birth_cue=ml_birth_cue)
        if partial:      # look-alikes stay explicit O when unlabelled tokens are masked
            taken = [(s["start"], s["end"]) for s in spans]
            spans += [{"start": g["start"], "end": g["end"], "label": "O"} for g in negs
                      if not any(a < g["end"] and g["start"] < b for a, b in taken)]
            spans.sort(key=lambda s: s["start"])
        out.append({**r, "spans": spans})
    return out


def english_rows(split: str, n: int, seed: int) -> list[dict]:
    path = PROC / ("train.jsonl" if split == "train" else "val.jsonl")
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    random.Random(seed).shuffle(rows)
    return [{**r, "lang": "en"} for r in rows[:n]]


def split_zh_hant(rows: list[dict], share: float, seed: int) -> list[dict]:
    """A seeded `share` of zh docs is converted to Traditional characters (OpenCC s2t); kept only
    if the length is unchanged so span offsets stay valid. The rest stays zh-Hans."""
    import opencc
    cc = opencc.OpenCC("s2t")
    rng = random.Random(seed)
    out = []
    for r in rows:
        if rng.random() < share:
            t = cc.convert(r["text"])
            if len(t) == len(r["text"]):
                out.append({**r, "id": r["id"] + "-hant", "lang": "zh-Hant", "text": t})
            continue
        out.append({**r, "lang": "zh-Hans"})
    return out


def load_source(name, cfg, split, seed, stats):
    cap = cfg.get("cap")
    reader = cfg.get("reader", name)
    if reader == "ai4privacy":
        rows = ai4p_rows(cfg["lang"], split, cap if split == "train" else 2000, seed, stats)
    elif reader == "english":
        rows = english_rows(split, cap if split == "train" else 2000, seed)
    elif reader == "generated":   # validation docs come from a different seed than training
        rows = generate(cfg["lang"], cap if split == "train" else 300, seed if split == "train" else seed + 1)
    else:
        if split == "validation" and name in ("anercorp", "kiii"):
            return []            # no dev split upstream (Kiii's held-out part is a test set)
        rows = mapped_rows(name, cfg, split, stats)
        random.Random(seed).shuffle(rows)
        if cap and split == "train":
            rows = rows[:cap]
    for r in rows:
        r.update(source=name, kind=cfg["kind"], partial=cfg.get("partial", ""),
                 lang=r.get("lang", cfg["lang"]) if cfg["lang"] != "zh" else "zh")
    return rows


def coverage(rows: list[dict]):
    cells = defaultdict(Counter)            # lang -> entity -> spans
    kinds = defaultdict(Counter)            # lang -> kind -> spans
    docs = Counter(r["lang"] for r in rows)
    for r in rows:
        for s in r["spans"]:
            if s["label"] in ENTITIES:
                cells[r["lang"]][s["label"]] += 1
                kinds[r["lang"]][r["kind"]] += 1
    return docs, cells, kinds


def report(docs, cells, kinds, min_spans) -> tuple[str, list]:
    gaps = []
    lines = [f"{'lang':8s} {'docs':>7s} " + " ".join(f"{e[:8]:>8s}" for e in ENTITIES) +
             "   financial    pii    ner  generated"]
    for lang in LANGS:
        row = []
        for e in ENTITIES:
            n = cells[lang][e]
            if n < min_spans:
                gaps.append((lang, e, n))
            row.append(f"{n:>7d}{'*' if n < min_spans else ' '}")
        tot = sum(kinds[lang].values()) or 1
        share = " ".join(f"{kinds[lang][k] / tot:>9.0%}" for k in ["financial", "pii", "ner", "generated"])
        lines.append(f"{lang:8s} {docs[lang]:7d} " + " ".join(row) + "  " + share)
    lines.append(f"\n* below min_spans={min_spans}: {len(gaps)} cells")
    for lang, e, n in gaps:
        lines.append(f"  {lang:8s} {e:9s} {n}")
    return "\n".join(lines), gaps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow_gaps", action="store_true", help="write outputs even if a cell is below min_spans")
    args = ap.parse_args()
    scfg = load_yaml("ml_sources.yaml")
    seed = load_yaml("train.yaml")["seed"]
    held_out = eval_texts()
    stats = Counter()

    train, val = [], defaultdict(list)
    for name, cfg in scfg["sources"].items():
        tr = load_source(name, cfg, "train", seed, stats)
        va = load_source(name, cfg, "validation", seed, stats)
        if cfg["lang"] == "zh":
            tr, va = split_zh_hant(tr, scfg["zh_hant_share"], seed), split_zh_hant(va, scfg["zh_hant_share"], seed)
        n0 = len(tr)
        tr = [r for r in tr if r["text"] not in held_out]
        va = [r for r in va if r["text"] not in held_out]
        train += tr
        for r in va:
            val[r["lang"]].append(r)
        print(f"{name:15s} train {len(tr):6d} (dropped {n0 - len(tr)} eval-set texts)  val {len(va):5d}")

    # validation: val_per_lang docs per language, spread evenly over that language's sources
    val_rows, rng = [], random.Random(seed)
    for lang, rows in val.items():
        by_src = defaultdict(list)
        for r in rows:
            by_src[r["source"]].append(r)
        for v in by_src.values():
            rng.shuffle(v)
        picked = []
        while len(picked) < scfg["val_per_lang"] and any(by_src.values()):
            for v in by_src.values():
                if v and len(picked) < scfg["val_per_lang"]:
                    picked.append(v.pop())
        val_rows += picked
    # look-alike form records per language (ml_lookalikes.py): real PII values from that language's
    # training docs next to codes / amounts that must stay O
    n_look = scfg.get("lookalikes_per_lang", 0)
    if n_look:
        pools = defaultdict(lambda: defaultdict(set))
        for r in train:     # generated docs count too: th / ar / hi have few real values for some types
            if r["source"].startswith("lookalike_"):
                continue
            for s in r["spans"]:
                v = r["text"][s["start"]:s["end"]]
                if s["label"] in ("PERSON", "ACCOUNT", "PHONE", "EMAIL", "TIN") and "\n" not in v and len(v) <= 60:
                    pools[r["lang"]][s["label"]].add(v)
        for lang in LANGS:
            if lang == "en":            # English has v2's snippet records already
                continue
            pool = {k: sorted(v) for k, v in pools[lang].items()}
            recs = lookalike_records(lang, pool, n_look, seed)
            train += recs
            print(f"lookalike_{lang:8s} train {len(recs):6d}  (pool sizes {dict((k, len(v)) for k, v in pool.items())})")

    val_texts = {r["text"] for r in val_rows}
    train = [r for r in train if r["text"] not in val_texts]
    rng.shuffle(train)

    docs, cells, kinds = coverage(train)
    txt, gaps = report(docs, cells, kinds, scfg["min_spans"])
    print("\n" + txt)
    print(f"\ncleanup: {dict(stats)}")
    (ROOT / "results").mkdir(exist_ok=True)
    with open(ROOT / "results/ml_train_coverage.txt", "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    with open(ROOT / "results/ml_train_coverage.json", "w", encoding="utf-8") as f:
        json.dump({"docs": docs, "spans": cells, "by_kind": kinds,
                   "val_docs": Counter(r["lang"] for r in val_rows), "gaps": gaps}, f, indent=1, ensure_ascii=False)
    if gaps and not args.allow_gaps:
        raise SystemExit(f"{len(gaps)} language x entity cells below min_spans - fill them "
                         f"(configs/ml_sources.yaml) or pass --allow_gaps")

    write_jsonl(PROC / "ml_train.jsonl", train)
    write_jsonl(PROC / "ml_val.jsonl", val_rows)
    kiii_test = []
    for r in mapped_rows("kiii", scfg["sources"]["kiii"], "test", Counter()):
        negs = [{"start": s["start"], "end": s["end"], "label": "code"} for s in r["spans"] if s["label"] == "O"]
        kiii_test.append(tag_scripts({**r, "source": "kiii_test", "spans": [s for s in r["spans"] if s["label"] != "O"],
                                      "negs": negs}))
    write_jsonl(PROC / "ml_kiii_test.jsonl", kiii_test)
    print(f"wrote ml_train {len(train)}, ml_val {len(val_rows)}, ml_kiii_test {len(kiii_test)}")


if __name__ == "__main__":
    main()
