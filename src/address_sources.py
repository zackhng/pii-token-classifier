"""Real addresses from public datasets, per country, for the address records in snippets.py.

Sources (configs/train.yaml -> data.addresses):
- libpostal parser training data (ellenhp/libpostal): OpenStreetMap addresses (ODbL) and UK
  OpenAddresses, stored as tokens + libpostal component tags; rebuilt here into component lists
- gagan1985/indian-addresses-raw (Apache-2.0): Indian registered-office (MCA) and bank-branch
  addresses as written

Output: data/addresses/addresses.jsonl, one {"country", "source", "lines": [...]} per address,
deduplicated, with a stable hash split ("train" 90% / "stress" 10%).
"""
import hashlib
import json
import random
import re
from collections import Counter, defaultdict

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

from common import ROOT, load_yaml

# libpostal label order (1-based tag ids in the HF export; checked against sample rows:
# 4 house_number, 5 road, 11 postcode, 14 city, 19 country)
TAGS = ["house", "category", "near", "house_number", "road", "unit", "level", "staircase",
        "entrance", "po_box", "postcode", "suburb", "city_district", "city", "island",
        "state_district", "state", "country_region", "country", "world_region"]
POSTCODE_FIRST = {"my", "vn", "de", "fr", "kr", "tw", "jp"}   # "57700 Kuala Lumpur"
COUNTRY_NAMES = {"sg": "Singapore", "ae": "United Arab Emirates", "gb": "United Kingdom",
                 "in": "India", "hk": "Hong Kong", "my": "Malaysia", "id": "Indonesia",
                 "th": "Thailand", "ph": "Philippines", "vn": "Vietnam", "cn": "China",
                 "tw": "Taiwan", "jp": "Japan", "kr": "South Korea", "sa": "Saudi Arabia",
                 "qa": "Qatar", "bh": "Bahrain", "kw": "Kuwait", "om": "Oman", "lk": "Sri Lanka",
                 "bd": "Bangladesh", "pk": "Pakistan", "mm": "Myanmar", "kh": "Cambodia",
                 "bn": "Brunei"}


def is_latin(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    return bool(letters) and sum(c.isascii() for c in letters) / len(letters) > 0.95


def components_to_lines(tokens, tags, country: str) -> list[str] | None:
    """libpostal tokens/tags -> address lines (street, area, city/postcode, state, country)."""
    comp = defaultdict(list)
    for tok, tag in zip(tokens, tags):
        if 1 <= tag <= len(TAGS):
            comp[TAGS[tag - 1]].append(tok)
    c = {k: " ".join(v) for k, v in comp.items()}
    if "category" in c or "near" in c:          # place queries ("cafes near ..."), not addresses
        return None
    # street-level addresses only: a road with a number / postcode / PO box / unit, or a
    # building on a road in a city - not "venue name, city"
    specific = any(k in c for k in ("house_number", "postcode", "po_box", "unit", "level"))
    if not (("road" in c and specific) or ("house" in c and "road" in c and "city" in c)):
        return None
    lines = []
    if "house" in c:
        lines.append(c["house"])
    street = " ".join(x for x in (c.get("unit"), c.get("level"), c.get("house_number"), c.get("road")) if x)
    if street:
        lines.append(street)
    area = ", ".join(x for x in (c.get("suburb"), c.get("city_district")) if x)
    if area:
        lines.append(area)
    city, pc = c.get("city"), c.get("postcode")
    if city or pc:
        lines.append(" ".join(x for x in ((pc, city) if country in POSTCODE_FIRST else (city, pc)) if x))
    for k in ("state_district", "state", "country"):
        if k in c:
            lines.append(c[k])
    return lines


def libpostal_rows(subset: str, files: list[str], countries: set[str], per_country: int, rng):
    want = defaultdict(list)
    for f in files:
        df = pd.read_parquet(hf_hub_download("ellenhp/libpostal", f, repo_type="dataset"))
        df = df[df["country"].isin(countries)]
        for country, tokens, tags in zip(df["country"], df["tokens"], df["tags"]):
            if len(want[country]) >= per_country * 3:
                continue
            if not is_latin(" ".join(tokens)):
                continue
            lines = components_to_lines(list(tokens), list(tags), country)
            if lines:
                want[country].append(lines)
    for country, rows in want.items():
        rng.shuffle(rows)
        for lines in rows[:per_country]:
            yield country, f"libpostal_{subset}", lines


_JUNK = re.compile(r"\[REDACTED\]|(?<=,)\s*-1\s*(?=,)|\bUnclassified\b", re.I)


def india_rows(per_source: dict, rng):
    df = pd.read_parquet(hf_hub_download("gagan1985/indian-addresses-raw", "data/raw_addresses.parquet",
                                         repo_type="dataset"), columns=["source_type", "raw_address"])
    for src, n in per_source.items():
        sub = df[df["source_type"] == src]["raw_address"].dropna()
        sub = sub.sample(min(len(sub), n * 2), random_state=rng.randint(0, 10**6))
        out = 0
        for raw in sub:
            text = re.sub(r"\s+", " ", _JUNK.sub("", str(raw)))
            parts, seen = [], set()
            for p in (p.strip(" ,") for p in text.split(",")):
                if p and p.lower() not in seen:       # bank rows often repeat district / state / PIN
                    seen.add(p.lower())
                    parts.append(p)
            if not 3 <= len(parts) <= 8 or not is_latin(text) or not re.search(r"\d{6}", text):
                continue
            yield "in", f"india_{src}", parts
            out += 1
            if out >= n:
                break


def main():
    cfg = load_yaml("train.yaml")["data"]["addresses"]
    rng = random.Random(cfg["seed"])
    out_dir = ROOT / "data/addresses"
    out_dir.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    files = sorted(s.rfilename for s in api.dataset_info("ellenhp/libpostal").siblings
                   if s.rfilename.startswith("openstreetmap_addresses/train-"))[:cfg["libpostal_files"]]
    uk_files = sorted(s.rfilename for s in api.dataset_info("ellenhp/libpostal").siblings
                      if s.rfilename.startswith("uk_openaddresses/train-"))

    rows, seen = [], set()
    # UK OpenAddresses first: cleaner than OSM for the UK (house numbers + full postcodes)
    streams = [libpostal_rows("uk_openaddresses", uk_files, {"gb"}, cfg["per_country"]["gb"] // 2, rng),
               libpostal_rows("osm", files, set(cfg["per_country"]), max(cfg["per_country"].values()), rng),
               india_rows(cfg["india_raw"], rng)]
    count = Counter()
    for stream in streams:
        for country, source, lines in stream:
            cap = cfg["per_country"].get(country, 0) + (sum(cfg["india_raw"].values()) if source.startswith("india") else 0)
            if count[country] >= cap:
                continue
            key = " ".join(lines).lower()
            if key in seen:
                continue
            seen.add(key)
            if rng.random() < cfg.get("add_country_prob", 0.3) and country in COUNTRY_NAMES \
                    and COUNTRY_NAMES[country].lower() not in key:
                lines = lines + [COUNTRY_NAMES[country]]
            split = "stress" if int(hashlib.md5(key.encode()).hexdigest(), 16) % 10 == 0 else "train"
            rows.append({"country": country, "source": source, "split": split, "lines": lines})
            count[country] += 1
    with open(out_dir / "addresses.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(rows)} addresses:", dict(count.most_common()))
    print("by source:", dict(Counter(r["source"] for r in rows)))
    for c in ["sg", "ae", "gb", "in", "hk", "my", "id", "cn"]:
        ex = [r for r in rows if r["country"] == c][:2]
        for r in ex:
            print(f"  {c}: {' | '.join(r['lines'])}")


if __name__ == "__main__":
    main()
