"""Build the v3 pretraining corpus (configs/pretrain.yaml) from public datasets.

Per domain: read sources -> clean -> drop short / duplicate docs -> sample up to the token
budget -> hold out 1% of docs. Train docs (with per-domain `repeat`) are shuffled together and
packed into [CLS] ... [SEP] sequences of `seq_len` tokens (docs joined by [SEP]), so every slice
of the training stream has the same domain mix.

Output (data/pretrain/): train_ids.npy [N, seq_len] int32, train_domain_tokens.npy [N, D] int32
(tokens of each domain per sequence), heldout_ids.npy, heldout_domain.npy, meta.json (sizes per
domain and source).
"""
import ast
import hashlib
import json
import lzma
import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_from_disk
from huggingface_hub import HfApi, hf_hub_download
from transformers import AutoTokenizer

from build_dataset import SOURCE_FIELDS, english_only
from common import ROOT, load_yaml
from tokenize_bio import visible_breaks


def dl(repo: str, path: str) -> str:
    return hf_hub_download(repo, path, repo_type="dataset")


def jsonl(path: str, opener=open):
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def split_long(text: str, max_chars: int = 20000) -> list[str]:
    """Split very long documents (whole CFR titles, long filings) at paragraph breaks."""
    if len(text) <= max_chars:
        return [text]
    parts, cur = [], []
    size = 0
    for para in text.split("\n\n"):
        if size + len(para) > max_chars and cur:
            parts.append("\n\n".join(cur))
            cur, size = [], 0
        cur.append(para[:max_chars])
        size += len(para) + 2
    if cur:
        parts.append("\n\n".join(cur))
    return parts


def norm_ws(text: str) -> str:
    text = re.sub(r"[ \t ]+", " ", text.replace("\r", "\n"))
    return re.sub(r"\n\s*\n\s*\n+", "\n\n", text).strip()


# ---------------------------------------------------------------- A. task text
def task_docs(cfg, rng):
    """Raw text of the fine-tuning sources' *train* splits (val/test texts are removed in
    collect()). Sources are interleaved so the budget is shared between them."""
    dcfg = load_yaml("train.yaml")["data"]
    iters = []
    for source, (text_col, _, _) in SOURCE_FIELDS.items():
        raw = ROOT / dcfg["raw_dir"] / source
        if raw.exists():
            split = english_only(source, load_from_disk(str(raw))["train"]).shuffle(seed=42)
            iters.append((source, iter(split[text_col])))
    while iters:
        for item in list(iters):
            text = next(item[1], None)
            if text is None:
                iters.remove(item)
            elif text:
                yield item[0], text


# ---------------------------------------------------------------- B1. regulation (Asia/UK/UAE/EU)
UK_FIN = re.compile(r"financ|money laundering|bank|payment|investment|insurance|pension|securit|"
                    r"market|fraud|proceeds of crime|credit|building societ|sanction|terroris|"
                    r"companies|consumer|trust|collective|capital requirement|tax", re.I)
CENTRAL_BANKS = ["monetary_authority_of_singapore", "bank_negara_malaysia", "reserve_bank_of_india",
                 "bank_of_thailand", "central_bank_of_the_philippines", "bank_of_england",
                 "bank_of_japan", "bank_of_korea", "peoples_bank_of_china",
                 "central_bank_of_china_taiwan", "reserve_bank_of_australia"]


def regulation_asia_docs(cfg, rng):
    src = cfg["sources"]
    # SEBI circular chunks: rebuild one document per circular from its chunks
    by_circ = defaultdict(list)
    for r in jsonl(dl(src["sebi_circulars"], "chunks/chunks.jsonl")):
        key = r.get("circular_number") or r.get("source_url") or r.get("doc_id")
        by_circ[key].append((r.get("chunk_index", r.get("chunk_id", 0)), r.get("text", "")))
    for chunks in by_circ.values():
        yield "sebi_circulars", "\n".join(t for _, t in sorted(chunks, key=lambda c: str(c[0])))
    # RBI notifications (Q&A pairs drawn from RBI notifications)
    df = pd.read_parquet(dl(src["rbi_notifications"], "data/train-00000-of-00001.parquet"))
    for q, a in zip(df["input"], df["output"]):
        yield "rbi_notifications", f"{str(q).strip()}\n{str(a).strip()}"
    # ADGM rulebook passages (unique passages of the ObliQA questions)
    seen = {}
    for split in ("train", "dev", "test"):
        for q in json.load(open(dl(src["adgm_obliqa"], f"ObliQA/ObliQA_{split}.json"), encoding="utf-8")):
            ps = q["Passages"]
            for p in (ast.literal_eval(ps) if isinstance(ps, str) else ps):
                seen[(p["DocumentID"], p["PassageID"])] = p["Passage"]
    for (doc, pid), text in sorted(seen.items(), key=lambda kv: (kv[0][0], str(kv[0][1]))):
        yield "adgm_obliqa", f"{pid} {text}"
    # UK financial regulation passages
    seen = set()
    for split in ("train", "dev", "test"):
        for r in jsonl(dl(src["uk_fin_regulation"], f"data/{split}.jsonl")):
            for t in (r["source_text"], r["target_text"]):
                if t not in seen:
                    seen.add(t)
                    yield "uk_fin_regulation", t
    # UK legislation: financial Acts / SIs only (title = first line)
    api = HfApi()
    for f in api.dataset_info(src["uk_legislation"]).siblings:
        if f.rfilename.endswith(".parquet"):
            for text in pd.read_parquet(dl(src["uk_legislation"], f.rfilename))["text"]:
                if UK_FIN.search(text.split("\n", 1)[0]):
                    yield "uk_legislation", text
    # EU financial regulation (MiFID II, MiCA, AIFMD, UCITS, DORA, ...), English, one doc per article
    arts = defaultdict(list)
    for r in jsonl(dl(src["eu_fin_regulation"], "data/corpus.jsonl")):
        if r["language"] == "en":
            arts[(r["act_name"], r["article"])].append(r["text"])
    for (act, art), paras in arts.items():
        yield "eu_fin_regulation", f"{act} Article {art}\n" + "\n".join(paras)
    # central-bank statements (sentences grouped by bank and year)
    for bank in CENTRAL_BANKS:
        repo = f"{src['central_banks']}/{bank}"
        try:
            files = [f.rfilename for f in api.dataset_info(repo).siblings
                     if f.rfilename.endswith(".parquet")]
        except Exception:   # noqa: BLE001 - optional source
            continue
        seed_dir = sorted({f.split("/")[0] for f in files})[0]
        df = pd.concat([pd.read_parquet(dl(repo, f)) for f in files if f.startswith(seed_dir + "/")])
        for _, grp in df.groupby(df.get("year", pd.Series(0, index=df.index))):
            yield "central_banks", " ".join(grp["sentences"].astype(str))


# ---------------------------------------------------------------- B2. US regulation
US_FIN = re.compile(r"Securities and Exchange Commission|Commodity Futures Trading|Financial Crimes "
                    r"Enforcement|FinCEN|Comptroller of the Currency|Federal Reserve|Consumer Financial "
                    r"Protection|Federal Deposit Insurance|FINRA|investment adviser|broker-dealer|"
                    r"money laundering|Bank Secrecy Act|Commodity and Securities Exchanges|Banks and Banking|"
                    r"Money and Finance", re.I)


def regulation_us_docs(cfg, rng):
    for sub in cfg["sources"]:
        path = dl("pile-of-law/pile-of-law", f"data/train.{sub}.jsonl.xz")
        take_all = sub in ("sec", "cfpb_cc")
        for r in jsonl(path, lzma.open):
            text = r.get("text", "")
            if take_all or US_FIN.search(norm_ws(text[:5000])[:1500]):   # header, not a passing mention
                yield sub, text


# ---------------------------------------------------------------- C. investment / risk
def investment_docs(cfg, rng):
    years = list(cfg["years"])
    rng.shuffle(years)
    for y in years:
        for r in jsonl(dl("eloukas/edgar-corpus", f"{y}/train.jsonl")):
            for sec in cfg["sections"]:
                text = (r.get(sec) or "")[:cfg["max_chars_per_section"]]
                if text:
                    yield f"edgar_{sec}", text


# ---------------------------------------------------------------- D. communication
_QUOTE = re.compile(r"(-{3,}\s*Original Message\s*-{3,}|-{3,}\s*Forwarded by|^>|^From:.*\nSent:)",
                    re.M | re.I)
_HEADER = re.compile(r"^(Message-ID|Date|From|To|Cc|Bcc|Subject|Mime-Version|Content-Type|"
                     r"Content-Transfer-Encoding|X-[\w-]+):.*$", re.M | re.I)


def clean_email(body: str) -> str:
    m = _QUOTE.search(body)
    if m:
        body = body[:m.start()]
    return _HEADER.sub("", body)


def communication_docs(cfg, rng):
    src = cfg["sources"]
    api = HfApi()
    files = [f.rfilename for f in api.dataset_info(src["enron"]).siblings
             if f.rfilename.endswith(".parquet")]
    emails = []
    for f in files:
        df = pd.read_parquet(dl(src["enron"], f))
        col = next(c for c in ("body", "text", "content", "message") if c in df.columns)
        subj = df["subject"] if "subject" in df.columns else None
        for i, body in enumerate(df[col].astype(str)):
            s = f"Subject: {subj.iloc[i]}\n" if subj is not None and subj.iloc[i] else ""
            emails.append(s + clean_email(body))
    rng.shuffle(emails)
    fiqa = pd.read_parquet(dl(src["fiqa"], "corpus/corpus-00000-of-00001.parquet"))
    fiqa_texts = [f"{t}\n{x}".strip() for t, x in zip(fiqa.get("title", [""] * len(fiqa)), fiqa["text"])]
    # interleave so the budget is not filled by one source first
    for i in range(max(len(emails), len(fiqa_texts))):
        if i < len(fiqa_texts):
            yield "fiqa", fiqa_texts[i]
        if i < len(emails):
            yield "enron", emails[i]


READERS = {"task": task_docs, "regulation_asia_uk": regulation_asia_docs,
           "regulation_us": regulation_us_docs, "investment_risk": investment_docs,
           "communication": communication_docs}


# ---------------------------------------------------------------- assembly
def collect(name, dcfg, tok, ccfg, rng, held_texts):
    """Docs of one domain up to its budget: list of (source, token array)."""
    budget, docs, n_tok, seen = dcfg["budget"] // dcfg.get("repeat", 1), [], 0, set()
    caps = dcfg.get("max_source_tokens", {})
    stats = defaultdict(lambda: {"docs": 0, "tokens": 0, "chars": 0})
    batch = []

    def flush():
        nonlocal n_tok
        # same line-break / tab markers as fine-tuning (tokenize_bio.visible_breaks)
        enc = tok([visible_breaks(t)[0] for _, t in batch], add_special_tokens=False)["input_ids"]
        for (src, text), ids in zip(batch, enc):
            if n_tok >= budget:
                break
            if src in caps and stats[src]["tokens"] >= caps[src]:
                continue
            docs.append((src, np.asarray(ids, dtype=np.int32)))
            n_tok += len(ids)
            s = stats[src]
            s["docs"] += 1; s["tokens"] += len(ids); s["chars"] += len(text)
        batch.clear()

    for src, full in READERS[name](dcfg, rng):
        for text in split_long(norm_ws(full)):
            if len(text) < ccfg["min_chars"] or text in held_texts:
                continue
            h = hashlib.md5(text.lower().encode()).hexdigest()
            if h in seen:
                continue
            seen.add(h)
            batch.append((src, text))
        if len(batch) >= 256:
            flush()
            if n_tok >= budget:
                break
    if batch and n_tok < budget:
        flush()
    print(f"== {name}: {len(docs)} docs, {n_tok / 1e6:.1f}M unique tokens "
          f"(budget {budget / 1e6:.0f}M x repeat {dcfg.get('repeat', 1)})")
    for s, v in sorted(stats.items()):
        print(f"   {s:24s} docs={v['docs']:7d} tokens={v['tokens'] / 1e6:7.2f}M")
    return docs, {k: dict(v) for k, v in stats.items()}


def pack(stream, seq_len, cls_id, sep_id, n_domains):
    """stream of (domain_idx, ids) -> (seqs [N, seq_len], domain token counts [N, D])."""
    body = seq_len - 2
    seqs, doms, buf, buf_dom = [], [], [], []
    for d, ids in stream:
        for tok_ids, dom in ((ids, d), (np.array([sep_id], dtype=np.int32), d)):
            buf.append(tok_ids)
            buf_dom.append(np.full(len(tok_ids), dom, dtype=np.int8))
        cat = np.concatenate(buf)
        cat_d = np.concatenate(buf_dom)
        n = len(cat) // body
        for i in range(n):
            chunk = cat[i * body:(i + 1) * body]
            seqs.append(np.concatenate(([cls_id], chunk, [sep_id])).astype(np.int32))
            doms.append(np.bincount(cat_d[i * body:(i + 1) * body], minlength=n_domains))
        buf, buf_dom = [cat[n * body:]], [cat_d[n * body:]]
    if not seqs:
        return np.zeros((0, seq_len), np.int32), np.zeros((0, n_domains), np.int32)
    return np.stack(seqs), np.stack(doms).astype(np.int32)


def main():
    cfg = load_yaml("pretrain.yaml")
    ccfg = cfg["corpus"]
    rng = random.Random(ccfg["seed"])
    out = ROOT / ccfg["out_dir"]
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(cfg["base_model"])
    names = list(ccfg["domains"])

    # eval texts of the fine-tuning task must never be pretrained on
    pdir = ROOT / load_yaml("train.yaml")["data"]["processed_dir"]
    held_texts = {r["text"] for n in ("val", "test") for r in jsonl(pdir / f"{n}.jsonl")}

    train_stream, held_stream, meta = [], [], {"domains": names, "seq_len": ccfg["seq_len"],
                                               "per_domain": {}}
    for d, name in enumerate(names):
        dcfg = ccfg["domains"][name]
        docs, stats = collect(name, dcfg, tok, ccfg, rng, held_texts)
        rng.shuffle(docs)
        n_held = max(1, int(len(docs) * ccfg["heldout_frac"]))
        held, train = docs[:n_held], docs[n_held:]
        held_stream += [(d, ids) for _, ids in held]
        rep = dcfg.get("repeat", 1)
        train_stream += [(d, ids) for _, ids in train] * rep
        meta["per_domain"][name] = {
            "repeat": rep, "sources": stats, "docs": len(docs), "heldout_docs": n_held,
            "unique_tokens": int(sum(len(i) for _, i in docs)),
            "train_tokens": int(sum(len(i) for _, i in train) * rep)}

    rng.shuffle(train_stream)
    tr_ids, tr_dom = pack(train_stream, ccfg["seq_len"], tok.cls_token_id, tok.sep_token_id, len(names))
    np.save(out / "train_ids.npy", tr_ids)
    np.save(out / "train_domain_tokens.npy", tr_dom)
    # held-out: packed per domain so each sequence belongs to one domain
    h_ids, h_dom = [], []
    for d in range(len(names)):
        ids, _ = pack([x for x in held_stream if x[0] == d], ccfg["seq_len"], tok.cls_token_id,
                      tok.sep_token_id, len(names))
        h_ids.append(ids)
        h_dom.append(np.full(len(ids), d, dtype=np.int8))
    np.save(out / "heldout_ids.npy", np.concatenate(h_ids))
    np.save(out / "heldout_domain.npy", np.concatenate(h_dom))
    meta.update({"train_sequences": int(len(tr_ids)), "train_tokens": int(tr_ids.size),
                 "heldout_sequences": int(sum(len(x) for x in h_ids)),
                 "train_mix": {n: round(float(tr_dom[:, i].sum() / tr_dom.sum()), 4)
                               for i, n in enumerate(names)}})
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\ntrain: {len(tr_ids)} seqs = {tr_ids.size / 1e6:.0f}M tokens; mix {meta['train_mix']}")
    print(f"heldout: {meta['heldout_sequences']} seqs")


if __name__ == "__main__":
    main()
