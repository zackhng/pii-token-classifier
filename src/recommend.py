"""README model suggestions (v5 series), rebuilt from saved predictions and the latency summaries (no model
runs). Picks the most accurate model, the most cost-efficient one (accuracy vs CPU latency), the widest
coverage, and the best model per entity and per language. Rewrites the block between the markers in README.md:

    <!-- v5-suggestions:start --> ... <!-- v5-suggestions:end -->

    python recommend.py          (experiments_table.py also calls it)

Candidates = every trained v5 model (v5 data, status done) with saved predictions on all six multilingual sets.
Accuracy = mean strict F1 over the six sets; coverage = mean coverage over the six sets (scoring as in
experiments_table.py). Per entity / language: the six sets pooled (breakdown.py).
"""
import csv

from breakdown import ENTS, LANGS, cell, counts_for
from common import ROOT
from results_layout import find
from experiments_table import DONE, EXPS, SETS, score

START, END = "<!-- v5-suggestions:start -->", "<!-- v5-suggestions:end -->"
FAST_CPU_S = 10.0       # "CPU-fast" = at most this many seconds for a 100k-character document on 6 CPU threads
EFFICIENT_GAP = 0.010   # cost-efficient pick: fastest model whose mean strict F1 is within this of the best
NOISE = 0.005           # single seed: differences below this are not treated as real


def latency(dev):
    """{tag: {chars: median seconds}} from results/{gpu,cpu}_latency/summary.tsv."""
    p = find(f"{dev}_latency/summary.tsv")
    out = {}
    if p.exists():
        for r in csv.DictReader(open(p, encoding="utf-8"), delimiter="\t"):
            out.setdefault(r["model"], {})[int(r["chars"])] = float(r["median_s"])
    return out


def candidates():
    rows = []
    for _grp, name, tag, _log, _change, _parent, status in EXPS:
        if status != DONE or not tag.endswith("-v5data"):
            continue
        r = [score(tag, s) for s in SETS]
        if any(x is None for x in r):
            continue
        rows.append({"tag": tag, "name": name.replace("**", ""),
                     "strict": sum(x[0] for x in r) / len(r), "cov": sum(x[1] for x in r) / len(r),
                     "sets": r})
    return rows


LEGEND = [
    ("v5.0 (A-v5data)", "XLM-R base, fully fine-tuned, softmax BIO head (one label per token), trained on ml_train_v5. "
                        "The released v5 model and the parent of every other row."),
    ("H1", "encoder frozen, only the classifier head trains"),
    ("P1", "only the top 4 encoder layers + head train (partial unfreeze)"),
    ("S1", "encoder frozen + one shared bottleneck adapter for all languages"),
    ("B1 / B′", "encoder frozen + one adapter per language, routed by the window's main language "
                "(B1 on pretrained XLM-R, B′ on v5.0's encoder). Dropped: switching adapters adds latency on "
                "long mixed-language documents"),
    ("AL2 / AL4 / AL8", "\"Additional Layers\": v5.0 with 2 / 4 / 8 extra transformer layers between the encoder "
                        "and the classifier"),
    ("CRF", "v5.0 with a linear-chain CRF on the classifier: decodes the best label sequence per window instead "
            "of each token alone, so B/I tags stay consistent"),
    ("SPAN", "v5.0 with a span head (GlobalPointer) instead of per-token labels: scores every (start word, end word, "
             "type) candidate up to 160 tokens long and keeps the best non-overlapping spans. Better boundaries "
             "on long spans (ADDRESS, BUSINESS)"),
    ("SPAN-CE", "SPAN with a different loss: each candidate span is a softmax over {not PII + 8 types} "
                "(the span-classification loss of Zhu & Li, ACL 2022)"),
    ("SPAN-BS", "SPAN-CE + boundary smoothing (Zhu & Li 2022): 10% of each gold span's target is spread over spans "
                "whose start or end is one word off, so near-miss boundaries are not punished as hard"),
    ("mmBERT-small / -base", "encoder swapped to mmBERT (ModernBERT architecture, 1,800+ languages); 140M / 307M "
                             "parameters"),
    ("MiniLM-L12 / -L6", "encoder swapped to Multilingual-MiniLMv2, 12 or 6 layers of width 384, distilled from "
                         "XLM-R large (same tokenizer as XLM-R); about 117M / 107M parameters, mostly the "
                         "embedding table, so far fewer compute per token"),
    ("… + span / + span-BS", "that encoder with the SPAN or SPAN-BS head instead of the softmax head"),
    ("InfoXLM-base", "XLM-R base further pretrained with cross-lingual contrastive objectives (Microsoft)"),
    ("XLM-V-base", "XLM-R base architecture with a 1M-token vocabulary (fewer, better tokens for many languages)"),
    ("mDeBERTa-v3-base", "multilingual DeBERTa v3 (disentangled attention, ELECTRA-style pretraining)"),
    ("XLM-R large", "24-layer, 560M-parameter XLM-R"),
    ("merge central / avg, overlap 64", "inference-only changes: how predictions from overlapping 384-token windows "
                                        "are combined, and a 64-token instead of 96-token window overlap"),
]


def build_block():
    C = candidates()
    if not C:
        return START + "\n" + END
    cpu, gpu = latency("cpu"), latency("gpu")
    for c in C:
        c["cpu1k"] = cpu.get(c["tag"], {}).get(1000)
        c["cpu100k"] = cpu.get(c["tag"], {}).get(100000)
        c["gpu100k"] = gpu.get(c["tag"], {}).get(100000)
    by = {c["tag"]: c for c in C}
    base = by.get("A-v5data")
    f3 = lambda v: "–" if v is None else f"{v:.3f}"
    fs = lambda v: "–" if v is None else f"{v:.2f} s" if v >= 0.1 else f"{v * 1000:.0f} ms"

    acc = max(C, key=lambda c: c["strict"])
    cov = max(C, key=lambda c: c["cov"])
    timed = [c for c in C if c["cpu100k"] is not None]
    eff = min((c for c in timed if c["strict"] >= acc["strict"] - EFFICIENT_GAP),
              key=lambda c: c["cpu100k"], default=None)
    fastest_ok = min((c for c in timed if base and c["strict"] >= base["strict"] - 0.010),
                     key=lambda c: c["cpu100k"], default=None)

    def near(best, key):     # other models within NOISE of the winner
        t = [c["name"] for c in sorted(C, key=lambda c: -c[key]) if c is not best and best[key] - c[key] < NOISE]
        return t[:3] + ([f"{len(t) - 3} more"] if len(t) > 3 else [])

    def speed(c):
        if c["cpu100k"] is None or not base or base["cpu100k"] is None:
            return "CPU latency not measured yet"
        return (f"CPU {fs(c['cpu100k'])} per 100k characters ({base['cpu100k'] / c['cpu100k']:.1f}× v5.0's speed), "
                f"{fs(c['cpu1k'])} per 1k; GPU {fs(c['gpu100k'])} per 100k")

    def summary_row(use, c, why):
        return (f"| {use} | **{c['name']}** | {f3(c['strict'])} | {f3(c['cov'])} | {fs(c['cpu1k'])} | "
                f"{fs(c['cpu100k'])} | {fs(c['gpu100k'])} | {why} |")

    lines = [START, "## Which v5 model to use (suggestions)", "",
             f"Built from the experiment log below by `src/recommend.py` (re-run by `experiments_table.py`), "
             f"so it changes as queued runs finish. **Accuracy** = mean strict F1 over the six multilingual sets "
             f"(struct, wm, synth, real, mixed, Kiii); **coverage** = mean share of gold PII spans overlapped by "
             f"any prediction. Latency: one document, ml_real text, CPU = 6 threads fp32, GPU = RTX 5060 Ti, median of "
             f"warm runs. Single seed: gaps under {NOISE:.3f} are within noise, so near-ties are listed. Dev sets "
             f"only; nothing here has been checked on the held-out test sets.", "",
             "**Model names**", "", "| Name | What it is |", "|---|---|",
             *[f"| {n} | {d} |" for n, d in LEGEND], "",
             "| Use | Model | Accuracy | Coverage | CPU 1k chars | CPU 100k chars | GPU 100k chars | Why |",
             "|---|---|---|---|---|---|---|---|"]
    t = near(acc, "strict")
    lines.append(summary_row("Most accurate", acc, "highest mean strict F1" + (
        f"; within noise: {', '.join(t)}" if t else "")))
    if eff:
        lines.append(summary_row("Most cost-efficient (CPU)", eff,
                                 f"fastest model within {EFFICIENT_GAP:.3f} strict F1 of the most accurate"
                                 + (f"; {base['cpu100k'] / eff['cpu100k']:.1f}× faster than v5.0 on CPU"
                                    if base and base["cpu100k"] else "")))
    if fastest_ok and fastest_ok is not eff:
        lines.append(summary_row("Fastest acceptable", fastest_ok,
                                 "fastest model within 0.010 strict F1 of v5.0"))
    t = near(cov, "cov")
    lines.append(summary_row("Widest coverage (masking)", cov, "highest mean coverage" + (
        f"; within noise: {', '.join(t)}" if t else "")))
    if base:
        lines.append(summary_row("Reference", base, "v5.0 as released"))
    lines.append("")

    # accuracy vs latency: models not beaten on both accuracy and CPU latency by another model
    front = [c for c in timed if not any(o["strict"] >= c["strict"] and o["cpu100k"] < c["cpu100k"]
                                         and o is not c for o in timed)]
    if front:
        lines += ["### Accuracy vs CPU latency (models that nothing beats on both)", "",
                  "| Model | Accuracy | Coverage | CPU 100k chars | Speed vs v5.0 |", "|---|---|---|---|---|"]
        for c in sorted(front, key=lambda c: c["cpu100k"]):
            sp = f"{base['cpu100k'] / c['cpu100k']:.1f}×" if base and base["cpu100k"] else "–"
            lines.append(f"| {c['name']} | {f3(c['strict'])} | {f3(c['cov'])} | {fs(c['cpu100k'])} | {sp} |")
        lines.append("")

    # per entity and per language (six sets pooled)
    K = {c["tag"]: counts_for(c["tag"]) for c in C}
    fast = [c for c in timed if c["cpu100k"] <= FAST_CPU_S]

    def best(pool, key, i):
        vals = [(cell(K[c["tag"]], ("ALLSETS", key)), c) for c in pool]
        vals = [(v, c) for v, c in vals if v]
        if not vals:
            return "–"
        vals.sort(key=lambda x: -x[0][i])
        (v, c), rest = vals[0], vals[1:]
        tie = [o["name"] for w, o in rest if v[i] - w[i] < NOISE]
        return f"{c['name']} {v[i]:.3f}" + (f" (ties: {', '.join(tie[:3])}{' …' if len(tie) > 3 else ''})"
                                            if tie else "")

    def ref(c, key):
        v = cell(K[c["tag"]], ("ALLSETS", key)) if c else None
        return "–" if not v else f"{v[0]:.3f} / {v[1]:.3f}"

    for title, rows in (("entity", [(e, f"ent/{e}") for e in ENTS]),
                        ("language", [(l, f"lang/{l}") for l in LANGS])):
        lines += [f"### Best model per {title} (six sets pooled)", "",
                  f"Best strict F1, best coverage, and the best strict F1 among CPU-fast models (≤ {FAST_CPU_S:.0f} s "
                  f"per 100k characters: {', '.join(c['name'] for c in fast) or 'none measured'}). Last columns: "
                  f"strict F1 / coverage of the overall picks for comparison.", "",
                  f"| {title.capitalize()} | n | Best strict F1 | Best coverage | Best CPU-fast strict F1 | "
                  f"{acc['name']} | {eff['name'] if eff else '–'} |",
                  "|---|---|---|---|---|---|---|"]
        for label, key in rows:
            n = next((v[2] for v in (cell(K[c["tag"]], ("ALLSETS", key)) for c in C) if v), 0)
            if not n:
                continue
            lines.append(f"| {label} | {n} | {best(C, key, 0)} | {best(C, key, 1)} | {best(fast, key, 0)} | "
                         f"{ref(acc, key)} | {ref(eff, key)} |")
        lines.append("")
    lines.append(END)
    return "\n".join(lines)


def write_readme():
    block = build_block()
    readme = ROOT / "README.md"
    text = readme.read_text(encoding="utf-8")
    if START in text:
        text = text[:text.index(START)] + block + text[text.index(END) + len(END):]
    else:
        anchor = "<!-- v5-experiments:start -->"
        text = text.replace(anchor, block + "\n\n" + anchor, 1)
    readme.write_text(text, encoding="utf-8")
    return block


if __name__ == "__main__":
    print(write_readme())
