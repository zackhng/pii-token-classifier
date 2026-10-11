"""Where result files live. results/ is grouped by series and run:

    results/english/<v1 | v2 | v2.1 | v3 | misc>/   English-only models (v1-v3)
    results/v4/<A | B64 | A-minilm | A-pilot>/        v4 releases (A = v4.1, B64 = v4.2, A-minilm = v4.3)
    results/v4/{train_set,latency,router}/
    results/v5/runs/<run tag>/                        every v5 model, incl. its inference-only variants
                                                      (merge modes, overlap 64, latency, int8 CPU)
    results/v5/{pilots,sweeps}/<tag>/                 frame-share pilots, learning-rate sweeps
    results/v5/train_set/                             training-data coverage, difficulty (CEIM)
    results/v5/latency/{gpu,cpu}_latency/             latency by document size (summary.tsv)
    results/v5/summary/                               cross-run tables (breakdowns, thresholds, release)

File names are unchanged (they start with the run tag), so a file is found by name wherever it is:

    find("SPAN-v5data_en_test.json")          readers
    result_path("../results/X_ml_real")       writers: a flat results/ path -> its group folder

Files written straight into results/ (shell redirects in queue scripts) are filed by organize(), which
experiments_table.py runs first:  python results_layout.py
"""
import os
import re
import sys
from pathlib import Path

from common import ROOT

RESULTS = ROOT / "results"
V4 = {"A", "B64", "A-minilm", "A-pilot"}
ENGLISH = {"v1", "v2", "v2.1", "v3"}


def group_of(name: str) -> str:
    """results/-relative folder for a file (or latency folder) name."""
    if name in ("gpu_latency", "cpu_latency"):
        return f"v5/latency/{name}"
    f0 = name.split("_")[0]
    if f0 in ENGLISH:
        return f"english/{f0}"
    if name.startswith("test_metrics"):
        return "english/misc"
    if f0 in V4:
        return f"v4/{f0}"
    if name.startswith("latency_"):
        return "v4/latency"
    if name.startswith("router_"):
        return "v4/router"
    if name.startswith("ml_train_v5") or f0 == "difficulty":
        return "v5/train_set"
    if name.startswith("ml_train"):
        return "v4/train_set"
    m = re.match(r"(pilot2?_f\d+)", name)
    if m:
        return f"v5/pilots/{m.group(1)}"
    m = re.match(r"(sweep_[^_]+_[^_]+|dhsweep_[^_]+_[^_]+_[^_]+)_", name)
    if m:
        return f"v5/sweeps/{m.group(1)}"
    m = re.match(r"cpu_(.+?-v5data)_(fp32|int8)", name)
    if m:
        return f"v5/runs/{m.group(1)}"
    if f0 == "v5":
        return "v5/summary"
    if f0.endswith("-v5data") or "-v5data-" in f0 or f0.startswith("v5."):
        return f"v5/runs/{f0}"
    return "misc"


def result_path(out) -> str:
    """Writers: an output path directly in results/ (with or without extension) -> the same name in its
    group folder (created). Any other path is returned unchanged."""
    p = Path(out).resolve()
    if p.parent != RESULTS.resolve():
        return str(out)
    d = RESULTS / group_of(p.name)
    d.mkdir(parents=True, exist_ok=True)
    return str(d / p.name)


def find(rel: str) -> Path:
    """Readers: results file by name, or by a short relative path such as 'gpu_latency/summary.tsv'.
    Looks in the flat legacy location first, then in the group folder, then anywhere under results/."""
    flat = RESULTS / rel
    if flat.exists():
        return flat
    name = Path(rel).name
    head = Path(rel).parts[0] if len(Path(rel).parts) > 1 else name
    grouped = RESULTS / group_of(head) / (rel if len(Path(rel).parts) == 1 else Path(*Path(rel).parts[1:]))
    if grouped.exists():
        return grouped
    hits = [p for p in RESULTS.rglob(name) if p.as_posix().endswith(rel)]
    return hits[0] if hits else flat


def latency_summary(d: Path):
    """Rebuild <d>/summary.tsv from the bench_latency.py outputs <tag>_<chars>.txt (last line = JSON)."""
    import json
    keys = ("chars", "tokens", "windows", "median_s", "min_s", "max_s")
    rows = ["\t".join(("model",) + keys)]
    for f in sorted(d.glob("*_*.txt")):
        try:
            r = json.loads(f.read_text(encoding="utf-8").strip().splitlines()[-1])
        except (ValueError, IndexError):
            continue                        # failed or unfinished run
        rows.append("\t".join([f.stem.rsplit("_", 1)[0]] + [str(r[k]) for k in keys]))
    (d / "summary.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")


def organize(verbose=False) -> int:
    """Move files and latency folders lying directly in results/ into their group folders. Idempotent. A flat
    file is always the newest write (a re-run), so it replaces a grouped file of the same name."""
    moved = 0
    for p in sorted(RESULTS.iterdir()):
        if p.is_dir() and p.name not in ("gpu_latency", "cpu_latency"):
            continue
        dest = RESULTS / group_of(p.name)
        if p.is_dir():                      # merge the folder's files into results/v5/latency/<name>/
            dest.mkdir(parents=True, exist_ok=True)
            new = [f for f in p.iterdir() if f.name != "summary.tsv"]
            for f in new:
                os.replace(f, dest / f.name)
                moved += 1
            if new:
                latency_summary(dest)
            continue                        # the empty folder stays: queue scripts write into it without mkdir
        dest.mkdir(parents=True, exist_ok=True)
        os.replace(p, dest / p.name)
        moved += 1
        if verbose:
            print(f"{p.name} -> {dest.relative_to(RESULTS)}/")
    return moved


if __name__ == "__main__":
    print(f"moved {organize(verbose='-v' in sys.argv)} files")
