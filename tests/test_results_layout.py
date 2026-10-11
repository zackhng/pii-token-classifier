"""results/ layout: every result file name maps to its series / run folder; writers and readers agree."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import results_layout as rl  # noqa: E402


def test_group_of():
    cases = {
        "SPAN-BS-v5data_ml_wm_dev_grid.json": "v5/runs/SPAN-BS-v5data",
        "A-v5data_merge-avg_ml_real.txt": "v5/runs/A-v5data",
        "B1-v5data-lr3e-4_en_test.json": "v5/runs/B1-v5data-lr3e-4",
        "cpu_SPAN-v5data_int8_sample.json": "v5/runs/SPAN-v5data",
        "v5.0.1_en_test.json": "v5/runs/v5.0.1",
        "pilot2_f25_ml_real.json": "v5/pilots/pilot2_f25",
        "sweep_S64_3e-4_val.json": "v5/sweeps/sweep_S64_3e-4",
        "dhsweep_ft_L4_1e-4_val.json": "v5/sweeps/dhsweep_ft_L4_1e-4",
        "v5_postproc.json": "v5/summary",
        "ml_train_v5_coverage.txt": "v5/train_set",
        "ml_train_coverage.txt": "v4/train_set",
        "A_ml_real.json": "v4/A",
        "B64_ml_real.json": "v4/B64",
        "v2.1_test.txt": "english/v2.1",
        "gpu_latency": "v5/latency/gpu_latency",
    }
    for name, group in cases.items():
        assert rl.group_of(name) == group, name


def test_writer_and_reader_agree(tmp_path, monkeypatch):
    monkeypatch.setattr(rl, "RESULTS", tmp_path)
    out = rl.result_path(tmp_path / "X-v5data_ml_real")            # eval_multilingual style: no extension
    assert Path(out) == tmp_path / "v5/runs/X-v5data/X-v5data_ml_real"
    Path(out + ".json").write_text("{}")
    assert rl.find("X-v5data_ml_real.json") == Path(out + ".json")
    assert rl.result_path(tmp_path / "sub" / "a.json") == str(tmp_path / "sub" / "a.json")  # not flat: unchanged


def test_organize_files_flat_writes(tmp_path, monkeypatch):
    monkeypatch.setattr(rl, "RESULTS", tmp_path)
    (tmp_path / "Y-v5data_en_test.txt").write_text("new")
    (tmp_path / "v5/runs/Y-v5data").mkdir(parents=True)
    (tmp_path / "v5/runs/Y-v5data/Y-v5data_en_test.txt").write_text("old")
    lat = tmp_path / "gpu_latency"
    lat.mkdir()
    (lat / "Y-v5data_1000.txt").write_text('{"chars": 1000, "tokens": 300, "windows": 1, "median_s": 0.01, '
                                           '"min_s": 0.01, "max_s": 0.02}')
    assert rl.organize() == 2
    assert (tmp_path / "v5/runs/Y-v5data/Y-v5data_en_test.txt").read_text() == "new"   # newest write wins
    assert lat.is_dir() and not any(lat.iterdir())        # queue scripts write here without mkdir
    summary = (tmp_path / "v5/latency/gpu_latency/summary.tsv").read_text().splitlines()
    assert summary[1].split("\t")[:2] == ["Y-v5data", "1000"]
    assert rl.find("gpu_latency/summary.tsv") == tmp_path / "v5/latency/gpu_latency/summary.tsv"
