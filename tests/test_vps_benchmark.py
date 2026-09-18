"""Headless tests for the T-one VPS performance benchmark (no model needed)."""

from __future__ import annotations

import json
import time

import numpy as np
import pytest
import soundfile as sf

from src.t_one_train import vps_benchmark as vb
from src.t_one_train.vps_benchmark import (
    BenchmarkMode,
    ResourceSampler,
    build_experiment_plan,
    build_file_row,
    build_summary,
    dedupe_plan,
    determinism_check,
    detect_bottleneck,
    is_exact_match,
    load_dataset_records,
    percentile,
    pick_best_threads,
    reference_found,
    render_summary_md,
    render_system_info,
    results_dir,
    summarize_mode,
    system_info,
)


# --- benchmark geometry -------------------------------------------------------


def test_benchmark_mode_naming_and_dirs():
    seq = BenchmarkMode(kind="sequential", threads=None, concurrency=1)
    assert seq.name == "sequential" and seq.subdir == "sequential"
    th = BenchmarkMode(kind="threads", threads=4, concurrency=1)
    assert th.name == "threads_4" and th.subdir == "threads/threads_4"
    cc = BenchmarkMode(kind="concurrency", threads=1, concurrency=4)
    assert cc.name == "threads_1_concurrency_4"
    assert cc.subdir == "concurrency/threads_1_concurrency_4"
    comb = BenchmarkMode(kind="combined", threads=2, concurrency=2)
    assert comb.name == "threads_2_concurrency_2"
    assert comb.subdir == "combined/threads_2_concurrency_2"
    beam = BenchmarkMode(kind="beam_sweep", threads=None, concurrency=1, beam_width=50)
    assert beam.name == "beam_50" and beam.subdir == "beam_sweep/beam_50"


def test_mode_as_dict_has_reference_params():
    d = BenchmarkMode(kind="sequential", threads=None, concurrency=1).as_dict()
    assert d["beam_width"] == vb.REFERENCE_BEAM_WIDTH == 200


# --- dataset ------------------------------------------------------------------


@pytest.fixture()
def fake_dataset(tmp_path):
    d = tmp_path / "ds"
    d.mkdir()
    rng = np.random.default_rng(0)
    rows = []
    for i, (status, variant) in enumerate(
        [("saved", "full"), ("saved", "full"), ("saved", "cut"), ("failed", "full")]
    ):
        wav = d / f"rec{i}.wav"
        sf.write(wav, rng.integers(-1000, 1000, 8000, dtype=np.int16), 8000)
        rows.append({"id": f"street_{i:03d}_{variant}", "status": status,
                     "variant": variant, "audio": wav.name,
                     "street": "ленина", "reference": "улица ленина"})
    (d / "manifest.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return d


def test_load_dataset_records_full_variant_only(fake_dataset):
    recs = load_dataset_records(fake_dataset)
    assert [r["id"] for r in recs] == ["street_000_full", "street_001_full"]
    assert all(r["_wav"].exists() for r in recs)
    assert len(load_dataset_records(fake_dataset, limit=1)) == 1


def test_load_dataset_records_missing_manifest(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_dataset_records(tmp_path)


# --- matching / metrics -------------------------------------------------------


def test_exact_match_and_reference_found():
    assert is_exact_match("Улица Ленина", "ленина") is False  # not equal as whole
    assert is_exact_match("улица ленина", "Улица Ленина")
    assert reference_found("я еду по ленина 5", "ленина")
    assert not reference_found("я еду по садовой", "ленина")
    assert not reference_found("", "ленина")



def test_percentile_and_distribution():
    vals = list(range(1, 101))
    assert percentile(vals, 50) == pytest.approx(50.5)
    assert percentile(vals, 0) == 1 and percentile(vals, 100) == 100
    assert percentile([], 95) is None
    dist = vb.distribution(vals)
    assert dist["n"] == 100 and dist["p95"] == pytest.approx(95.05)
    assert dist["min"] == 1 and dist["max"] == 100


def _rows(n: int = 4, mode: BenchmarkMode | None = None, errors: int = 0) -> list[dict]:
    mode = mode or BenchmarkMode(kind="sequential", threads=None, concurrency=1)
    out = []
    for i in range(n):
        rec = {"id": f"f{i}", "reference": "улица ленина", "_wav": f"/tmp/f{i}.wav"}
        err = f"Boom{i}" if i < errors else None
        out.append(build_file_row(
            mode=mode, index=i, record=rec,
            hypothesis="улица ленина" if not err else "",
            audio_duration_seconds=2.0 + i,
            wav_load_seconds=0.1, model_inference_seconds=0.5,
            decoder_seconds=0.2, n_frames=100, n_phrases=1,
            acoustic_score=-10.0, combined_score=-12.0,
            detected_streets=["ленина"], started_at="t", error=err,
        ))
    return out


def test_build_file_row_metrics():
    row = _rows(1)[0]
    # total = 0.1 + 0.5 + 0.2 = 0.8; duration 2.0 -> rtf 0.4
    assert row["total_inference_seconds"] == pytest.approx(0.8)
    assert row["rtf"] == pytest.approx(0.4)
    assert row["exact_match"] and row["reference_found"]
    assert row["detected_streets"] == "ленина"
    assert row["beam_width"] == 200


def test_summarize_mode_aggregates():
    mode = BenchmarkMode(kind="sequential", threads=None, concurrency=1)
    rows = _rows(4, mode=mode)
    s = summarize_mode(mode=mode, rows=rows, wall_clock_seconds=3.2,
                       session_init_seconds=1.0, resources={"peak": 1})
    assert s["files_total"] == 4 and s["files_ok"] == 4 and s["files_failed"] == 0
    assert s["exact_matches"] == 4
    assert s["exact_accuracy"] == 1.0
    assert s["reference_found"] == 4
    assert s["latency_seconds"]["mean"] == pytest.approx(0.8)
    assert s["latency_seconds"]["p95"] is not None
    # per-file RTF mean = mean(0.8/2, 0.8/3, 0.8/4, 0.8/5)
    assert s["rtf"]["mean"] == pytest.approx(0.2566667, abs=1e-4)
    assert s["components_seconds"]["model_inference_total"] == pytest.approx(2.0)
    assert s["components_seconds"]["decoder_total"] == pytest.approx(0.8)
    assert s["components_seconds"]["compute_total"] == pytest.approx(2.8)
    assert s["totals"]["wall_clock_seconds"] == pytest.approx(3.2)
    assert s["audio"]["total_duration_seconds"] == pytest.approx(2.0 + 3 + 4 + 5)
    assert s["mode"] == "sequential"


def test_summarize_mode_counts_failures():
    mode = BenchmarkMode(kind="sequential", threads=None, concurrency=1)
    s = summarize_mode(mode=mode, rows=_rows(4, mode=mode, errors=1),
                       wall_clock_seconds=1.0, session_init_seconds=None,
                       resources=None)
    assert s["files_ok"] == 3 and s["files_failed"] == 1
    assert s["failed_ids"] == ["f0"]


# --- experiment plan ----------------------------------------------------------


def test_plan_sequential_first_and_no_duplicates():
    plan = build_experiment_plan(thread_modes=(1, 2), concurrency_modes=(2, 4))
    kinds = [m.kind for m in plan]
    assert kinds[0] == "sequential"
    names = [m.name for m in plan]
    assert "threads_1" in names and "threads_2" in names
    assert "threads_1_concurrency_2" in names
    assert "threads_2_concurrency_2" in names  # combined threads==concurrency
    keys = [(m.threads, m.concurrency, m.beam_width)
            for m in plan if m.kind != "sequential"]
    assert len(keys) == len(set(keys))


def test_plan_best_threads_variants():
    plan = build_experiment_plan(thread_modes=(1, 2, 4), concurrency_modes=(2, 4),
                                 best_threads=4, with_threads=False,
                                 with_combined=False)
    names = [m.name for m in plan]
    assert names[0] == "sequential"
    assert "threads_4_concurrency_2" in names
    assert "threads_4_concurrency_4" in names
    assert "threads_2_concurrency_2" not in names


def test_dedupe_keeps_order():
    plan = [
        BenchmarkMode(kind="concurrency", threads=1, concurrency=2),
        BenchmarkMode(kind="concurrency", threads=1, concurrency=2),
        BenchmarkMode(kind="combined", threads=2, concurrency=2),
    ]
    out = dedupe_plan(plan)
    assert len(out) == 2 and out[0].concurrency == 2 and out[1].concurrency == 2


def test_pick_best_threads_prefers_lowest_mean():
    def s(mode: str, threads, mean):
        return {"kind": mode, "threads": threads, "concurrency": 1,
                "beam_width": 200, "latency_seconds": {"mean": mean}}

    # threads sweep beats the library-default sequential run
    assert pick_best_threads([s("sequential", None, 0.5), s("threads", 1, 0.4),
                              s("threads", 2, 0.3)]) == 2
    # ties -> fewer threads
    assert pick_best_threads([s("threads", 2, 0.4), s("threads", 4, 0.4)]) == 2
    # library default wins -> None
    assert pick_best_threads([s("sequential", None, 0.1), s("threads", 2, 0.4)]) is None


def test_determinism_check():
    base = {"sequential": _rows(2)}
    other_mode = BenchmarkMode(kind="threads", threads=1, concurrency=1)
    same = _rows(2, mode=other_mode)
    diff = [dict(r, hypothesis="другое") for r in _rows(2, mode=other_mode)]
    assert determinism_check(base)["identical"]
    chk = determinism_check({**base, "threads_1": diff})
    assert not chk["identical"] and chk["differences_total"] == 2
    assert not determinism_check({})["checked"]


def test_detect_bottleneck():
    baseline = {"components_seconds": {"model_inference_total": 9.0,
                                       "decoder_total": 1.0,
                                       "compute_total": 10.0}}
    b = detect_bottleneck(baseline)
    assert b["verdict"] == "acoustic_model" and b["model_share"] == pytest.approx(0.9)
    b2 = detect_bottleneck({"components_seconds": {"model_inference_total": 1.0,
                                                   "decoder_total": 9.0,
                                                   "compute_total": 10.0}})
    assert b2["verdict"] == "decoder_beam_search_kenlm"


# --- summary / report ---------------------------------------------------------


def test_build_summary_quality_section():
    mode = BenchmarkMode(kind="sequential", threads=None, concurrency=1)
    rows = _rows(4, mode=mode)
    s = summarize_mode(mode=mode, rows=rows, wall_clock_seconds=1.0,
                       session_init_seconds=None, resources=None)
    total = build_summary(config={"x": 1}, system={"cpu": {}}, startup={},
                          modes=[s], rows_by_mode={"sequential": rows})
    assert total["quality"]["files"] == 4
    assert total["quality"]["reference_found"] == 4
    assert total["quality"]["matches_expected_baseline"] is False  # 4 != 109
    assert total["bottleneck"]["verdict"] in ("acoustic_model",
                                              "decoder_beam_kenlm")
    md = render_summary_md(total)
    assert "| Test |" in md and "Quality" in md and "Bottleneck" in md
    assert total["determinism"]["checked"]


def test_render_system_info_handles_missing_fields():
    text = render_system_info(system_info())
    assert "T-one VPS benchmark" in text and "[CPU]" in text and "[RAM]" in text


def test_results_dir_unique(tmp_path):
    d1 = results_dir(tmp_path)
    d2 = results_dir(tmp_path)
    assert d1 != d2
    assert d1.name.startswith("vps_benchmark_") and d2.name.startswith("vps_benchmark_")


def test_resource_sampler_smoke():
    rs = ResourceSampler(interval=0.05).start()
    x = 0
    for _ in range(200_000):
        x += 1
    time.sleep(0.15)
    agg = rs.stop()
    assert x > 0
    assert agg["samples"] >= 1
    assert agg["cpu_seconds"] >= 0.0
    assert "cpu_peak_percent" in agg and "ram_rss_peak_mb" in agg

    mode = BenchmarkMode(kind="sequential", threads=None, concurrency=1)
    s = summarize_mode(mode=mode, rows=_rows(4, mode=mode, errors=1),
                       wall_clock_seconds=1.0, session_init_seconds=None,
                       resources=None)
    assert s["files_ok"] == 3 and s["files_failed"] == 1
    assert s["failed_ids"] == ["f0"]
