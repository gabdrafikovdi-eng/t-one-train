#!/usr/bin/env python3
"""T-one CPU performance benchmark runner (VPS / Docker).

Runs the FULL series of experiments sequentially, in one process:

  0. system info + cold start (model / KenLM / decoder init)
  1. sequential baseline      (reference quality config, library-default threads)
  2. thread sweep             (1 / 2 / 4 / 6 ORT intra-op threads)
  3. concurrency sweep        (1 / 2 / 4 / 6 parallel requests, 1 thread each)
  4. combined thread+concurrency variants based on the measured best threads
  5. optional beam-width sweep (--beam-sweep 50 100 150), NOT run by default

The decoder pipeline is EXACTLY the quality-baseline configuration that scored
95/109 (reference found) on the real dataset:

  T-one official ONNX -> official kenlm.bin -> pyctcdecode CTC Beam Search
  beam_width=200, alpha=0.4, beta=0.9 -> canonical hotwords from streets.txt
  (hotwords.load_streets) with hotword_weight=10

Threads / concurrency NEVER change decoder parameters; the runner also verifies
that hypotheses are identical across modes (determinism check in summary.json).

All results are written to results/vps_benchmark_<timestamp>/ on the mounted
host volume (summary.json, summary.md, all_results.json/csv, config.json,
system_info.txt, stdout.log and per-mode raw outputs).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.t_one_train import vps_benchmark as vb  # noqa: E402
from src.t_one_train.decoder_experiment import (  # noqa: E402
    env_versions,
    load_wav_any,
    write_json,
    write_rows_csv,
)
from src.t_one_train.hotwords import load_streets  # noqa: F401 (hotwords.json audit)
from src.t_one_train.street_forms_experiment import (  # noqa: E402
    build_canonical_hotwords,
    build_canonical_index,
    load_street_names,
)
from src.t_one_train.test_decoders_core import (  # noqa: E402
    collect_phrase_logprobs,
    decode_with_scores,
    load_acoustic_model,
    load_beam_decoder,
    resolve_model_path,
)
from src.t_one_train.vps_benchmark import (  # noqa: E402
    DEFAULT_CONCURRENCY_MODES,
    DEFAULT_DATASET,
    DEFAULT_THREAD_MODES,
    REFERENCE_ALPHA,
    REFERENCE_BEAM_WIDTH,
    REFERENCE_BETA,
    REFERENCE_HOTWORD_WEIGHT,
    BenchmarkMode,
    ResourceSampler,
    Tee,
    build_experiment_plan,
    build_file_row,
    build_summary,
    chown_results,
    dedupe_plan,
    flatten_rows,
    load_dataset_records,
    pick_best_threads,
    render_summary_md,
    render_system_info,
    results_dir,
    summarize_mode,
    system_info,
    write_mode_outputs,
)

DEFAULT_STREETS = ROOT / "streets.txt"
WARMUP_FILES = 2


class Pipeline:
    """The reference decoder pipeline; only ONNX threads vary between modes."""

    def __init__(self, model_path: str | None, kenlm_path: str | None,
                 streets_file: Path, beam_width: int, hotword_weight: float):
        self.beam_width = int(beam_width)
        self.hotword_weight = float(hotword_weight)
        t0 = time.perf_counter()
        self.model_path = resolve_model_path(model_path)
        self.model_download_seconds = time.perf_counter() - t0

        t0 = time.perf_counter()
        self.model = load_acoustic_model(self.model_path)  # cold start: default threads
        self.model_load_seconds = time.perf_counter() - t0

        t0 = time.perf_counter()
        self.kenlm_path = kenlm_path  # resolved inside load_beam_decoder (HF cache)
        self.beam = load_beam_decoder(kenlm_path, alpha=REFERENCE_ALPHA, beta=REFERENCE_BETA)
        self.kenlm_load_seconds = time.perf_counter() - t0

        t0 = time.perf_counter()
        self.street_names = load_street_names(streets_file)
        self.hotwords = build_canonical_hotwords(self.street_names)
        self.hotword_index = build_canonical_index(self.street_names)
        self.hotwords_audit = load_streets(streets_file)[1]
        if [e.hotword for e in self.hotwords_audit if not e.excluded] != self.hotwords:
            raise ValueError("canonical hotwords differ from hotwords.load_streets output; "
                             "streets.txt parsing is inconsistent")
        self.hotwords_init_seconds = time.perf_counter() - t0

    def with_threads(self, num_threads: int | None) -> "Pipeline":
        """Same pipeline, ONNX session re-created with explicit thread counts."""
        if num_threads is None:
            return self
        clone = object.__new__(Pipeline)
        clone.__dict__.update(self.__dict__)
        clone.model = load_acoustic_model(self.model_path, num_threads=num_threads)
        return clone

    def infer(self, wav_path: Path) -> tuple[str, dict]:
        """One warm inference: WAV load + acoustic model + beam/KenLM decode."""
        t0 = time.perf_counter()
        pcm, _sr = load_wav_any(wav_path)
        wav_load = time.perf_counter() - t0

        t0 = time.perf_counter()
        phrases, frames = collect_phrase_logprobs(self.model, pcm)
        model_t = time.perf_counter() - t0
        n_frames = int(sum(e - s for s, e in frames))

        t0 = time.perf_counter()
        if phrases:
            lp = np.concatenate(phrases, axis=0)
            hypothesis, acoustic_score, combined_score = decode_with_scores(
                self.beam, lp, beam_width=self.beam_width,
                hotwords=self.hotwords, hotword_weight=self.hotword_weight,
            )
        else:
            hypothesis, acoustic_score, combined_score = "", None, None
        decoder_t = time.perf_counter() - t0

        meta = {
            "audio_duration_seconds": len(pcm) / 8000.0,
            "wav_load_seconds": wav_load,
            "model_inference_seconds": model_t,
            "decoder_seconds": decoder_t,
            "n_phrases": len(phrases),
            "n_frames": n_frames,
            "acoustic_score": acoustic_score,
            "combined_score": combined_score,
        }
        return hypothesis, meta

    def detect_streets(self, hypothesis: str) -> list[str]:
        return sorted(self.hotword_index.detect(hypothesis))


def run_mode(*, pipeline: Pipeline, mode: BenchmarkMode, records: list[dict],
             out_dir: Path, warmup: int = WARMUP_FILES,
             sampler_interval: float = 0.5) -> tuple[list[dict], dict]:
    """Run one mode (sequential or thread-pool concurrency) over all records."""
    sampler = ResourceSampler(interval=sampler_interval).start()
    session_init_seconds: float | None = None
    warmup_seconds: float | None = None

    def one(rec: dict, index: int) -> dict:
        started = time.strftime("%Y-%m-%dT%H:%M:%S")
        try:
            hypothesis, meta = pipeline.infer(rec["_wav"])
            return build_file_row(
                mode=mode, index=index, record=rec, hypothesis=hypothesis,
                detected_streets=pipeline.detect_streets(hypothesis),
                started_at=started, **meta,
            )
        except Exception as exc:  # keep the benchmark running on single-file errors
            return build_file_row(
                mode=mode, index=index, record=rec, hypothesis="",
                audio_duration_seconds=0.0, wav_load_seconds=0.0,
                model_inference_seconds=0.0, decoder_seconds=0.0,
                n_phrases=0, n_frames=0, acoustic_score=None,
                combined_score=None, detected_streets=[], started_at=started,
                error=f"{type(exc).__name__}: {exc}",
            )

    try:
        if warmup:
            t0 = time.perf_counter()
            for rec in records[:warmup]:
                pipeline.infer(rec["_wav"])
            warmup_seconds = time.perf_counter() - t0
        if mode.concurrency <= 1:
            rows, wall = [], 0.0
            for i, rec in enumerate(records):
                t0 = time.perf_counter()
                rows.append(one(rec, i))
                wall += time.perf_counter() - t0
        else:
            t0 = time.perf_counter()
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=mode.concurrency
            ) as pool:
                futures = [pool.submit(one, rec, i) for i, rec in enumerate(records)]
                rows = [f.result() for f in futures]
            wall = time.perf_counter() - t0
    finally:
        resources = sampler.stop()

    summary = summarize_mode(
        mode=mode, rows=rows, wall_clock_seconds=wall,
        session_init_seconds=session_init_seconds, resources=resources,
        warmup_seconds=warmup_seconds,
    )
    write_mode_outputs(out_dir, mode, rows, summary)
    print(f"[{mode.name}] files_ok={summary['files_ok']}/{summary['files_total']} "
          f"wall={summary['totals']['wall_clock_seconds']}s "
          f"mean_rtf={(summary['rtf'] or {}).get('mean')}", flush=True)
    return rows, summary

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="T-one CPU VPS performance benchmark.")
    p.add_argument("--dataset", default=DEFAULT_DATASET,
                   help="dataset dir with manifest.jsonl + WAVs")
    p.add_argument("--streets", type=Path, default=DEFAULT_STREETS,
                   help="canonical hotword source (streets.txt)")
    p.add_argument("--out", default="results", help="base results dir (host volume)")
    p.add_argument("--model-path", default=None, help="local model.onnx (else HF cache)")
    p.add_argument("--kenlm-path", default=None, help="local kenlm.bin (else HF cache)")
    p.add_argument("--limit", type=int, default=None, help="first N WAVs only (smoke test)")
    p.add_argument("--threads", type=int, nargs="+", default=list(DEFAULT_THREAD_MODES),
                   help="sequential ORT thread counts to sweep")
    p.add_argument("--concurrency", type=int, nargs="+", default=list(DEFAULT_CONCURRENCY_MODES),
                   help="parallel-request levels to sweep")
    p.add_argument("--beam-sweep", type=int, nargs="*", default=[],
                   help="OPTIONAL extra beam_width sweep (quality/perf trade-off)")
    p.add_argument("--skip-threads", action="store_true")
    p.add_argument("--skip-concurrency", action="store_true")
    p.add_argument("--skip-combined", action="store_true")
    p.add_argument("--no-warmup", action="store_true")
    p.add_argument("--sampler-interval", type=float, default=0.5)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = results_dir(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    with Tee(out_dir / "stdout.log"):
        t_total0 = time.perf_counter()
        records = load_dataset_records(args.dataset, limit=args.limit)
        print(f"Dataset: {args.dataset} -> {len(records)} WAVs (limit={args.limit})")
        print(f"Streets: {args.streets}")
        warmup = 0 if args.no_warmup else WARMUP_FILES

        # --- cold start (measured separately, never part of inference latency) --
        t0 = time.perf_counter()
        pipeline = Pipeline(args.model_path, args.kenlm_path, args.streets,
                            REFERENCE_BEAM_WIDTH, REFERENCE_HOTWORD_WEIGHT)
        startup = {
            "startup_total_seconds": round(time.perf_counter() - t0, 4),
            "model_resolve_seconds": round(pipeline.model_download_seconds, 4),
            "acoustic_model_load_seconds": round(pipeline.model_load_seconds, 4),
            "kenlm_decoder_init_seconds": round(pipeline.kenlm_load_seconds, 4),
            "hotwords_build_seconds": round(pipeline.hotwords_init_seconds, 4),
            "hotwords_count": len(pipeline.hotwords),
            "hotwords_source": str(args.streets),
            "kenlm_path": str(pipeline.kenlm_path),
            "model_path": str(pipeline.model_path),
        }
        print(f"Cold start: {startup['startup_total_seconds']}s "
              f"(model {pipeline.model_load_seconds:.2f}s, "
              f"kenlm+decoder {pipeline.kenlm_load_seconds:.2f}s, "
              f"hotwords {pipeline.hotwords_init_seconds:.2f}s "
              f"n={len(pipeline.hotwords)})")

        sys_info = system_info({"package_versions": env_versions()})
        (out_dir / "system_info.txt").write_text(
            render_system_info(sys_info), encoding="utf-8")

        config = {
            "dataset": str(args.dataset),
            "dataset_files": len(records),
            "streets_file": str(args.streets),
            "model": "t-tech/T-one (official ONNX)",
            "kenlm": "official kenlm.bin (t-tech/T-one)",
            "decoder": "pyctcdecode CTC Beam Search",
            "beam_width": REFERENCE_BEAM_WIDTH,
            "alpha": REFERENCE_ALPHA,
            "beta": REFERENCE_BETA,
            "hotwords_source": str(args.streets),
            "hotword_weight": REFERENCE_HOTWORD_WEIGHT,
            "hotwords_count": len(pipeline.hotwords),
            "timestamp": out_dir.name.replace(vb.RESULTS_PREFIX, ""),
            "git_commit": sys_info.get("git_commit"),
            "limit": args.limit,
            "warmup_files": warmup,
        }
        write_json(out_dir / "config.json", config)
        from src.t_one_train.decoder_experiment import build_hotwords_json
        write_json(out_dir / "hotwords.json",
                   build_hotwords_json(pipeline.hotwords_audit, str(args.streets)))

        # --- experiment plan (strict order; baseline first) --------------------
        plan = build_experiment_plan(
            thread_modes=tuple(args.threads),
            concurrency_modes=tuple(args.concurrency),
            with_threads=not args.skip_threads,
            with_concurrency=not args.skip_concurrency,
            with_combined=not args.skip_combined,
            beam_sweep=tuple(args.beam_sweep),
        )
        static_plan = [
            m for m in plan if m.kind in ("sequential", "threads", "beam_sweep")
        ]

        rows_by_mode: dict[str, list[dict]] = {}
        summaries: list[dict] = []

        def run_one(mode: BenchmarkMode, pipe: Pipeline) -> None:
            print(f"=== mode {mode.name} ===", flush=True)
            rows, summary = run_mode(
                pipeline=pipe, mode=mode, records=records, out_dir=out_dir,
                warmup=warmup, sampler_interval=args.sampler_interval,
            )
            rows_by_mode[mode.name] = rows
            summaries.append(summary)

        for mode in static_plan:
            pipe = pipeline if mode.threads is None else pipeline.with_threads(mode.threads)
            run_one(mode, pipe)

        best_threads = pick_best_threads(summaries)
        print(f"Best sequential threads (measured): "
              f"{best_threads if best_threads else 'library default'}", flush=True)

        # concurrency sweep + production-oriented best-threads variants
        concurrency_plan = [
            m for m in build_experiment_plan(
                thread_modes=tuple(args.threads),
                concurrency_modes=tuple(args.concurrency),
                best_threads=best_threads,
                with_threads=False,
                with_combined=not args.skip_combined,
                beam_sweep=(),
            ) if m.kind in ("concurrency", "combined")
        ]
        for mode in dedupe_plan(concurrency_plan):
            pipe = pipeline if mode.threads is None else pipeline.with_threads(mode.threads)
            run_one(mode, pipe)

        # --- aggregated outputs ------------------------------------------------
        config["git_commit"] = sys_info.get("git_commit")
        summary = build_summary(
            config=config, system=sys_info, startup=startup,
            modes=summaries, rows_by_mode=rows_by_mode,
            plan=[m.as_dict() for m in static_plan + concurrency_plan],
        )
        write_json(out_dir / "summary.json", summary)
        write_json(out_dir / "all_results.json",
                   {name: rows for name, rows in rows_by_mode.items()})
        csv_rows = [r for rows in rows_by_mode.values() for r in flatten_rows(rows)]
        write_rows_csv(out_dir / "all_results.csv", vb.MODE_FIELDS, csv_rows)
        (out_dir / "summary.md").write_text(render_summary_md(summary), encoding="utf-8")

        q = summary["quality"]
        print("\n==== BENCHMARK COMPLETE ====")
        print(f"quality (reference found): {q['reference_found']}/{q['files']} "
              f"= {q['reference_found_accuracy']}")
        print(f"expected baseline 95/109: match={q['matches_expected_baseline']}")
        print(f"bottleneck: {summary['bottleneck']}")
        print(f"results dir: {out_dir}")

    # results written by the container root -> give them back to the host user
    host_uid = os.environ.get("HOST_UID") or os.environ.get("HOST_GID")
    host_gid = os.environ.get("HOST_GID")
    chown_results(
        out_dir,
        int(host_uid) if host_uid and host_uid.isdigit() else None,
        int(host_gid) if host_gid and host_gid.isdigit() else None,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
