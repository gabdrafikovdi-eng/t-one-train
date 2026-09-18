"""Pure, dependency-light logic for the T-one VPS **performance** benchmark.

This module deliberately has NO tone/pyctcdecode/onnxruntime imports: the runner
(``scripts/run_vps_benchmark.py``) owns the real decoder pipeline, while
everything that can be verified headless lives here — dataset loading, exact /
reference matching, latency + RTF statistics, experiment plans, summary
aggregation, system info and CPU/RAM sampling.

Reference configuration (exactly the quality baseline that scored
95/109 = 87.2% on ``results/real_dataset_2026-09-18_17-16-22``):

    T-one (official ONNX) -> official kenlm.bin -> pyctcdecode CTC Beam Search
    beam_width=200, alpha=0.4, beta=0.9 -> canonical hotwords from streets.txt
    with hotword_weight=10

Performance experiments (threads / concurrency / optional beam sweep) NEVER
change those decoder parameters: only CPU threading and request concurrency
vary, so every run stays comparable with the quality baseline.
"""

from __future__ import annotations

import datetime
import json
import os
import platform
import re
import resource
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from src.t_one_train.decoder_experiment import (  # noqa: E402
    env_versions,
    write_json,
    write_rows_csv,
)
from src.t_one_train.street_forms_experiment import (  # noqa: E402
    build_canonical_index,
    contains_form,
    load_street_names,
    normalize_text,
)

ROOT = Path(__file__).resolve().parent.parent.parent

# --- reference decoder configuration (must never be changed by the benchmark) --
REFERENCE_ALPHA = 0.4
REFERENCE_BETA = 0.9
REFERENCE_BEAM_WIDTH = 200
REFERENCE_HOTWORD_WEIGHT = 10.0

#: the quality result the reference configuration already produced (documented)
QUALITY_BASELINE = {
    "configuration": "canonical@w10",
    "files": 109,
    "reference_found": 95,
    "reference_found_accuracy": 0.8716,
    "metric": "reference token match in the decoder output (street_exact_match of the quality benchmark)",
    "source": "results/street_hotword_experiment_2026-09-18_18-01-51",
}

# --- benchmark geometry -------------------------------------------------------
DEFAULT_DATASET = "results/real_dataset_2026-09-18_17-16-22"
DEFAULT_THREAD_MODES = (1, 2, 4, 6)      # 6 vCPU VPS
DEFAULT_CONCURRENCY_MODES = (2, 4, 6)    # 1 == sequential baseline

RESULTS_PREFIX = "vps_benchmark_"

#: per-file CSV/JSONL columns (one row per file per mode)
MODE_FIELDS = [
    "mode",
    "kind",
    "threads",
    "concurrency",
    "beam_width",
    "index",
    "id",
    "filename",
    "street",
    "reference",
    "hypothesis",
    "audio_duration_seconds",
    "wav_load_seconds",
    "model_inference_seconds",
    "decoder_seconds",
    "total_inference_seconds",
    "rtf",
    "exact_match",
    "reference_found",
    "detected_streets",
    "acoustic_score",
    "combined_score",
    "n_phrases",
    "n_frames",
    "started_at",
    "error",
]


@dataclass(frozen=True)
class BenchmarkMode:
    """One measured configuration: decoder params are fixed, CPU usage varies."""

    kind: str                 # sequential | threads | concurrency | combined | beam_sweep
    threads: int | None       # ORT intra/inter-op threads, None = library default
    concurrency: int          # simultaneously processed utterances
    beam_width: int = REFERENCE_BEAM_WIDTH
    label: str = ""

    @property
    def name(self) -> str:
        """Stable directory / table name."""
        if self.kind == "sequential":
            return "sequential"
        if self.kind == "beam_sweep":
            return f"beam_{self.beam_width}"
        if self.kind == "threads":
            return f"threads_{self.threads}"
        return f"threads_{self.threads}_concurrency_{self.concurrency}"

    @property
    def subdir(self) -> str:
        if self.kind == "sequential":
            return "sequential"
        if self.kind == "beam_sweep":
            return f"beam_sweep/{self.name}"
        if self.kind == "threads":
            return f"threads/{self.name}"
        if self.kind == "concurrency":
            return f"concurrency/{self.name}"
        return f"combined/{self.name}"

    def as_dict(self) -> dict:
        return {
            "mode": self.name,
            "label": self.label or self.name,
            "kind": self.kind,
            "threads": self.threads,
            "concurrency": self.concurrency,
            "beam_width": self.beam_width,
            "dir": self.subdir,
        }


# --- dataset ------------------------------------------------------------------


def load_dataset_records(
    dataset_dir: str | Path, variant: str = "full", limit: int | None = None
) -> list[dict]:
    """Saved records of ``variant`` in manifest order (the quality-benchmark set).

    The dicts are the manifest lines plus ``_wav`` (absolute WAV path), so every
    downstream metric uses the very same ``reference`` that the
    ``street_hotword_experiment`` benchmark used.
    """
    dataset_dir = Path(dataset_dir)
    manifest = dataset_dir / "manifest.jsonl"
    if not manifest.exists():
        raise FileNotFoundError(f"manifest not found: {manifest}")
    records: list[dict] = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("status") != "saved" or rec.get("variant") != variant:
            continue
        audio = rec.get("audio")
        if not audio:
            continue
        wav = dataset_dir / audio
        if not wav.exists():
            raise FileNotFoundError(f"WAV missing for {rec.get('id')}: {wav}")
        rec["_wav"] = wav
        records.append(rec)
    if limit:
        records = records[:limit]
    return records


# --- matching / metrics -------------------------------------------------------


def is_exact_match(hypothesis: str, reference: str) -> bool:
    """Strict normalized equality of the whole decoder output and the reference."""
    return normalize_text(hypothesis) == normalize_text(reference)


def reference_found(hypothesis: str, reference: str) -> bool:
    """Contiguous token match of the reference inside the hypothesis.

    This is the metric of the quality benchmark (``expected_reference_match``),
    i.e. the one that produced 95/109 = 87.2% for canonical@10.
    """
    return contains_form(hypothesis, reference)


def percentile(values: list[float], q: float) -> float | None:
    """Linear-interpolated percentile (q in [0, 100]); None for an empty list."""
    if not values:
        return None
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * (q / 100.0)
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def _is_nan(value) -> bool:
    try:
        return float(value) != float(value)
    except (TypeError, ValueError):
        return True


def distribution(values: list[float]) -> dict:
    """mean / median / p90 / p95 / p99 / min / max / std for a value list."""
    clean = [float(v) for v in values if v is not None and not _is_nan(v)]
    if not clean:
        return {
            "n": 0, "mean": None, "median": None, "p50": None, "p90": None,
            "p95": None, "p99": None, "min": None, "max": None, "std": None,
        }
    mean = sum(clean) / len(clean)
    var = sum((v - mean) ** 2 for v in clean) / len(clean)
    return {
        "n": len(clean),
        "mean": round(mean, 6),
        "median": round(percentile(clean, 50), 6),
        "p50": round(percentile(clean, 50), 6),
        "p90": round(percentile(clean, 90), 6),
        "p95": round(percentile(clean, 95), 6),
        "p99": round(percentile(clean, 99), 6),
        "min": round(min(clean), 6),
        "max": round(max(clean), 6),
        "std": round(var ** 0.5, 6),
    }


def _round(value: float | None, digits: int = 6) -> float | None:
    return None if value is None else round(float(value), digits)
# --- per-file / per-mode records ---------------------------------------------


def build_file_row(
    *,
    mode: BenchmarkMode,
    index: int,
    record: dict,
    hypothesis: str,
    audio_duration_seconds: float,
    wav_load_seconds: float,
    model_inference_seconds: float,
    decoder_seconds: float,
    n_phrases: int,
    n_frames: int,
    acoustic_score: float | None,
    combined_score: float | None,
    detected_streets: list[str],
    started_at: str,
    error: str | None = None,
) -> dict:
    """One measured file: timings, hypothesis and quality verdicts.

    ``total_inference_seconds`` is the whole per-request cost (WAV I/O + acoustic
    model + CTC/KenLM decoding); ``rtf = total_inference_seconds / duration``.
    """
    total = wav_load_seconds + model_inference_seconds + decoder_seconds
    reference = record.get("reference") or ""
    duration = float(audio_duration_seconds) or 0.0
    return {
        "mode": mode.name,
        "kind": mode.kind,
        "threads": mode.threads,
        "concurrency": mode.concurrency,
        "beam_width": mode.beam_width,
        "index": index,
        "id": record.get("id"),
        "filename": Path(record["_wav"]).name if record.get("_wav") else None,
        "street": record.get("street"),
        "reference": reference,
        "hypothesis": hypothesis,
        "audio_duration_seconds": _round(duration, 4),
        "wav_load_seconds": _round(wav_load_seconds),
        "model_inference_seconds": _round(model_inference_seconds),
        "decoder_seconds": _round(decoder_seconds),
        "total_inference_seconds": _round(total),
        "rtf": _round(total / duration, 5) if duration > 0 else None,
        "exact_match": is_exact_match(hypothesis, reference),
        "reference_found": reference_found(hypothesis, reference),
        "detected_streets": "; ".join(detected_streets),
        "acoustic_score": _round(acoustic_score, 4),
        "combined_score": _round(combined_score, 4),
        "n_phrases": n_phrases,
        "n_frames": n_frames,
        "started_at": started_at,
        "error": error,
    }


def summarize_mode(
    *,
    mode: BenchmarkMode,
    rows: list[dict],
    wall_clock_seconds: float,
    session_init_seconds: float | None,
    resources: dict | None,
    warmup_seconds: float | None = None,
    created: str | None = None,
) -> dict:
    """Aggregate per-file rows of one mode into the metrics required by the task."""
    ok = [r for r in rows if not r.get("error")]
    failed = [r for r in rows if r.get("error")]
    latencies = [r["total_inference_seconds"] for r in ok if r.get("total_inference_seconds") is not None]
    rtfs = [r["rtf"] for r in ok if r.get("rtf") is not None]
    durations = [r["audio_duration_seconds"] for r in ok if r.get("audio_duration_seconds")]
    model_times = [r["model_inference_seconds"] for r in ok if r.get("model_inference_seconds") is not None]
    decoder_times = [r["decoder_seconds"] for r in ok if r.get("decoder_seconds") is not None]
    wav_times = [r["wav_load_seconds"] for r in ok if r.get("wav_load_seconds") is not None]
    exact = sum(1 for r in ok if r.get("exact_match"))
    found = sum(1 for r in ok if r.get("reference_found"))

    total_audio = sum(durations)
    total_inference = sum(latencies)
    model_sum = sum(model_times)
    decoder_sum = sum(decoder_times)
    wav_sum = sum(wav_times)
    compute_sum = model_sum + decoder_sum

    return {
        "mode": mode.name,
        "label": mode.label or mode.name,
        "kind": mode.kind,
        "threads": mode.threads,
        "concurrency": mode.concurrency,
        "beam_width": mode.beam_width,
        "created": created or datetime.datetime.now().isoformat(timespec="seconds"),
        "files_total": len(rows),
        "files_ok": len(ok),
        "files_failed": len(failed),
        "failed_ids": [r.get("id") for r in failed],
        "exact_matches": exact,
        "exact_accuracy": round(exact / len(ok), 6) if ok else None,
        "reference_found": found,
        "reference_found_accuracy": round(found / len(ok), 6) if ok else None,
        "latency_seconds": distribution(latencies),
        "rtf": {
            "mean": _round(sum(rtfs) / len(rtfs), 6) if rtfs else None,
            "p50": _round(percentile(rtfs, 50), 6),
            "p95": _round(percentile(rtfs, 95), 6),
            "p99": _round(percentile(rtfs, 99), 6),
            "min": _round(min(rtfs), 6) if rtfs else None,
            "max": _round(max(rtfs), 6) if rtfs else None,
        },
        "components_seconds": {
            "wav_load_total": _round(wav_sum),
            "model_inference_total": _round(model_sum),
            "decoder_total": _round(decoder_sum),
            "compute_total": _round(compute_sum),
            "model_inference_mean": _round(model_sum / len(model_times), 6) if model_times else None,
            "decoder_mean": _round(decoder_sum / len(decoder_times), 6) if decoder_times else None,
            "model_inference_share": _round(model_sum / compute_sum, 5) if compute_sum else None,
            "decoder_share": _round(decoder_sum / compute_sum, 5) if compute_sum else None,
        },
        "audio": {
            "files": len(durations),
            "total_duration_seconds": _round(total_audio, 4),
            "mean_duration_seconds": _round(total_audio / len(durations), 4) if durations else None,
        },
        "totals": {
            "total_inference_seconds": _round(total_inference, 4),
            "wall_clock_seconds": _round(wall_clock_seconds, 4),
            "throughput_files_per_second": _round(len(ok) / wall_clock_seconds, 5) if wall_clock_seconds > 0 else None,
            "throughput_audio_seconds_per_second": _round(total_audio / wall_clock_seconds, 5) if wall_clock_seconds > 0 else None,
            "wall_clock_rtf": _round(wall_clock_seconds / total_audio, 5) if total_audio > 0 else None,
        },
        "resources": resources or {},
        "session_init_seconds": _round(session_init_seconds),
        "warmup_seconds": _round(warmup_seconds),
    }


def determinism_check(rows_by_mode: dict[str, list[dict]], baseline_name: str = "sequential") -> dict:
    """Do all modes decode identically? (threads/concurrency must not change quality)"""
    if baseline_name not in rows_by_mode:
        return {"checked": False, "reason": f"no '{baseline_name}' baseline"}
    baseline = {
        r["id"]: r.get("hypothesis")
        for r in rows_by_mode[baseline_name] if not r.get("error")
    }
    differences: list[dict] = []
    compared = 0
    for name, rows in rows_by_mode.items():
        if name == baseline_name:
            continue
        for r in rows:
            if r.get("error") or r["id"] not in baseline:
                continue
            compared += 1
            if r.get("hypothesis") != baseline[r["id"]]:
                differences.append({
                    "mode": name,
                    "id": r["id"],
                    "baseline": baseline[r["id"]],
                    "hypothesis": r.get("hypothesis"),
                })
    return {
        "checked": True,
        "baseline_mode": baseline_name,
        "compared_rows": compared,
        "identical": not differences,
        "differences_total": len(differences),
        "differences": differences[:20],
    }


def detect_bottleneck(baseline: dict) -> dict:
    """Which part dominates: acoustic model or CTC/KenLM beam search?"""
    comp = baseline.get("components_seconds") or {}
    model = comp.get("model_inference_total") or 0.0
    decoder = comp.get("decoder_total") or 0.0
    total = model + decoder
    if total <= 0:
        return {"verdict": "unknown", "basis": "no component timings"}
    model_share = model / total
    decoder_share = decoder / total
    if model_share >= 0.6:
        verdict = "acoustic_model"
    elif decoder_share >= 0.6:
        verdict = "decoder_beam_search_kenlm"
    else:
        verdict = "mixed"
    return {
        "verdict": verdict,
        "basis": f"sequential baseline ({baseline.get('mode')})",
        "model_inference_seconds": _round(model),
        "decoder_seconds": _round(decoder),
        "model_share": _round(model_share, 5),
        "decoder_share": _round(decoder_share, 5),
    }


# --- experiment plans ---------------------------------------------------------


def build_experiment_plan(
    *,
    thread_modes: tuple[int, ...] = DEFAULT_THREAD_MODES,
    concurrency_modes: tuple[int, ...] = DEFAULT_CONCURRENCY_MODES,
    best_threads: int | None = None,
    beam_sweep: tuple[int, ...] = (),
    with_threads: bool = True,
    with_concurrency: bool = True,
    with_combined: bool = True,
) -> list[BenchmarkMode]:
    """Everything to run, in strict order; the sequential baseline always first.

    ``best_threads`` (the winner of the sequential thread sweep) is unknown
    before the run, so this function is called twice: once for the static phases
    and once more with the measured value to append the production-oriented
    thread+concurrency variants.
    """
    plan: list[BenchmarkMode] = [
        BenchmarkMode(
            kind="sequential", threads=None, concurrency=1,
            label="sequential baseline (threads=library default, concurrency=1)",
        ),
    ]
    if with_threads:
        for t in thread_modes:
            plan.append(BenchmarkMode(
                kind="threads", threads=int(t), concurrency=1,
                label=f"sequential threads={t}",
            ))
    if with_concurrency:
        # one intra-op thread per request: pure request parallelism
        for c in concurrency_modes:
            plan.append(BenchmarkMode(
                kind="concurrency", threads=1, concurrency=int(c),
                label=f"concurrency={c} threads=1",
            ))
    if with_concurrency and best_threads is not None and best_threads > 1:
        for c in concurrency_modes:
            plan.append(BenchmarkMode(
                kind="concurrency", threads=int(best_threads), concurrency=int(c),
                label=f"concurrency={c} threads={best_threads} (best sequential)",
            ))
    if with_combined:
        for c in concurrency_modes:
            plan.append(BenchmarkMode(
                kind="combined", threads=int(c), concurrency=int(c),
                label=f"threads={c} concurrency={c}",
            ))
    for bw in beam_sweep:
        plan.append(BenchmarkMode(
            kind="beam_sweep", threads=None, concurrency=1, beam_width=int(bw),
            label=f"beam_width={bw} (optional quality/performance trade-off)",
        ))
    return dedupe_plan(plan)


def dedupe_plan(plan: list[BenchmarkMode]) -> list[BenchmarkMode]:
    """Drop repeated (threads, concurrency, beam_width) combos, keep the order."""
    seen: set[tuple] = set()
    out: list[BenchmarkMode] = []
    for mode in plan:
        if mode.kind == "sequential":
            out.append(mode)
            continue
        key = (mode.threads, mode.concurrency, mode.beam_width)
        if key in seen:
            continue
        seen.add(key)
        out.append(mode)
    return out


def pick_best_threads(summaries: list[dict]) -> int | None:
    """Lowest mean sequential latency in the thread sweep; ties -> fewer threads."""
    candidates = [
        s for s in summaries
        if s["kind"] in ("threads", "sequential")
        and s["concurrency"] == 1
        and s["beam_width"] == REFERENCE_BEAM_WIDTH
        and (s.get("latency_seconds") or {}).get("mean") is not None
    ]
    if not candidates:
        return None
    best = min(candidates, key=lambda s: (s["latency_seconds"]["mean"], s["threads"] or 0))
    return best["threads"]


# --- results dirs / writers ---------------------------------------------------


def results_dir(base: str | Path) -> Path:
    """Unique ``<base>/vps_benchmark_<YYYY-MM-DD_HH-MM-SS>`` directory."""
    base = Path(base)
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out = base / f"{RESULTS_PREFIX}{stamp}"
    n = 1
    while out.exists():
        n += 1
        out = base / f"{RESULTS_PREFIX}{stamp}-{n}"
    out.mkdir(parents=True)
    return out


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: str | Path, row: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def flatten_rows(rows: list[dict]) -> list[dict]:
    """Per-file rows -> flat CSV rows (lists joined, booleans as YES/NO)."""
    flat: list[dict] = []
    for row in rows:
        flat.append({
            k: ("; ".join(v) if isinstance(v, list)
                else ("YES" if v is True else "NO" if v is False
                      else ("" if v is None else v)))
            for k, v in row.items()
        })
    return flat


class Tee:
    """Duplicate stdout/stderr into ``stdout.log`` of the results directory."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._fh = None

    def __enter__(self) -> "Tee":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8", buffering=1)
        sys.stdout = _TeeStream(sys.__stdout__, self._fh)
        sys.stderr = _TeeStream(sys.__stderr__, self._fh)
        return self

    def __exit__(self, *exc) -> None:
        sys.stdout.flush()
        sys.stderr.flush()
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        if self._fh:
            self._fh.close()
            self._fh = None


class _TeeStream:
    def __init__(self, *streams) -> None:
        self._streams = streams

    def write(self, data: str) -> int:
        for s in self._streams:
            try:
                s.write(data)
            except Exception:  # pragma: no cover - logging must never break the run
                pass
        return len(data)

    def flush(self) -> None:
        for s in self._streams:
            try:
                s.flush()
            except Exception:  # pragma: no cover
                pass

    def isatty(self) -> bool:
        return False

    @property
    def encoding(self) -> str:
        return getattr(self._streams[0], "encoding", "utf-8")

    def fileno(self):  # pragma: no cover - some libraries probe it
        return self._streams[0].fileno()

    def __getattr__(self, item):
        return getattr(self._streams[0], item)


# --- system info --------------------------------------------------------------

_PAGE_SIZE = resource.getpagesize()
_CGROUP_FILES = (
    "/sys/fs/cgroup/memory.max",                      # cgroup v2
    "/sys/fs/cgroup/memory/memory.limit_in_bytes",    # cgroup v1
)


def _read_text(path: str | Path) -> str | None:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def _meminfo() -> dict[str, int]:
    """/proc/meminfo in bytes (empty dict off Linux)."""
    text = _read_text("/proc/meminfo")
    if not text:
        return {}
    out: dict[str, int] = {}
    for line in text.splitlines():
        key, _, rest = line.partition(":")
        parts = rest.split()
        if parts and parts[0].isdigit():
            out[key.strip()] = int(parts[0]) * 1024
    return out


def _physical_cpu_count() -> int | None:
    text = _read_text("/proc/cpuinfo")
    if text:
        pairs = set()
        phys: str | None = None
        core: str | None = None
        for line in text.splitlines() + [""]:
            key, _, value = line.partition(":")
            key, value = key.strip(), value.strip()
            if key == "physical id":
                phys = value
            elif key == "core id":
                core = value
            elif not line and phys is not None and core is not None:
                pairs.add((phys, core))
                phys = core = None
        if pairs:
            return len(pairs)
    try:  # macOS
        out = subprocess.run(["sysctl", "-n", "hw.physicalcpu"], capture_output=True, text=True, timeout=5)
        return int(out.stdout.strip()) or None
    except Exception:  # pragma: no cover
        return None


def _cpuinfo() -> dict:
    """CPU model / counts from /proc/cpuinfo + os (works on macOS too)."""
    logical = os.cpu_count() or 1
    affinity = logical
    if hasattr(os, "sched_getaffinity"):
        try:
            affinity = len(os.sched_getaffinity(0))
        except OSError:  # pragma: no cover - platform dependent
            pass
    model = None
    flags: list[str] = []
    text = _read_text("/proc/cpuinfo")
    if text:
        for line in text.splitlines():
            key, _, value = line.partition(":")
            key, value = key.strip(), value.strip()
            if key in ("model name", "Model", "Hardware", "cpu") and value and not model:
                model = value
            if key == "flags" and not flags:
                flags = value.split()
    if model is None:  # macOS
        try:
            model = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=5,
            ).stdout.strip() or platform.processor() or None
        except Exception:  # pragma: no cover
            model = platform.processor() or None
    if model is None:
        # slim/ARM images and some VPS kernels ship /proc/cpuinfo without a
        # model line — fall back to the architecture string.
        model = platform.machine()
    return {
        "model": model,
        "logical_cpus": logical,
        "affinity_cpus": affinity,
        "physical_cpus": _physical_cpu_count(),
        "architecture": platform.machine(),
        "avx2": "avx2" in flags if flags else None,
        "avx512": any(f.startswith("avx512") for f in flags) if flags else None,
    }


def _os_release() -> dict:
    text = _read_text("/etc/os-release")
    info: dict = {"system": platform.system(), "release": platform.release(), "version": platform.version()}
    if text:
        parsed: dict[str, str] = {}
        for line in text.splitlines():
            key, _, value = line.partition("=")
            if key and value:
                parsed[key.strip()] = value.strip().strip('"')
        info["name"] = parsed.get("NAME")
        info["version_id"] = parsed.get("VERSION_ID")
        info["pretty"] = parsed.get("PRETTY_NAME")
        info["id"] = parsed.get("ID")
    else:
        info["pretty"] = platform.platform()
    return info


def _cgroup_memory_limit() -> int | None:
    for path in _CGROUP_FILES:
        text = _read_text(path)
        if not text:
            continue
        value = text.strip()
        if value == "max":
            return None
        try:
            limit = int(value)
        except ValueError:
            continue
        if 0 < limit < (1 << 62):
            return limit
    return None


def _docker_info() -> dict:
    inside = Path("/.dockerenv").exists() or bool(_read_text("/proc/self/cgroup") or "")
    image_id = os.environ.get("BENCHMARK_IMAGE_ID")
    if not image_id:  # inside the container: our own image is the last layer
        try:
            image_id = (
                _read_text("/etc/hostname") or ""
            ).strip() or None
        except Exception:  # pragma: no cover
            image_id = None
    return {
        "inside_container": inside,
        "docker_version": os.environ.get("DOCKER_VERSION")
        or os.environ.get("BENCHMARK_DOCKER_VERSION"),
        "image": os.environ.get("BENCHMARK_IMAGE")
        or os.environ.get("BENCHMARK_IMAGE_TAG"),
        "image_id": image_id,
        "compose_project": os.environ.get("COMPOSE_PROJECT_NAME"),
    }


def _git_commit(root: str | Path = ROOT) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def system_info(extra: dict | None = None) -> dict:
    """CPU / RAM / OS / Docker / Python / package versions for ONE report."""
    mem = _meminfo()
    info = {
        "generated": datetime.datetime.now().isoformat(timespec="seconds"),
        "hostname": platform.node(),
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "os": _os_release(),
        "cpu": _cpuinfo(),
        "memory": {
            "total_bytes": mem.get("MemTotal"),
            "available_bytes": mem.get("MemAvailable"),
            "total_gb": round(mem["MemTotal"] / 1e9, 2) if mem.get("MemTotal") else None,
            "available_gb": round(mem["MemAvailable"] / 1e9, 2) if mem.get("MemAvailable") else None,
            "cgroup_limit_bytes": _cgroup_memory_limit(),
        },
        "docker": _docker_info(),
        "git_commit": _git_commit(),
        "package_versions": env_versions(),
        "env": {
            k: os.environ.get(k)
            for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                      "HPU_NUM_THREADS", "BENCHMARK_IMAGE", "HF_HOME",
                      "TOKENIZERS_PARALLELISM")
        },
    }
    if extra:
        info.update(extra)
    return info


def render_system_info(info: dict) -> str:
    """Human-readable ``system_info.txt``."""
    cpu = info.get("cpu") or {}
    mem = info.get("memory") or {}
    osr = info.get("os") or {}
    docker = info.get("docker") or {}
    pkg = info.get("package_versions") or {}
    env = info.get("env") or {}
    lines = [
        "T-one VPS benchmark — system information",
        "=" * 44,
        f"generated:      {info.get('generated')}",
        f"hostname:       {info.get('hostname')}",
        f"git commit:     {info.get('git_commit')}",
        "",
        "[OS]",
        f"os:             {osr.get('pretty')}",
        f"id/version:     {osr.get('id')} {osr.get('version_id')}",
        f"kernel:         {osr.get('system')} {osr.get('release')}",
        f"platform:       {info.get('platform')}",
        f"architecture:   {cpu.get('architecture')}",
        "",
        "[CPU]",
        f"model:          {cpu.get('model')}",
        f"logical cpus:   {cpu.get('logical_cpus')}",
        f"affinity cpus:  {cpu.get('affinity_cpus')}",
        f"physical cpus:  {cpu.get('physical_cpus')}",
        f"avx2/avx512:    {cpu.get('avx2')} / {cpu.get('avx512')}",
        "",
        "[RAM]",
        f"total:          {mem.get('total_gb')} GB ({mem.get('total_bytes')} bytes)",
        f"available:      {mem.get('available_gb')} GB",
        f"cgroup limit:   {mem.get('cgroup_limit_bytes')}",
        "",
        "[DOCKER]",
        f"inside:         {docker.get('inside_container')}",
        f"docker version: {docker.get('docker_version')}",
        f"image:          {docker.get('image')}",
        f"image id:       {docker.get('image_id')}",
        "",
        "[PYTHON]",
        f"python:         {info.get('python')} ({info.get('python_executable')})",
    ]
    for name, version in pkg.items():
        lines.append(f"{name + ':':<16}{version}")
    lines += ["", "[THREAD ENV]"]
    for key, value in env.items():
        lines.append(f"{key + ':':<24}{value}")
    return "\n".join(lines) + "\n"


# --- CPU / RAM sampling -------------------------------------------------------


def process_cpu_seconds() -> float:
    """Cumulative process CPU time (user+sys) — used for mean/peak CPU%."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return float(usage.ru_utime + usage.ru_stime)


def process_rss_bytes() -> int | None:
    """Resident set size of this process (Linux /proc, else getrusage)."""
    text = _read_text("/proc/self/statm")
    if text:
        parts = text.split()
        if len(parts) >= 2:
            return int(parts[1]) * _PAGE_SIZE
    usage = resource.getrusage(resource.RUSAGE_SELF)
    if usage.ru_maxrss:
        # Linux: kB, macOS: bytes
        return int(usage.ru_maxrss) * (1024 if sys.platform.startswith("linux") else 1)
    return None


def system_cpu_times() -> tuple[float, float] | None:
    """(total_jiffies, idle_jiffies) of the whole host from /proc/stat (Linux)."""
    text = _read_text("/proc/stat")
    if not text:
        return None
    for line in text.splitlines():
        if line.startswith("cpu "):
            parts = [float(p) for p in line.split()[1:] if p.replace(".", "").isdigit()]
            if len(parts) >= 4:
                total = sum(parts)
                idle = parts[3] + (parts[4] if len(parts) > 4 else 0.0)
                return total, idle
            return None
    return None


class ResourceSampler:
    """Background CPU/RAM sampler: stdlib only, no monitoring stack.

    Process CPU% comes from ``getrusage`` deltas (can exceed 100% with threads),
    host CPU%/RAM from /proc (Linux/VPS). Sampling runs only while one benchmark
    mode executes, so the numbers belong to that mode alone.
    """

    def __init__(self, interval: float = 0.5) -> None:
        self.interval = max(0.05, float(interval))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._samples: list[dict] = []
        self._t_start = 0.0
        self._cpu_start = 0.0

    def _sample(self) -> dict:
        now = time.perf_counter()
        cpu = process_cpu_seconds()
        prev = self._samples[-1] if self._samples else None
        dt = now - prev["_t"] if prev else 0.0
        cpu_pct = ((cpu - prev["_cpu"]) / dt * 100.0) if prev and dt > 0 else None
        sys_times = system_cpu_times()
        sys_cpu_pct = None
        if sys_times and prev and prev.get("_sys_times") and dt > 0:
            dt_total = sys_times[0] - prev["_sys_times"][0]
            dt_idle = sys_times[1] - prev["_sys_times"][1]
            if dt_total > 0:
                sys_cpu_pct = max(0.0, min(100.0, (1.0 - dt_idle / dt_total) * 100.0))
        mem = _meminfo()
        used = None
        if mem.get("MemTotal") and mem.get("MemAvailable"):
            used = mem["MemTotal"] - mem["MemAvailable"]
        return {
            "_t": now,
            "_cpu": cpu,
            "_sys_times": sys_times,
            "wall_seconds": now - self._t_start,
            "process_cpu_seconds": cpu - self._cpu_start,
            "process_cpu_percent": cpu_pct,
            "process_rss_bytes": process_rss_bytes(),
            "system_cpu_percent": sys_cpu_pct,
            "system_ram_used_bytes": used,
            "system_ram_total_bytes": mem.get("MemTotal"),
        }

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                self._samples.append(self._sample())
            except Exception:  # pragma: no cover - sampling must never break the run
                pass

    def start(self) -> "ResourceSampler":
        self._t_start = time.perf_counter()
        self._cpu_start = process_cpu_seconds()
        self._samples = []
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="resource-sampler", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> dict:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=max(2.0, self.interval * 2))
            self._thread = None
        try:
            self._samples.append(self._sample())
        except Exception:  # pragma: no cover
            pass
        return self.aggregate()

    def aggregate(self) -> dict:
        """peak/mean CPU and RAM over the sampled window."""
        samples = list(self._samples)
        if not samples:
            return {"samples": 0}
        wall = samples[-1]["wall_seconds"] or 0.0
        cpu_total = samples[-1]["process_cpu_seconds"]
        cpu_pcts = [s["process_cpu_percent"] for s in samples if s["process_cpu_percent"] is not None]
        sys_pcts = [s["system_cpu_percent"] for s in samples if s.get("system_cpu_percent") is not None]
        rss = [s["process_rss_bytes"] for s in samples if s.get("process_rss_bytes")]
        ram_used = [s["system_ram_used_bytes"] for s in samples if s.get("system_ram_used_bytes")]
        ram_total = next(
            (s["system_ram_total_bytes"] for s in reversed(samples) if s.get("system_ram_total_bytes")),
            None,
        )
        cpus = os.cpu_count() or 1
        mean_cpu_pct = sum(cpu_pcts) / len(cpu_pcts) if cpu_pcts else None
        return {
            "samples": len(samples),
            "window_seconds": round(wall, 4),
            "cpu_seconds": round(cpu_total, 4),
            "cpu_mean_percent": _round(mean_cpu_pct, 3),
            "cpu_peak_percent": _round(max(cpu_pcts), 3) if cpu_pcts else None,
            "cpu_mean_percent_of_total": _round(mean_cpu_pct / cpus, 3) if mean_cpu_pct is not None else None,
            "cpu_load_average_cores": _round(cpu_total / wall, 3) if wall > 0 else None,
            "system_cpu_mean_percent": _round(sum(sys_pcts) / len(sys_pcts), 3) if sys_pcts else None,
            "system_cpu_peak_percent": _round(max(sys_pcts), 3) if sys_pcts else None,
            "ram_rss_peak_mb": _round(max(rss) / 1e6, 2) if rss else None,
            "ram_rss_mean_mb": _round(sum(rss) / len(rss) / 1e6, 2) if rss else None,
            "system_ram_total_mb": _round(ram_total / 1e6, 2) if ram_total else None,
            "system_ram_used_peak_mb": _round(max(ram_used) / 1e6, 2) if ram_used else None,
            "system_ram_used_mean_mb": _round(sum(ram_used) / len(ram_used) / 1e6, 2) if ram_used else None,
            "system_ram_used_peak_percent": (
                _round(max(ram_used) / ram_total * 100, 2) if ram_used and ram_total else None
            ),
        }
# --- summary aggregation ------------------------------------------------------


def build_summary(
    *,
    config: dict,
    system: dict,
    startup: dict,
    modes: list[dict],
    rows_by_mode: dict[str, list[dict]],
    plan: list[dict] | None = None,
    created: str | None = None,
) -> dict:
    """ONE aggregated summary.json for the whole series of experiments."""
    baseline = next((m for m in modes if m["kind"] == "sequential"), modes[0] if modes else {})
    quality = {
        "configuration": (
            f"T-one + official KenLM + beam_width={REFERENCE_BEAM_WIDTH} "
            f"(alpha={REFERENCE_ALPHA}, beta={REFERENCE_BETA}) + canonical hotwords "
            f"@{REFERENCE_HOTWORD_WEIGHT:g}"
        ),
        "files": baseline.get("files_ok"),
        "exact_matches": baseline.get("exact_matches"),
        "exact_accuracy": baseline.get("exact_accuracy"),
        "reference_found": baseline.get("reference_found"),
        "reference_found_accuracy": baseline.get("reference_found_accuracy"),
        "expected_quality_baseline": QUALITY_BASELINE,
        "matches_expected_baseline": (
            baseline.get("reference_found") == QUALITY_BASELINE["reference_found"]
            and baseline.get("files_ok") == QUALITY_BASELINE["files"]
        ),
    }
    usable = [m for m in modes if (m.get("latency_seconds") or {}).get("mean") is not None]
    best_latency = min(usable, key=lambda m: m["latency_seconds"]["mean"]) if usable else None
    best_throughput = max(
        (m for m in modes if (m.get("totals") or {}).get("throughput_files_per_second")),
        key=lambda m: m["totals"]["throughput_files_per_second"],
        default=None,
    )
    return {
        "benchmark": "T-one CPU inference performance benchmark (VPS)",
        "created": created or datetime.datetime.now().isoformat(timespec="seconds"),
        "config": config,
        "system": system,
        "startup": startup,
        "quality": quality,
        "plan": plan or [m.get("mode") for m in modes],
        "modes": modes,
        "best": {
            "lowest_mean_latency": _best_entry(best_latency),
            "highest_throughput": _best_entry(best_throughput),
        },
        "bottleneck": detect_bottleneck(baseline),
        "determinism": determinism_check(rows_by_mode),
        "notes": [
            "Decoder parameters (beam_width, alpha, beta, hotwords, weight) are IDENTICAL in "
            "every mode: only CPU threading and request concurrency vary.",
            "total_inference_seconds = WAV load + acoustic model + CTC/KenLM decoding (per request); "
            "rtf = total_inference_seconds / audio_duration_seconds.",
            "wall_clock_seconds is the batch time of the mode; per-request latency is never mixed with it.",
            "Warmup run(s) are excluded from all statistics; startup time is measured separately.",
        ],
    }


def _best_entry(mode: dict | None) -> dict | None:
    if not mode:
        return None
    return {
        "mode": mode.get("mode"),
        "threads": mode.get("threads"),
        "concurrency": mode.get("concurrency"),
        "mean_latency_seconds": (mode.get("latency_seconds") or {}).get("mean"),
        "mean_rtf": (mode.get("rtf") or {}).get("mean"),
        "throughput_files_per_second": (mode.get("totals") or {}).get("throughput_files_per_second"),
    }


def write_mode_outputs(out_dir: Path, mode: BenchmarkMode, rows: list[dict], summary: dict) -> None:
    """Per-mode ``results.jsonl`` / ``results.csv`` / ``summary.json`` (raw data kept)."""
    mode_dir = out_dir / mode.subdir
    mode_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(mode_dir / "results.jsonl", rows)
    write_rows_csv(mode_dir / "results.csv", MODE_FIELDS, flatten_rows(rows))
    write_json(mode_dir / "summary.json", summary)


def chown_results(path: str | Path, uid: int | None, gid: int | None) -> None:
    """Best-effort: give the host user ownership of files written by the root container."""
    if not uid or os.geteuid() != 0:
        return
    gid = gid if gid is not None else uid
    root = Path(path)
    for target in [root, *root.rglob("*")]:
        try:
            os.chown(target, uid, gid)
        except OSError:  # pragma: no cover - best effort only
            pass


# --- human-readable report ----------------------------------------------------


def _ms(seconds: float | None) -> str:
    return "—" if seconds is None else f"{seconds * 1000:.0f}"


def _num(value, digits: int = 3, dash: str = "—") -> str:
    if value is None:
        return dash
    if isinstance(value, str):
        return value
    return f"{value:.{digits}f}"


def _mode_table(modes: list[dict]) -> list[str]:
    lines = [
        "| Test | Threads | Concurrency | Mean ms | P50 ms | P90 ms | P95 ms | P99 ms | "
        "RTF mean | RTF p95 | Wall s | Files/s | CPU mean % | RAM peak MB |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for m in modes:
        lat = m.get("latency_seconds") or {}
        rtf = m.get("rtf") or {}
        totals = m.get("totals") or {}
        res = m.get("resources") or {}
        lines.append(
            f"| `{m['mode']}` | {m.get('threads') if m.get('threads') else 'default'} "
            f"| {m.get('concurrency')} | {_ms(lat.get('mean'))} | {_ms(lat.get('p50'))} "
            f"| {_ms(lat.get('p90'))} | {_ms(lat.get('p95'))} | {_ms(lat.get('p99'))} "
            f"| {_num(rtf.get('mean'), 3)} | {_num(rtf.get('p95'), 3)} "
            f"| {_num(totals.get('wall_clock_seconds'), 1)} "
            f"| {_num(totals.get('throughput_files_per_second'), 2)} "
            f"| {_num(res.get('cpu_mean_percent'), 1)} "
            f"| {_num(res.get('ram_rss_peak_mb'), 0)} |"
        )
    return lines


def render_summary_md(summary: dict) -> str:
    """Human-readable ``summary.md``: system, quality, performance, bottleneck."""
    config = summary.get("config") or {}
    system = summary.get("system") or {}
    cpu = system.get("cpu") or {}
    mem = system.get("memory") or {}
    osr = system.get("os") or {}
    startup = summary.get("startup") or {}
    quality = summary.get("quality") or {}
    modes = summary.get("modes") or []
    base = next((m for m in modes if m["kind"] == "sequential"), modes[0] if modes else {})
    expected = quality.get("expected_quality_baseline") or {}

    lines: list[str] = [
        "# T-one VPS benchmark — summary",
        "",
        f"- created: `{summary.get('created')}`",
        f"- results dir: `{config.get('results_dir')}`",
        f"- git commit: `{system.get('git_commit')}`",
        "",
        "## System",
        "",
        f"- CPU: **{cpu.get('model')}** — {cpu.get('logical_cpus')} logical / "
        f"{cpu.get('physical_cpus')} physical CPUs (affinity: {cpu.get('affinity_cpus')})",
        f"- RAM: {mem.get('total_gb')} GB total, {mem.get('available_gb')} GB available at start",
        f"- OS: {osr.get('pretty')} ({cpu.get('architecture')})",
        f"- Docker: {system.get('docker', {}).get('docker_version')} "
        f"(image {system.get('docker', {}).get('image')}, id {system.get('docker', {}).get('image_id')})",
        f"- Python: {system.get('python')}; packages: "
        + ", ".join(f"{k} {v}" for k, v in (system.get('package_versions') or {}).items() if v),
        "",
        "## Quality (reference configuration, unchanged)",
        "",
        f"- decoder: `{config.get('decoder')}`",
        f"- beam_width={config.get('beam_width')}, alpha={config.get('alpha')}, "
        f"beta={config.get('beta')}, hotword_weight={config.get('hotword_weight')}",
        f"- hotwords: {config.get('hotword_count')} canonical names from `{config.get('streets_file')}`",
        f"- dataset: `{config.get('dataset_dir')}` ({config.get('files_total')} WAV)",
        "",
        f"- reference found (street_exact_match): **{quality.get('reference_found')}/"
        f"{quality.get('files')} = {_num((quality.get('reference_found_accuracy') or 0) * 100, 1)}%**",
        f"- exact match (whole utterance == reference): {quality.get('exact_matches')}/"
        f"{quality.get('files')} = {_num((quality.get('exact_accuracy') or 0) * 100, 1)}%",
        f"- expected quality baseline: {expected.get('reference_found')}/{expected.get('files')} = "
        f"{_num((expected.get('reference_found_accuracy') or 0) * 100, 1)}% ({expected.get('configuration')})",
        f"- matches expected baseline: **{quality.get('matches_expected_baseline')}**",
        "",
        "## Cold start",
        "",
        f"- import/runtime: {_ms(startup.get('import_seconds'))} ms",
        f"- total environment setup (start → model+KenLM+decoder ready): "
        f"**{_num(startup.get('startup_total_seconds'), 2)} s**",
        f"- model load: {_num(startup.get('model_load_seconds'), 2)} s ({startup.get('model_source')})",
        f"- KenLM + decoder build: {_num(startup.get('decoder_load_seconds'), 2)} s",
        f"- hotwords from streets.txt: {_num(startup.get('hotwords_seconds'), 3)} s",
        f"- first inference (warmup, excluded from statistics): "
        f"{_num(startup.get('first_inference_seconds'), 2)} s",
        "",
        "## Sequential performance",
        "",
        f"Dataset audio: {_num((base.get('audio') or {}).get('total_duration_seconds'), 1)} s "
        f"({_num((base.get('audio') or {}).get('mean_duration_seconds'), 2)} s per file).",
        "",
        *_mode_table(modes),
        "",
    ]

    thread_modes = [m for m in modes if m["kind"] in ("sequential", "threads")]
    concurrency_modes = [m for m in modes if m["kind"] in ("concurrency", "combined")]
    if thread_modes:
        lines += ["### Thread scaling (concurrency=1)", "", *_mode_table(thread_modes), ""]
    if concurrency_modes:
        lines += [
            "### Concurrency",
            "",
            "`Wall s` is the batch time; `Mean/P50/P95` are **per-request latencies**.",
            "",
            *_mode_table(concurrency_modes),
            "",
        ]

    comp = base.get("components_seconds") or {}
    bottleneck = summary.get("bottleneck") or {}
    lines += [
        "## Bottleneck (measured, sequential baseline)",
        "",
        f"- acoustic model (T-one ONNX forward + logprob splitter): "
        f"{_num(comp.get('model_inference_total'), 1)} s "
        f"({_num((comp.get('model_inference_share') or 0) * 100, 1)}% of compute)",
        f"- CTC beam search + KenLM + hotwords (pyctcdecode): "
        f"{_num(comp.get('decoder_total'), 1)} s "
        f"({_num((comp.get('decoder_share') or 0) * 100, 1)}% of compute)",
        f"- WAV I/O: {_num(comp.get('wav_load_total'), 2)} s",
        f"- verdict: **{bottleneck.get('verdict')}** (basis: {bottleneck.get('basis')})",
        "",
    ]

    det = summary.get("determinism") or {}
    lines += [
        "## Correctness of the load tests",
        "",
        f"- hypotheses identical across all modes (threads/concurrency change nothing): "
        f"**{det.get('identical')}** "
        f"({det.get('compared_rows')} rows compared with `{det.get('baseline_mode')}`; "
        f"{det.get('differences_total')} differences)",
        "",
        "## Best configurations",
        "",
    ]
    for key, label in (("lowest_mean_latency", "lowest mean per-request latency"),
                       ("highest_throughput", "highest throughput")):
        entry = (summary.get("best") or {}).get(key) or {}
        if entry:
            lines.append(
                f"- {label}: `{entry.get('mode')}` — mean {_ms(entry.get('mean_latency_seconds'))} ms, "
                f"RTF {_num(entry.get('mean_rtf'), 3)}, "
                f"{_num(entry.get('throughput_files_per_second'), 2)} files/s"
            )
    lines += ["", "## Notes", ""]
    for note in summary.get("notes") or []:
        lines.append(f"- {note}")
    lines += [
        "- The benchmark does not change normalization, audio preprocessing, sample rate, "
        "channel handling, decoder labels, KenLM or beam parameters.",
        "- `total_inference_seconds` includes WAV loading; `components_seconds.compute_total` "
        "in each per-mode `summary.json` is model + decoder only.",
        "",
        "## Artifacts",
        "",
        "- `summary.json` — aggregated metrics for every mode",
        "- `summary.md` — this report",
        "- `all_results.json` / `all_results.csv` — one record per file per mode",
        "- `config.json` — reproducibility config (dataset, streets, decoder params, threads, concurrency)",
        "- `system_info.txt` — CPU/RAM/OS/Docker/Python/package versions",
        "- `stdout.log` — full console output of the run",
        "- `sequential/`, `threads/`, `concurrency/`, `combined/` — raw per-mode results",
        "",
    ]
    return "\n".join(lines)