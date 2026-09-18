"""Pure helpers for the T-one decoder experiment: run dirs, audio I/O, storage.

This module deliberately has no tone/pyctcdecode imports, so tests for the
experiment's own logic run without the decoders dependency group installed.
"""

from __future__ import annotations

import csv
import datetime
import importlib.metadata
import json
import platform
import re
import sys
import wave
from pathlib import Path

import numpy as np
import soundfile as sf

from src.t_one_train.street_forms_experiment import (  # noqa: E402
    EXPECTED_KEPT,
    HARMED,
    HELPED,
    NEITHER,
    UNCHANGED,
    config_label,
    normalize_tokens,
    partial_candidates,
    tokens_match,
)

TARGET_RATE = 8_000
MIC_CONFIG_WEIGHTS = (10.0, 15.0)  # canonical@10 and canonical@15 only
RUN_DIR_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}(-\d+)?$")


def run_dir(base: str | Path) -> Path:
    """Create a unique results dir <base>/<YYYY-MM-DD_HH-MM-SS>; suffix -2, -3..."""
    base = Path(base)
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    d = base / stamp
    n = 1
    while d.exists():
        n += 1
        d = base / f"{stamp}-{n}"
    d.mkdir(parents=True)
    return d


def save_wav(path: str | Path, pcm16: np.ndarray, rate: int) -> None:
    """Save mono PCM16 8 kHz WAV (the format T-one expects)."""
    path = Path(path)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(rate)
        f.writeframes(pcm16.astype(np.int16).tobytes())


def load_wav_any(path: str | Path) -> tuple[np.ndarray, int]:
    """Load any WAV file; convert to mono PCM16 8 kHz, resampling only if needed."""
    audio, sr = sf.read(str(path), dtype="int16")
    if audio.ndim > 1:
        audio = audio.mean(axis=1).astype(np.int16)
    if sr != TARGET_RATE:
        audio = _resample(audio.astype(np.float32) / 32768.0, sr)
        audio = np.clip(audio * 32767.0, -32768, 32767).astype(np.int16)
        sr = TARGET_RATE
    return audio.astype(np.int32), sr


def _resample(x: np.ndarray, orig_rate: int) -> np.ndarray:
    if orig_rate == TARGET_RATE:
        return x
    from math import gcd

    from scipy.signal import resample_poly

    g = gcd(int(orig_rate), TARGET_RATE)
    return resample_poly(x, TARGET_RATE // g, orig_rate // g)


def write_json(path: str | Path, data: dict) -> None:
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def build_hotwords_json(hotword_audit: list, streets_file: str) -> dict:
    """Audit structure for hotwords.json (source -> prepared hotword)."""
    return {
        "streets_file": streets_file,
        "total_lines": len(hotword_audit),
        "hotwords_used": [e.hotword for e in hotword_audit if not e.excluded],
        "entries": [
            {
                "source": e.source,
                "hotword": e.hotword or None,
                "excluded": e.excluded,
                "reason": e.reason,
            }
            for e in hotword_audit
        ],
    }


def build_mic_config_plan(weights=MIC_CONFIG_WEIGHTS) -> list[tuple[str, str, float | None]]:
    """The four real-voice configs: greedy, beam (no hotwords), canonical@w1, canonical@w2.

    Returns ``[(label, kind, weight)]`` where ``label`` is the stable id used in
    JSON/CSV, ``kind`` selects the decode call and ``weight`` is the pyctcdecode
    ``hotword_weight`` (``None`` for greedy/beam). No other canonical weight
    (3/5/7/20) and no experimental additional forms are ever produced here.
    """
    plan: list[tuple[str, str, float | None]] = [
        ("greedy", "greedy", None),
        ("beam_no_hotwords", "beam_no_hotwords", None),
    ]
    plan += [(config_label("canonical", float(w)), "canonical", float(w)) for w in weights]
    return plan


def mic_display_label(label: str) -> str:
    """Human label for the interactive output: Greedy / Beam / Canonical @10 ..."""
    if label == "greedy":
        return "Greedy"
    if label == "beam_no_hotwords":
        return "Beam"
    if label.startswith("canonical@w"):
        return f"Canonical @{label.removeprefix('canonical@w')}"
    return label


def mic_config_dicts(plan: list[tuple[str, str, float | None]], hotword_count: int) -> list[dict]:
    """Plan tuples -> per-config dicts used in JSON records and CSV rows."""
    return [
        {
            "label": label,
            "display": mic_display_label(label),
            "kind": kind,
            "weight": weight,
            "hotword_count": hotword_count if kind == "canonical" else 0,
        }
        for label, kind, weight in plan
    ]


def _tolerant_contains(text: str, tokens: tuple[str, ...]) -> bool:
    """Inflection-tolerant contiguous match of ``tokens`` inside ``text``.

    Real speech inflects street names ('на Лесной' for 'Лесная'); the decoder
    output must not fail the expected-street check because of a declension.
    """
    if not tokens:
        return False
    words = normalize_tokens(text)
    n = len(tokens)
    if len(words) < n:
        return False
    return any(
        all(tokens_match(w, e) for w, e in zip(words[i : i + n], tokens))
        for i in range(len(words) - n + 1)
    )


def _tolerant_detect(text: str, street_names: list[str]) -> list[str]:
    """Streets whose (inflected) full token sequence is present in ``text``."""
    return sorted(
        s for s in street_names
        if _tolerant_contains(text, normalize_tokens(s))
    )


def evaluate_mic_texts(
    texts: dict[str, str],
    plan: list[tuple[str, str, float | None]],
    expected_street: str,
    reference: str,
    canonical_index,
    additional_index,
    combined_index,
    street_names: list[str],
) -> dict[str, dict]:
    """Score every config against the beam-no-hotwords baseline + expected street.

    Same output schema as the offline ``classify_change`` (so records stay
    comparable with dataset runs) but with real-voice semantics: the expected
    street may be spoken in any inflection ('на Лесной' ≈ 'Лесная'), and a
    hotword-driven switch to a DIFFERENT known street counts as HARMED
    (переубеждение hotwords), not NEITHER. ``expected_street``/``reference``
    come from the resolved operator input and are never passed to the decoders.
    """
    baseline = texts.get("beam_no_hotwords", "")
    greedy_text = texts.get("greedy", "")
    baseline_tokens = set(normalize_tokens(baseline)) | set(normalize_tokens(greedy_text))
    exp_tokens = normalize_tokens(expected_street) if expected_street else ()
    detected_baseline = _tolerant_detect(baseline, street_names)

    out: dict[str, dict] = {}
    for label, kind, _weight in plan:
        cfg_text = texts.get(label, "")
        # "changed" is a hotword/beam effect: the greedy row is never "changed"
        changed = cfg_text != baseline and kind != "greedy"
        expected_before = _tolerant_contains(baseline, exp_tokens)
        expected_after = _tolerant_contains(cfg_text, exp_tokens)
        detected = _tolerant_detect(cfg_text, street_names)
        other = sorted(s for s in detected if s != expected_street)
        unsupported = [
            s for s in other
            if not any(t in baseline_tokens for t in normalize_tokens(s))
        ]
        if not changed:
            category = UNCHANGED
        elif expected_before and not expected_after:
            category = HARMED
        elif expected_after and not expected_before:
            category = HELPED
        elif expected_after:
            category = EXPECTED_KEPT
        elif other:
            # hotwords convinced the decoder to output a different known street
            category = HARMED
        else:
            category = NEITHER
        partials = [] if expected_after else partial_candidates(cfg_text, street_names)
        review_reasons: list[str] = []
        if category == NEITHER:
            review_reasons.append("changed but expected street absent in baseline and config")
        if unsupported:
            review_reasons.append("unsupported street introduced: " + ", ".join(unsupported))
        if partials:
            review_reasons.append("partial street-word match (ambiguous): " + ", ".join(partials))
        out[label] = {
            "text": cfg_text,
            "found": expected_after if expected_street else None,
            "expected_reference_match": expected_after,
            "expected_reference_match_baseline": expected_before,
            "changed_from_baseline": changed,
            "classification": category,
            "expected_street": expected_street,
            "detected_canonical": detected,
            "detected_additional_forms": [],
            "detected_combined": detected,
            "other_streets_detected": other,
            "additionally_detected_vs_baseline": sorted(set(detected) - set(detected_baseline)),
            "unsupported_street_introduced": unsupported,
            "partial_candidates": partials,
            "no_street_detected": not detected and not partials,
            "requires_manual_review": bool(review_reasons),
            "review_reasons": review_reasons,
        }
    return out


def build_mic_phrase_record(
    *,
    phrase_id: str,
    source: str,
    wav_name: str,
    expected_street: str | None,
    expected_street_input: str | None,
    evaluated: bool,
    configs: list[dict],
    decoded: dict[str, str],
    evaluation: dict[str, dict],
    sample_rate: int,
    duration_seconds: float,
    error: str | None = None,
) -> dict:
    """One phrase of the real-voice A/B/C/D test (JSON + results.jsonl payload).

    ``decoded`` maps config label -> text; ``evaluation`` maps config label -> the
    ``classify_change`` dict computed against the beam baseline. When the expected
    street is unknown (``evaluated=False``) the expected-dependent fields are
    ``None``/``False`` while the detected-street fields stay valid.
    """
    evaluations: dict[str, dict] = {}
    for cfg in configs:
        label = cfg["label"]
        info = dict(evaluation.get(label) or {})
        evaluations[label] = {
            "detected_streets": list(info.get("detected_canonical") or []),
            "other_streets": list(info.get("other_streets_detected") or []),
            "unsupported_streets": list(info.get("unsupported_street_introduced") or []),
            "changed_from_beam": bool(info.get("changed_from_baseline")),
            "found": bool(info.get("expected_reference_match")) if evaluated else None,
            "classification": info.get("classification") if evaluated else None,
            "helped": (info.get("classification") == HELPED) if evaluated else False,
            "harmed": (info.get("classification") == HARMED) if evaluated else False,
            "requires_manual_review": bool(info.get("requires_manual_review")) if evaluated else False,
            "review_reasons": list(info.get("review_reasons") or []) if evaluated else [],
        }
    rec = {
        "id": phrase_id,
        "source": source,
        "audio": wav_name,
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "sample_rate": sample_rate,
        "duration_seconds": round(duration_seconds, 3),
        "expected_street_input": expected_street_input or None,
        "expected_street": expected_street or None,
        "evaluated": bool(evaluated),
        "configs": [dict(c) for c in configs],
        "decoded": dict(decoded),
        "evaluation": evaluations,
    }
    if error:
        rec["error"] = error
    return rec


def append_results_jsonl(out_dir: Path, rec: dict) -> None:
    with (out_dir / "results.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


MIC_RESULT_FIELDS = [
    "phrase_id", "timestamp", "audio", "expected_street", "expected_street_input",
    "configuration", "kind", "weight", "hotword_count", "text", "found",
    "changed_from_beam", "classification", "helped", "harmed", "detected_streets",
    "other_streets", "unsupported_streets", "requires_manual_review", "duration_seconds",
]

SESSION_FIELDS = ["configuration", "kind", "weight", "correct", "helped", "harmed", "evaluated"]


def _yesno(value: bool | None) -> str:
    if value is None:
        return ""
    return "YES" if value else "NO"


def mic_config_rows(rec: dict) -> list[dict]:
    """Flatten one phrase record into one CSV row per decoder configuration."""
    rows: list[dict] = []
    for cfg in rec["configs"]:
        label = cfg["label"]
        ev = rec["evaluation"].get(label) or {}
        rows.append(
            {
                "phrase_id": rec["id"],
                "timestamp": rec["timestamp"],
                "audio": rec["audio"],
                "expected_street": rec["expected_street"] or "",
                "expected_street_input": rec["expected_street_input"] or "",
                "configuration": cfg["display"],
                "kind": cfg["kind"],
                "weight": "" if cfg["weight"] is None else f"{cfg['weight']:g}",
                "hotword_count": cfg.get("hotword_count", 0),
                "text": rec["decoded"].get(label, ""),
                "found": "" if ev.get("found") is None else ("FOUND" if ev["found"] else "NOT FOUND"),
                "changed_from_beam": _yesno(ev.get("changed_from_beam")),
                "classification": ev.get("classification") or "",
                "helped": _yesno(ev.get("helped")),
                "harmed": _yesno(ev.get("harmed")),
                "detected_streets": "; ".join(ev.get("detected_streets") or []),
                "other_streets": "; ".join(ev.get("other_streets") or []),
                "unsupported_streets": "; ".join(ev.get("unsupported_streets") or []),
                "requires_manual_review": _yesno(ev.get("requires_manual_review")),
                "duration_seconds": rec["duration_seconds"],
            }
        )
    return rows


def build_session_summary(rows: list[dict], configs: list[dict]) -> dict:
    """Aggregate correctness/helped/harmed over records that have an expected street.

    Only really recorded and scored phrases are counted; every number is derived
    from the per-phrase evaluations, nothing is estimated.
    """
    evaluated = [r for r in rows if r.get("evaluated")]
    per_config: list[dict] = []
    for cfg in configs:
        label = cfg["label"]
        correct = helped = harmed = 0
        for rec in evaluated:
            ev = rec["evaluation"].get(label) or {}
            correct += int(bool(ev.get("found")))
            helped += int(bool(ev.get("helped")))
            harmed += int(bool(ev.get("harmed")))
        per_config.append(
            {
                "label": label,
                "display": cfg["display"],
                "kind": cfg["kind"],
                "weight": cfg["weight"],
                "correct": correct,
                "helped": helped,
                "harmed": harmed,
                "evaluated": len(evaluated),
            }
        )
    return {"tests": len(rows), "evaluated": len(evaluated), "configs": per_config}


def session_summary_rows(summary: dict) -> list[dict]:
    return [
        {
            "configuration": row["display"],
            "kind": row["kind"],
            "weight": "" if row["weight"] is None else f"{row['weight']:g}",
            "correct": row["correct"],
            "helped": row["helped"],
            "harmed": row["harmed"],
            "evaluated": row["evaluated"],
        }
        for row in summary["configs"]
    ]


def write_rows_csv(path: str | Path, fieldnames: list[str], rows: list[dict]) -> None:
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def device_info() -> str:
    try:
        import onnxruntime as ort

        return ",".join(ort.get_available_providers())
    except Exception:
        return "unknown"


def env_versions() -> dict:
    """Best-effort environment versions for run.json."""

    def ver(dist: str) -> str | None:
        try:
            return importlib.metadata.version(dist)
        except Exception:
            return None

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": ver("torch"),
        "tone": ver("tone"),
        "pyctcdecode": ver("pyctcdecode"),
        "kenlm": ver("kenlm"),
        "onnxruntime": ver("onnxruntime"),
        "sounddevice": ver("sounddevice"),
        "onnxruntime_providers": device_info(),
        "cpu_count": __import__("os").cpu_count(),
    }
