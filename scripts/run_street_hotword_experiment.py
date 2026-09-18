"""Reproducible street hotword-forms experiment on the real 109-record dataset.

Research-only: no training, no changes to production hotwords/decoder/weights.
The T-one acoustic model runs ONCE per WAV (log-probabilities cached to an npz),
then the SAME logprobs are decoded by every configuration:

    A. greedy
    B. beam + KenLM without hotwords (the true baseline)
    C. beam + KenLM + canonical hotwords        (streets.txt)
    D. beam + KenLM + canonical + experimental forms
       (src/t_one_train/street_hotword_forms_experiment.json)

for a set of hotword weights. Beam width is identical everywhere.

There is NO ground-truth transcript for these recordings, so WER/CER are NOT
computed: the metric is whether the decoder output contains the dataset
reference (the street actually spoken) and how each config differs from B.

Usage:
    uv run --group decoders python scripts/run_street_hotword_experiment.py
    uv run --group decoders python scripts/run_street_hotword_experiment.py \
        --dataset-dir results/real_dataset_2026-09-18_17-16-22 --weights 1,3,5,7,10
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.t_one_train.decoder_experiment import env_versions, load_wav_any  # noqa: E402
from src.t_one_train.hotwords import load_streets  # noqa: E402
from src.t_one_train.street_forms_experiment import (  # noqa: E402
    build_additional_index,
    build_canonical_hotwords,
    build_canonical_index,
    build_combined_index,
    build_config_plan,
    build_experiment_config,
    build_experiment_hotwords,
    classify_change,
    config_label,
    load_experiment_forms,
    load_street_names,
    validate_experiment_forms,
)

DEFAULT_DATASET = ROOT / "results" / "real_dataset_2026-09-18_17-16-22"
DEFAULT_STREETS = ROOT / "streets.txt"
BEAM_WIDTH = 200
DECODER_PARAMS = {"alpha": 0.4, "beta": 0.9}

SUMMARY_FIELDS = [
    "configuration",
    "kind",
    "weight",
    "hotword_count",
    "n",
    "street_exact_match",
    "street_exact_match_rate",
    "changed_from_baseline",
    "helped",
    "harmed",
    "expected_kept",
    "neither",
    "unchanged",
    "no_street_detected",
    "other_street_records",
    "unsupported_street_introduced",
    "requires_manual_review",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="run_street_hotword_experiment",
        description="Compare greedy / beam / beam+hotwords / beam+hotwords+extra forms on the real dataset.",
    )
    p.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET,
                   help="real dataset dir with manifest.jsonl and audio/")
    p.add_argument("--streets", type=Path, default=DEFAULT_STREETS,
                   help="canonical hotword source (production streets.txt)")
    p.add_argument("--forms", type=Path, default=None,
                   help="experimental additional-forms dictionary (default: package JSON)")
    p.add_argument("--out-base", type=Path, default=ROOT / "results")
    p.add_argument("--weights", default="1,3,5,7,10")
    p.add_argument("--beam-width", type=int, default=BEAM_WIDTH)
    p.add_argument("--limit", type=int, default=None, help="process only the first N records")
    p.add_argument("--variant", default="full", help="dataset variant to evaluate (default: full)")
    p.add_argument("--cache", type=Path, default=None,
                   help="reuse an existing logprobs_cache.npz instead of running the acoustic model")
    p.add_argument("--kenlm-path", type=Path, default=None)
    p.add_argument("--model-path", type=Path, default=None)
    return p.parse_args(argv)


def parse_weights(spec: str) -> list[float]:
    weights = [float(x) for x in spec.split(",") if x.strip()]
    if not weights:
        raise SystemExit("--weights must contain at least one value")
    return weights


def load_records(dataset_dir: Path, variant: str, limit: int | None = None) -> list[dict]:
    """Saved records of the requested variant, in manifest order."""
    manifest = dataset_dir / "manifest.jsonl"
    if not manifest.exists():
        raise SystemExit(f"manifest not found: {manifest}")
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
            raise SystemExit(f"WAV missing for {rec.get('id')}: {wav}")
        rec["_wav"] = wav
        records.append(rec)
    if limit:
        records = records[:limit]
    return records


def make_out_dir(base: Path) -> Path:
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out = base / f"street_hotword_experiment_{stamp}"
    n = 1
    while out.exists():
        n += 1
        out = base / f"street_hotword_experiment_{stamp}-{n}"
    out.mkdir(parents=True)
    return out


def run_acoustic(records: list[dict], out_dir: Path, model_path: str | None) -> dict:
    """Run the acoustic model ONCE per WAV and cache phrase log-probs to npz.

    Returns {"arrays": {record_id: np.ndarray}, "processed": [record...],
             "skipped_no_speech": [record_id...], "forwards": int}.
    """
    from src.t_one_train.test_decoders_core import collect_phrase_logprobs, load_acoustic_model

    model = load_acoustic_model(model_path)
    arrays: dict[str, np.ndarray] = {}
    processed: list[dict] = []
    skipped: list[str] = []
    for idx, rec in enumerate(records, 1):
        pcm, _sr = load_wav_any(rec["_wav"])
        phrases, _frames = collect_phrase_logprobs(model, pcm.astype(np.int32))
        if not phrases:
            skipped.append(rec["id"])
            print(f"[{idx}/{len(records)}] {rec['id']}: no speech (splitter) -> skipped")
            continue
        arrays[rec["id"]] = np.concatenate(phrases, axis=0)
        processed.append(rec)
        print(f"[{idx}/{len(records)}] {rec['id']}: {rec['street']} logprobs={arrays[rec['id']].shape}",
              flush=True)
    np.savez_compressed(out_dir / "logprobs_cache.npz", **arrays)
    return {"arrays": arrays, "processed": processed, "skipped_no_speech": skipped,
            "forwards": len(processed)}


def load_acoustic_cache(cache_path: Path) -> dict:
    data = np.load(cache_path, allow_pickle=False)
    arrays = {k: data[k] for k in data.files}
    return {"arrays": arrays, "forwards": 0, "cache_path": str(cache_path)}


def decode_all(records: list[dict], arrays: dict, plan: list, beam_width: int,
               canonical_hw: list[str], plus_forms_hw: list[str],
               kenlm_path: str | None = None) -> tuple[dict, int]:
    """Decode every config on the SAME cached logprobs. Returns (texts, decoder_runs)."""
    from tone.decoder import GreedyCTCDecoder

    from src.t_one_train.test_decoders_core import load_beam_decoder

    greedy = GreedyCTCDecoder()
    beam = load_beam_decoder(kenlm_path)
    hotwords_by_kind = {"canonical": canonical_hw, "canonical_plus_forms": plus_forms_hw}

    texts: dict[str, dict[str, str]] = {}
    runs = 0
    for kind, weight in plan:
        label = config_label(kind, weight)
        per_record: dict[str, str] = {}
        for rec in records:
            lp = arrays[rec["id"]]
            if kind == "greedy":
                text = greedy.forward(lp)
            elif kind == "beam_no_hotwords":
                text = beam.decode(lp, beam_width=beam_width)
            else:
                text = beam.decode(lp, beam_width=beam_width,
                                   hotwords=hotwords_by_kind[kind], hotword_weight=weight)
            per_record[rec["id"]] = text
            runs += 1
        texts[label] = per_record
        print(f"  decoded {label}", flush=True)
    return texts, runs


def classify_all(records: list[dict], texts: dict, plan: list, indexes: dict,
                 street_names: list[str]) -> tuple[list[dict], list[dict]]:
    """Classify every (record, config) output; aggregate per-config counters."""
    baseline_label = "beam_no_hotwords"
    rows: list[dict] = []
    summary: list[dict] = []
    canonical_index = indexes["canonical"]
    additional_index = indexes["additional"]
    combined_index = indexes["combined"]

    for kind, weight in plan:
        label = config_label(kind, weight)
        counters = {
            "configuration": label, "kind": kind, "weight": weight, "n": len(records),
            "street_exact_match": 0, "changed_from_baseline": 0, "helped": 0, "harmed": 0,
            "expected_kept": 0, "neither": 0, "unchanged": 0, "no_street_detected": 0,
            "other_street_records": 0, "unsupported_street_introduced": 0,
            "requires_manual_review": 0,
        }
        for rec in records:
            text = texts[label][rec["id"]]
            info = classify_change(
                baseline_text=texts[baseline_label][rec["id"]],
                config_text=text,
                reference=rec["reference"],
                expected_street=rec["street"],
                canonical_index=canonical_index,
                additional_index=additional_index,
                combined_index=combined_index,
                greedy_text=texts["greedy"][rec["id"]],
                street_names=street_names,
            )
            rows.append({
                "id": rec["id"], "street_index": rec["street_index"],
                "expected_street": rec["street"], "reference": rec["reference"],
                "configuration": label, "kind": kind, "weight": weight,
                "baseline_text": texts[baseline_label][rec["id"]],
                "greedy_text": texts["greedy"][rec["id"]],
                **info,
            })
            counters["street_exact_match"] += int(info["expected_reference_match"])
            counters["changed_from_baseline"] += int(info["changed_from_baseline"])
            counters[info["classification"].lower()] += 1
            counters["no_street_detected"] += int(info["no_street_detected"])
            counters["other_street_records"] += int(bool(info["other_streets_detected"]))
            counters["unsupported_street_introduced"] += len(info["unsupported_street_introduced"])
            counters["requires_manual_review"] += int(info["requires_manual_review"])
        summary.append(counters)
    return rows, summary


def write_summary(out_dir: Path, config: dict, summary: list[dict], counts: dict,
                  canonical_hw: list[str], plus_forms_hw: list[str]) -> dict:
    per_config = []
    for row in summary:
        kind = row["kind"]
        hotword_count = 0
        if kind == "canonical":
            hotword_count = len(canonical_hw)
        elif kind == "canonical_plus_forms":
            hotword_count = len(plus_forms_hw)
        item = dict(row)
        item["hotword_count"] = hotword_count
        item["street_exact_match_rate"] = (
            round(row["street_exact_match"] / row["n"], 4) if row["n"] else 0.0
        )
        per_config.append(item)

    hotword_rows = [r for r in per_config if r["kind"] in ("canonical", "canonical_plus_forms")]
    best = max(hotword_rows, key=lambda r: (r["street_exact_match"], -r["harmed"])) if hotword_rows else None
    payload = {
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "config": config,
        "counts": counts,
        "configs": per_config,
        "best_hotword_config_by_street_exact_match": (
            {"configuration": best["configuration"], "street_exact_match": best["street_exact_match"],
             "street_exact_match_rate": best["street_exact_match_rate"], "harmed": best["harmed"],
             "helped": best["helped"]} if best else None
        ),
        "notes": [
            "street_exact_match compares the decoder output with the dataset reference (street spoken).",
            "WER/CER are NOT computed: no ground-truth transcript exists for these recordings.",
            "classification is relative to the beam+KenLM no-hotword baseline.",
            "requires_manual_review marks records where the automatic metric cannot decide.",
        ],
    }
    (out_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (out_dir / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in per_config:
            writer.writerow({k: row.get(k, "") for k in SUMMARY_FIELDS})
    return payload


PER_FILE_FIELDS = [
    "id", "street_index", "expected_street", "reference", "configuration", "kind", "weight",
    "text", "baseline_text", "greedy_text", "expected_reference_match", "changed_from_baseline",
    "classification", "detected_combined", "other_streets_detected",
    "unsupported_street_introduced", "no_street_detected", "requires_manual_review",
    "review_reasons",
]


def _join(value) -> str:
    if isinstance(value, (list, tuple)):
        return "; ".join(str(v) for v in value)
    return "" if value is None else str(value)


def write_per_file(out_dir: Path, rows: list[dict]) -> None:
    with (out_dir / "per_file_results.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (out_dir / "per_file_results.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PER_FILE_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _join(row.get(k)) for k in PER_FILE_FIELDS})


def build_examples(rows: list[dict], weights: list[float], targeted: list[str]) -> dict:
    """Example tables: forms vs beam baseline, forms vs canonical, targeted streets."""
    by_label: dict[str, dict[str, dict]] = {}
    for row in rows:
        by_label.setdefault(row["configuration"], {})[row["id"]] = row

    def brief(row: dict) -> dict:
        return {
            "id": row["id"], "expected_street": row["expected_street"],
            "baseline": row["baseline_text"], "text": row["text"],
            "classification": row["classification"],
            "detected": row["detected_combined"],
            "review_reasons": row["review_reasons"],
        }

    vs_beam: dict[str, dict] = {}
    for label, per_record in by_label.items():
        helped = [brief(r) for r in per_record.values() if r["classification"] == "HELPED"]
        harmed = [brief(r) for r in per_record.values() if r["classification"] == "HARMED"]
        neither = [brief(r) for r in per_record.values() if r["classification"] == "NEITHER"]
        vs_beam[label] = {
            "helped": helped, "harmed": harmed, "neither": neither,
            "counts": {"helped": len(helped), "harmed": len(harmed), "neither": len(neither)},
        }

    forms_effect: dict[str, dict] = {}
    for w in weights:
        canon = by_label.get(f"canonical@w{w:g}", {})
        plus = by_label.get(f"canonical_plus_forms@w{w:g}", {})
        fixed, broken, identical = [], [], 0
        for rid, c in canon.items():
            p = plus.get(rid)
            if p is None:
                continue
            if c["text"] == p["text"]:
                identical += 1
            elif not c["expected_reference_match"] and p["expected_reference_match"]:
                fixed.append(brief(p))
            elif c["expected_reference_match"] and not p["expected_reference_match"]:
                broken.append(brief(p))
        forms_effect[f"w{w:g}"] = {
            "identical_text": identical, "fixed_by_forms": fixed, "broken_by_forms": broken,
            "counts": {"identical_text": identical, "fixed_by_forms": len(fixed),
                       "broken_by_forms": len(broken)},
        }

    targeted_rows: dict[str, dict] = {}
    for street in targeted:
        entry: dict[str, list] = {}
        for label, per_record in by_label.items():
            hits = [brief(r) for r in per_record.values() if r["expected_street"] == street]
            if hits:
                entry[label] = hits
        targeted_rows[street] = entry
    return {"vs_beam_baseline": vs_beam, "forms_effect_vs_canonical": forms_effect,
            "targeted_streets": targeted_rows}
def write_readme(out_dir: Path, payload: dict, examples: dict, weights: list[float],
                 counts: dict, canonical_hw: list[str], plus_forms_hw: list[str],
                 exp_forms: dict, targeted: list[str]) -> None:
    cfg = payload["config"]
    lines: list[str] = []
    add = lines.append
    add("# Street hotword-forms experiment")
    add("")
    add(f"- Created: {payload['created']}")
    add(f"- Dataset: `{cfg['dataset_dir']}` (variant `full`)")
    add(f"- Records processed: {counts['records_processed']} / {counts['records_total']}"
        + (f"; skipped (no speech in splitter): {len(counts['skipped_no_speech'])}"
           if counts["skipped_no_speech"] else ""))
    add(f"- Acoustic forwards: {counts['acoustic_forwards']} "
        "(one per WAV, cached to `logprobs_cache.npz`)")
    add(f"- Decoder runs: {counts['decoder_runs']} (configs x records, same cached logprobs)")
    add(f"- Beam: beam_width={cfg['beam_width']}, alpha={cfg['decoder_params']['alpha']}, "
        f"beta={cfg['decoder_params']['beta']}, official `kenlm.bin`")
    add(f"- Hotword weights: {weights}")
    add(f"- Canonical hotwords: {len(canonical_hw)} "
        f"(from `{cfg['streets_file']}` via hotwords.load_streets)")
    add(f"- Canonical + experimental forms: {len(plus_forms_hw)} hotwords "
        f"({len(plus_forms_hw) - len(canonical_hw)} extra forms from `{exp_forms['path']}`)")
    add(f"- Extra forms: {sorted(f for f in plus_forms_hw if f not in set(canonical_hw))}")
    add("")
    add("## Configurations")
    add("")
    add("- `greedy` — CTC greedy, no LM, no hotwords.")
    add("- `beam_no_hotwords` — BeamSearch + KenLM, TRUE baseline (no HotwordScorer at all).")
    add("- `canonical@wN` — beam + KenLM + canonical hotwords (streets.txt), weight N.")
    add("- `canonical_plus_forms@wN` — beam + KenLM + canonical hotwords + experimental forms, weight N.")
    add("")
    add("## Metrics (NOT WER/CER)")
    add("")
    add("There is **no ground-truth transcript** for these recordings, so WER/CER are NOT computed.")
    add("The objective anchor is the dataset reference (the street actually spoken, `manifest.reference`):")
    add("")
    add("- `street_exact_match` — the decoder output contains the spoken street name (token match).")
    add("- `changed_from_baseline` — output differs from `beam_no_hotwords`.")
    add("- `helped` — changed and the expected street appeared where the baseline lacked it.")
    add("- `harmed` — changed and the expected street was lost vs the baseline.")
    add("- `expected_kept` — changed but the expected street is still present.")
    add("- `neither` — changed and the expected street is absent in both baseline and config.")
    add("- `no_street_detected` — no known street form (canonical or experimental) detected.")
    add("- `other_street_records` — a street different from the expected one was detected.")
    add("- `unsupported_street_introduced` — an introduced street whose tokens appear in neither "
        "baseline nor greedy text (false-bias candidate).")
    add("- `requires_manual_review` — the automatic metric cannot decide "
        "(see review reasons in `per_file_results.jsonl`).")
    add("")
    add("A change is NEVER counted as an improvement just because the output now contains a dictionary")
    add("street: only a change toward the *expected* street counts as `helped`.")
    add("")
    add("## Limitations")
    add("")
    add("- Every record in this dataset is a FULL recording: the canonical (nominative) street name")
    add("  is spoken. The extra oblique forms are meant for colloquial speech where `улица` is dropped")
    add("  (\"на Советской\", \"с Садовой\"); on this nominative-only set they can only be neutral or")
    add("  harmful, so their benefit cannot be measured here. A NATURAL set is required for that.")
    add("- pyctcdecode 0.5.0 `HotwordScorer` splits every hotword into whitespace-separated unigrams")
    add("  and scores whole-word matches, so multi-word anthroponyms are not enforced as phrases.")
    add("- No KenLM retraining, no fine-tuning, no production change.")
    add("")
    add("## Summary")
    add("")
    add("| configuration | hotwords | street_exact_match | rate | changed | helped | harmed | kept |"
        " neither | no_street | other_street | unsupported | manual_review |")
    add("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for row in payload["configs"]:
        add(f"| {row['configuration']} | {row['hotword_count']} | {row['street_exact_match']} | "
            f"{row['street_exact_match_rate']} | {row['changed_from_baseline']} | {row['helped']} | "
            f"{row['harmed']} | {row['expected_kept']} | {row['neither']} | {row['no_street_detected']} | "
            f"{row['other_street_records']} | {row['unsupported_street_introduced']} | "
            f"{row['requires_manual_review']} |")
    add("")
    add("## Effect of the extra forms (canonical vs canonical+forms, same weight)")
    add("")
    add("| weight | identical output | fixed by forms | broken by forms |")
    add("|---|---|---|---|")
    for w in weights:
        c = examples["forms_effect_vs_canonical"][f"w{w:g}"]["counts"]
        add(f"| {w:g} | {c['identical_text']} | {c['fixed_by_forms']} | {c['broken_by_forms']} |")
    add("")
    add("## Examples: helped by the extra forms")
    add("")
    any_fixed = False
    for w in weights:
        for r in examples["forms_effect_vs_canonical"][f"w{w:g}"]["fixed_by_forms"]:
            any_fixed = True
            add(f"- w{w:g} {r['id']} ({r['expected_street']}): canonical missing -> `{r['text']}`")
    if not any_fixed:
        add("None: on this nominative-only set the extra forms fixed no record.")
    add("")
    add("## Examples: harmed by the extra forms")
    add("")
    any_broken = False
    for w in weights:
        for r in examples["forms_effect_vs_canonical"][f"w{w:g}"]["broken_by_forms"]:
            any_broken = True
            add(f"- w{w:g} {r['id']} ({r['expected_street']}): canonical correct -> `{r['text']}` "
                f"({'; '.join(r['review_reasons'])})")
    if not any_broken:
        add("None: the extra forms broke no record where canonical was correct.")
    add("")
    add("## Targeted streets")
    add("")
    add("Records for: " + ", ".join(targeted) + ".")
    add("See `examples.json` -> `targeted_streets` for the full per-config text.")
    add("")
    add("## Files")
    add("")
    add("- `summary.json` / `summary.csv` — per-configuration counters and the best hotword config.")
    add("- `per_file_results.jsonl` / `per_file_results.csv` — per (record, configuration) classification.")
    add("- `examples.json` — helped / harmed / neither tables and the forms-vs-canonical effect.")
    add("- `config.json` — reproducible experiment configuration, hotwords and env versions.")
    add("- `logprobs_cache.npz` — acoustic log-probs, one array per WAV (acoustic forward ran once).")
    add("")
    add("## Reproduce")
    add("")
    add("```")
    add("uv run --group decoders python scripts/run_street_hotword_experiment.py \\")
    add(f"    --dataset-dir {cfg['dataset_dir']} --weights {','.join(f'{w:g}' for w in weights)}")
    add("```")
    add("")
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")

TARGETED_STREETS = ["Советская", "Садовая", "Лесная", "Пятая", "Искра", "Караташ", "Сабантуй"]


def print_report(payload: dict, counts: dict, examples: dict, out_dir: Path,
                 weights: list[float]) -> None:
    print()
    print("=" * 72)
    print("STREET HOTWORD-FORMS EXPERIMENT — RESULT")
    print("=" * 72)
    print(f"results dir:        {out_dir}")
    print(f"records total:      {counts['records_total']}")
    print(f"records processed:  {counts['records_processed']}")
    print(f"acoustic forwards:  {counts['acoustic_forwards']}")
    print(f"decoder runs:       {counts['decoder_runs']} "
          f"({counts['configs']} configs x {counts['records_processed']} records)")
    if counts["skipped_no_speech"]:
        print(f"skipped (no speech): {len(counts['skipped_no_speech'])}")
    print()
    header = (f"{'configuration':<26}{'exact':>6}{'rate':>8}{'chg':>5}{'help':>6}"
              f"{'harm':>6}{'kept':>6}{'neith':>6}{'nost':>6}{'other':>6}{'unsup':>6}{'review':>8}")
    print(header)
    print("-" * len(header))
    for row in payload["configs"]:
        print(f"{row['configuration']:<26}{row['street_exact_match']:>6}"
              f"{row['street_exact_match_rate']:>8}{row['changed_from_baseline']:>5}"
              f"{row['helped']:>6}{row['harmed']:>6}{row['expected_kept']:>6}"
              f"{row['neither']:>6}{row['no_street_detected']:>6}{row['other_street_records']:>6}"
              f"{row['unsupported_street_introduced']:>6}{row['requires_manual_review']:>8}")
    print()
    best = payload["best_hotword_config_by_street_exact_match"]
    if best:
        print(f"best hotword config by street_exact_match: {best['configuration']} "
              f"({best['street_exact_match']} / {counts['records_processed']}, "
              f"helped={best['helped']}, harmed={best['harmed']})")
    print()
    for w in weights:
        c = examples["forms_effect_vs_canonical"][f"w{w:g}"]["counts"]
        print(f"extra forms @w{w:g}: identical={c['identical_text']} "
              f"fixed={c['fixed_by_forms']} broken={c['broken_by_forms']}")
    print()
    print("most useful changes vs beam baseline (canonical_plus_forms@w5):")
    helped = examples["vs_beam_baseline"].get("canonical_plus_forms@w5", {}).get("helped", [])
    for r in helped[:10]:
        print(f"  + {r['id']} ({r['expected_street']}): `{r['baseline']}` -> `{r['text']}`")
    if not helped:
        print("  (none)")
    print("most harmful changes vs beam baseline (canonical_plus_forms@w5):")
    harmed = examples["vs_beam_baseline"].get("canonical_plus_forms@w5", {}).get("harmed", [])
    for r in harmed[:10]:
        print(f"  - {r['id']} ({r['expected_street']}): `{r['baseline']}` -> `{r['text']}`")
    if not harmed:
        print("  (none)")
    print("=" * 72)
def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    weights = parse_weights(args.weights)

    street_names = load_street_names(args.streets)
    exp = load_experiment_forms(args.forms) if args.forms else load_experiment_forms()
    errors = validate_experiment_forms(exp, street_names)
    if errors:
        print("experimental dictionary validation FAILED:")
        for e in errors:
            print(f"  - {e}")
        return 2

    canonical_hw = build_canonical_hotwords(street_names)
    prod_hw, _audit = load_streets(args.streets)
    if prod_hw != canonical_hw:
        print("WARNING: canonical hotwords differ from hotwords.load_streets output "
              "(experiment uses the canonical view of the same file)")
    plus_hw = build_experiment_hotwords(street_names, exp)
    indexes = {
        "canonical": build_canonical_index(street_names),
        "additional": build_additional_index(street_names, exp),
        "combined": build_combined_index(street_names, exp),
    }
    plan = build_config_plan(weights)

    records = load_records(args.dataset_dir, args.variant, args.limit)
    if not records:
        print(f"no saved '{args.variant}' records in {args.dataset_dir}")
        return 1

    out_dir = make_out_dir(args.out_base)
    print(f"Results dir: {out_dir}")
    print(f"Records: {len(records)}; canonical hotwords: {len(canonical_hw)}; "
          f"canonical+forms: {len(plus_hw)}; configs: {len(plan)}")

    config = build_experiment_config(
        dataset_dir=str(args.dataset_dir), streets_file=str(args.streets),
        street_count=len(street_names), n_records=len(records), weights=weights,
        beam_width=args.beam_width, decoder_params=DECODER_PARAMS,
        canonical_hotwords=canonical_hw, canonical_plus_forms_hotwords=plus_hw,
        experiment_forms=exp.as_dict(), env=env_versions(),
    )
    (out_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    t0 = time.time()
    if args.cache:
        import shutil
        acoustic = load_acoustic_cache(args.cache)
        if Path(args.cache).resolve() != (out_dir / "logprobs_cache.npz").resolve():
            shutil.copyfile(args.cache, out_dir / "logprobs_cache.npz")
        print(f"Reused logprobs cache: {args.cache} ({len(acoustic['arrays'])} arrays)")
    else:
        print("Running acoustic model once per WAV ...")
        acoustic = run_acoustic(
            records, out_dir, str(args.model_path) if args.model_path else None
        )
    arrays = acoustic["arrays"]
    processed = [rec for rec in records if rec["id"] in arrays]
    print(f"Acoustic forwards: {acoustic['forwards']}; processed records: {len(processed)}; "
          f"elapsed: {time.time() - t0:.1f}s")

    if not processed:
        print("no record produced log-probs; nothing to decode")
        return 1

    print("Decoding all configs on the SAME cached logprobs ...")
    texts, decoder_runs = decode_all(
        processed, arrays, plan, args.beam_width, canonical_hw, plus_hw,
        str(args.kenlm_path) if args.kenlm_path else None,
    )
    rows, summary = classify_all(processed, texts, plan, indexes, street_names)

    counts = {
        "records_total": len(records),
        "records_processed": len(processed),
        "skipped_no_speech": acoustic.get("skipped_no_speech", []),
        "acoustic_forwards": acoustic["forwards"],
        "decoder_runs": decoder_runs,
        "configs": len(plan),
    }
    payload = write_summary(out_dir, config, summary, counts, canonical_hw, plus_hw)
    write_per_file(out_dir, rows)
    examples = build_examples(rows, weights, TARGETED_STREETS)
    (out_dir / "examples.json").write_text(
        json.dumps(examples, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_readme(out_dir, payload, examples, weights, counts, canonical_hw, plus_hw,
                 exp.as_dict(), TARGETED_STREETS)
    print_report(payload, counts, examples, out_dir, weights)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

