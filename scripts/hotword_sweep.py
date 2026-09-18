"""Hotword structure & weight sweep on a fixed evaluation set (existing run's WAVs).

Reuses WAVs from a previous scripts/test_decoders.py run; acoustic model runs
ONCE per WAV (logprobs cached to npz), then the same logprobs are decoded by
every hotword structure x weight configuration. No fine-tuning, no new KenLM.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import difflib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.t_one_train.decoder_experiment import build_hotwords_json, load_wav_any  # noqa: E402
from src.t_one_train.hotwords import load_streets  # noqa: E402

TYPE_WORDS = ("улица", "переулок")


def find_latest_eval_run(base: Path) -> Path:
    candidates = []
    for d in sorted(base.iterdir()):
        if d.is_dir() and not d.name.startswith("hotword_sweep") \
                and list(d.glob("phrase_*.wav")) and (d / "run.json").exists():
            candidates.append((d.stat().st_mtime, d))
    if not candidates:
        raise SystemExit(f"No eval runs with WAVs found under {base}")
    return max(candidates)[1]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="hotword_sweep")
    p.add_argument("--eval-run", type=Path, default=None)
    p.add_argument("--out-base", type=Path, default=ROOT / "results")
    p.add_argument("--weights", default="0,1,2,3,5,7,10,15,20")
    p.add_argument("--sweep-structure", default="full")
    p.add_argument("--combo-weights", default="1,3,5,7,10")
    p.add_argument("--structures", default="full,name_only,components,full_plus_components")
    p.add_argument("--beam-width", type=int, default=200)
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args(argv)


def strip_type_words(hotword: str) -> str:
    return " ".join(w for w in hotword.split() if w not in TYPE_WORDS)


def build_structures(full_hotwords: list[str]) -> dict[str, list[str]]:
    """pyctcdecode 0.5.0 splits hotwords into unigrams, so 'full' and
    'full_plus_components' share the same unigram set (verified in results)."""
    name_only = [strip_type_words(h) for h in full_hotwords]
    components: list[str] = []
    seen: set[str] = set()
    for h in name_only:
        for w in h.split():
            if w not in seen:
                seen.add(w)
                components.append(w)
    return {
        "full": list(full_hotwords),
        "name_only": name_only,
        "components": components,
        "full_plus_components": list(full_hotwords) + components,
    }


def detect_streets(text: str, street_token_lists: list[list[str]]) -> set[str]:
    words = text.split()
    found: set[str] = set()
    for tokens in street_token_lists:
        n = len(tokens)
        if n == 0:
            continue
        for i in range(len(words) - n + 1):
            if words[i : i + n] == tokens:
                found.add(" ".join(tokens))
                break
    return found


def tokens_changed_ratio(a: str, b: str) -> float:
    if a == b:
        return 0.0
    return 1.0 - difflib.SequenceMatcher(None, a.split(), b.split()).ratio()


def classify(baseline_text: str, config_text: str, base_st: set[str], cfg_st: set[str],
             greedy_text: str) -> tuple[str, list[str]]:
    if config_text == baseline_text:
        return "UNCHANGED", []
    added, removed = cfg_st - base_st, base_st - cfg_st
    if added and not base_st:
        cat = "STREET_INTRODUCED"
    elif removed and not cfg_st:
        cat = "STREET_LOST"
    elif added and removed:
        cat = "STREET_REPLACED"
    elif added:
        cat = "STREET_EXTENDED"
    elif removed:
        cat = "STREET_NARROWED"
    elif cfg_st and base_st:
        cat = "SAME_STREET_KEPT"
    else:
        cat = "NO_STREET_CHANGED"
    unsupported = [s for s in added
                   if not any(t in greedy_text.split() or t in baseline_text.split() for t in s.split())]
    return cat, unsupported


def build_config_plan(args: argparse.Namespace) -> list[tuple[str, float]]:
    plan: list[tuple[str, float]] = [("baseline_beam", 0.0), ("baseline_greedy", 0.0)]
    for w in [float(x) for x in args.weights.split(",")]:
        plan.append((args.sweep_structure, w))
    for s in [x.strip() for x in args.structures.split(",")]:
        for w in [float(x) for x in args.combo_weights.split(",")]:
            if (s, w) not in plan:
                plan.append((s, w))
    return plan


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    eval_run = args.eval_run or find_latest_eval_run(args.out_base)
    print(f"Eval run: {eval_run}")

    wav_paths = sorted(eval_run.glob("phrase_*.wav"))
    if args.limit:
        wav_paths = wav_paths[: args.limit]
    meta: dict[str, dict] = {}
    for p in eval_run.glob("phrase_*.json"):
        try:
            meta[p.name.replace(".json", ".wav")] = json.loads(p.read_text())
        except Exception:
            pass

    out = args.out_base / f"hotword_sweep_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    out.mkdir(parents=True)
    print(f"Output: {out}")

    full_hotwords, audit = load_streets(ROOT / "streets.txt")
    structures = build_structures(full_hotwords)
    street_token_lists = [strip_type_words(h).split() for h in full_hotwords]
    plan = build_config_plan(args)
    print(f"Configs: {len(plan)} x {len(wav_paths)} WAVs")

    from src.t_one_train.test_decoders_core import (
        collect_phrase_logprobs,
        load_acoustic_model,
        load_beam_decoder,
    )
    from tone.decoder import GreedyCTCDecoder

    # --- acoustic forward ONCE per wav; logprobs cached to npz ---
    entries: list[dict] = []
    arrays: dict[str, np.ndarray] = {}
    model = load_acoustic_model()
    for idx, wav in enumerate(wav_paths, 1):
        prev = meta.get(wav.name)
        if not prev or "error" in prev:
            print(f"[{idx}/{len(wav_paths)}] {wav.name}: skipped (no speech in source run)")
            continue
        pcm, _sr = load_wav_any(wav)
        phrases, _ = collect_phrase_logprobs(model, pcm.astype(np.int32))
        if not phrases:
            print(f"[{idx}/{len(wav_paths)}] {wav.name}: no speech (splitter)")
            continue
        key = wav.stem
        arrays[key] = np.concatenate(phrases, axis=0)
        entries.append({
            "wav": wav.name, "key": key,
            "duration": prev.get("duration_seconds"),
            "prev_greedy": prev.get("greedy"),
            "prev_beam_kenlm": prev.get("beam_kenlm"),
            "prev_hotwords": prev.get("beam_kenlm_hotwords"),
        })
        print(f"[{idx}/{len(wav_paths)}] {wav.name}: logprobs {arrays[key].shape}")
    np.savez_compressed(out / "logprobs_cache.npz", **arrays)

    greedy = GreedyCTCDecoder()
    beam = load_beam_decoder()

    config = {
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "eval_run": str(eval_run),
        "n_wavs_total": len(wav_paths),
        "n_wavs_with_speech": len(entries),
        "acoustic_forwards": len(entries),
        "weights": args.weights,
        "combo_weights": args.combo_weights,
        "structures": args.structures,
        "sweep_structure": args.sweep_structure,
        "beam_width": args.beam_width,
        "decoder_params": {"alpha": 0.4, "beta": 0.9},
        "kenlm": "hf:t-tech/T-one/kenlm.bin (cached)",
        "note": "no reference transcripts; reference-free metrics only",
        "structure_sizes": {k: len(v) for k, v in structures.items()},
        "hotwords_audit": build_hotwords_json(audit, "streets.txt"),
    }
    (out / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    # --- decode: every config over the SAME cached logprobs ---
    baseline_by_wav: dict[str, tuple[str, set[str], str]] = {}
    per_config: list[tuple[str, float, list[dict]]] = []
    for ci, (structure, weight) in enumerate(plan):
        hotwords = None if (structure == "baseline_beam" or structure == "baseline_greedy") else structures[structure]
        label = structure if structure.startswith("baseline") else f"{structure}@w{weight:g}"
        print(f"[{ci+1}/{len(plan)}] {label}", flush=True)
        rows = []
        for e in entries:
            lp = arrays[e["key"]]
            if structure == "baseline_beam":
                text = beam.decode(lp, beam_width=args.beam_width)
            elif structure == "baseline_greedy":
                text = greedy.forward(lp)
            else:
                text = beam.decode(lp, beam_width=args.beam_width,
                                   hotwords=hotwords, hotword_weight=weight)
            rows.append({"wav": e["wav"], "structure": structure, "weight": weight,
                         "label": label, "text": text})
        per_config.append((structure, weight, rows))
        if structure == "baseline_beam":
            for r in rows:
                baseline_by_wav[r["wav"]] = (r["text"], detect_streets(r["text"], street_token_lists), "")
        if structure == "baseline_greedy":
            for r in rows:
                t, st, _ = baseline_by_wav.get(r["wav"], ("", set(), ""))
                baseline_by_wav[r["wav"]] = (t, st, r["text"])

    # second pass: classification vs baseline; detailed_results.jsonl
    cat_counters: dict[tuple[str, float], dict[str, int]] = {}
    with (out / "detailed_results.jsonl").open("w", encoding="utf-8") as df:
        for structure, weight, rows in per_config:
            if structure.startswith("baseline"):
                for r in rows:
                    df.write(json.dumps({**r, "category": "BASELINE", "unsupported": []},
                                        ensure_ascii=False) + "\n")
                continue
            for r in rows:
                b_text, b_st, g_text = baseline_by_wav[r["wav"]]
                c_st = detect_streets(r["text"], street_token_lists)
                cat, unsupported = classify(b_text, r["text"], b_st, c_st, g_text)
                ratio = tokens_changed_ratio(b_text, r["text"])
                df.write(json.dumps({
                    "wav": r["wav"], "structure": structure, "weight": weight,
                    "label": r["label"], "baseline_text": b_text,
                    "greedy_text": g_text, "config_text": r["text"],
                    "baseline_streets": sorted(b_st), "config_streets": sorted(c_st),
                    "category": cat, "unsupported_introduced": unsupported,
                    "tokens_changed_ratio": round(ratio, 3),
                }, ensure_ascii=False) + "\n")
                counters = cat_counters.setdefault((structure, weight), {
                    "changed": 0, "changed_gt30pct": 0, "unchanged": 0,
                    "street_introduced": 0,
                    "street_lost": 0, "street_replaced": 0, "street_extended": 0,
                    "street_narrowed": 0, "same_street_kept": 0, "no_street_changed": 0,
                    "unsupported_introduced": 0,
                })
                if cat == "UNCHANGED":
                    counters["unchanged"] += 1
                else:
                    counters["changed"] += 1
                if ratio > 0.3:
                    counters["changed_gt30pct"] += 1
                counters[cat.lower()] += 1
                counters["unsupported_introduced"] += len(unsupported)

    with (out / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        fields = ["configuration", "weight", "n", "changed", "unchanged", "changed_gt30pct",
                  "street_introduced", "street_lost", "street_replaced",
                  "street_extended", "street_narrowed", "same_street_kept",
                  "no_street_changed", "unsupported_introduced"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerow({"configuration": "baseline_beam", "weight": 0,
                         "n": len(entries), **{k: 0 for k in fields[3:]}})
        for (structure, weight), c in sorted(cat_counters.items(), key=lambda x: (x[0][0], x[0][1])):
            writer.writerow({"configuration": structure, "weight": weight,
                             "n": len(entries), **c})

    print(f"Done: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
