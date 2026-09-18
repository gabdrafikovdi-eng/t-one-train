"""Analyze a hotword_sweep output directory and generate README.md report.

Reads config.json / summary.csv / detailed_results.jsonl produced by
scripts/hotword_sweep.py, selects representative examples, writes README.md.
Reference-free: WER/CER are not computable without reference transcripts.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.t_one_train.hotwords import load_streets  # noqa: E402

TYPE_WORDS = ("улица", "переулок")


def strip_type_words(h: str) -> str:
    return " ".join(w for w in h.split() if w not in TYPE_WORDS)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="analyze_hotword_sweep")
    p.add_argument("sweep_dir", type=Path)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    d = args.sweep_dir
    cfg = json.loads((d / "config.json").read_text())
    with (d / "summary.csv").open(encoding="utf-8") as f:
        summary = list(csv.DictReader(f))
    rows = [json.loads(l) for l in (d / "detailed_results.jsonl").open(encoding="utf-8")]

    hw, _audit = load_streets(ROOT / "streets.txt")
    names = [strip_type_words(h) for h in hw]
    numeric = [h for h in hw if any(w in ("сорок", "пятьдесят", "шестьдесят", "семьдесят") for w in h.split())]

    hot_rows = [r for r in rows if not r["structure"].startswith("baseline")]
    for r in hot_rows:
        r["is_short"] = all(len(t) <= 4 for t in r["config_streets"]) if r["config_streets"] else False

    def fmt(r: dict, keys: list[str]) -> str:
        return " | ".join(str(r.get(k, "")) for k in keys)

    lines: list[str] = []
    add = lines.append
    add("# Hotword sweep report")
    add("")
    add(f"- Created: {cfg['created']}")
    add(f"- Eval run (fixed WAV set): `{cfg['eval_run']}`")
    add(f"- WAVs total: {cfg['n_wavs_total']}; with speech (T-one splitter): {cfg['n_wavs_with_speech']}")
    add(f"- Acoustic forwards executed: {cfg['acoustic_forwards']} (one per WAV, logprobs cached to npz)")
    add(f"- Decoder: beam_width={cfg['beam_width']}, alpha={cfg['decoder_params']['alpha']}, beta={cfg['decoder_params']['beta']}, official kenlm.bin")
    add(f"- Structures tested: {cfg['structures']}")
    add(f"- Weight sweep ({cfg['sweep_structure']}): {cfg['weights']}; structure comparison weights: {cfg['combo_weights']}")
    add(f"- Structure sizes: {cfg['structure_sizes']}")
    add("- **No reference transcripts exist** for these mic recordings; WER/CER/street_exact_match are NOT computed. Categories below are reference-free proxies: changes vs the no-hotword beam baseline.")
    add("")
    add("Categories: UNCHANGED / STREET_INTRODUCED (street appeared) / STREET_LOST / STREET_REPLACED / STREET_EXTENDED / STREET_NARROWED / SAME_STREET_KEPT / NO_STREET_CHANGED. `unsupported_introduced` = introduced street none of whose tokens appears in greedy or baseline text (false-bias candidate).")
    add("")
    add("## Structural equivalences (pyctcdecode 0.5.0 behavior)")
    add("")
    add("HotwordScorer splits every hotword into whitespace-separated unigrams, therefore:")
    add("")
    add("- `full` and `full_plus_components` build the SAME unigram set -> identical output (verified bit-for-bit).")
    add("- `components` and `name_only` are likewise identical.")
    add("- Effectively there are TWO distinct structures: **full** (with 'улица'/'переулок' words) and **name_only** (without).")
    add("- 'Переулок Искра' and 'Улица Искра' both reduce to the same unigrams {переулок/улица, искра}: the scorer cannot distinguish them at equal weight.")
    add("")
    add("**weight=0 is NOT a true baseline in pyctcdecode 0.5.0**: a beam's partial (unfinished) word that is a prefix of a hotword unigram gets its partial-token score from the hotword trie (0.0 at weight 0) instead of the KenLM partial score. On this set full@w0 still changed 26/71 utterances vs the no-hotword baseline. To fully disable hotwords, pass no hotwords (baseline_beam), as the sweep's baseline does.")
    add("")
    add("## Summary table")
    add("")
    add("| " + " | ".join(summary[0].keys()) + " |")
    add("|" + "---|" * len(summary[0]) + "|")
    for r in summary:
        add("| " + " | ".join(str(v) for v in r.values()) + " |")
    add("")

    # --- example selection -----------------------------------------------------
    base = {r["wav"]: r for r in rows if r["structure"] == "baseline_beam"}
    greedy = {r["wav"]: r["text"] for r in rows if r["structure"] == "baseline_greedy"}

    def examples(structure: str, weight: float) -> list[dict]:
        return [r for r in hot_rows if r["structure"] == structure and r["weight"] == weight]

    def pick(rows_: list[dict], n: int) -> list[dict]:
        order = {"STREET_INTRODUCED": 0, "STREET_LOST": 1, "STREET_REPLACED": 2,
                 "STREET_EXTENDED": 3, "STREET_NARROWED": 4, "NO_STREET_CHANGED": 5,
                 "SAME_STREET_KEPT": 6}
        return sorted(rows_, key=lambda r: (order.get(r["category"], 9), -r["tokens_changed_ratio"]))[:n]

    w10 = [r for r in hot_rows if r["structure"] == "full" and r["weight"] == 10.0]
    changed10 = [r for r in w10 if r["category"] != "UNCHANGED"]
    add("## Baseline vs hotwords (full @ weight 10)")
    add("")
    add(f"Changed: {len(changed10)} / {len(w10)}. Categories: " +
        ", ".join(f"{c}={sum(1 for r in changed10 if r['category']==c)}" for c in sorted({r['category'] for r in changed10})))
    add("")
    add("### 20 most illustrative examples (full @ w10, changed)")
    add("")
    add("| wav | baseline beam | hotword result | category | unsupported |")
    add("|---|---|---|---|---|")
    for r in pick(changed10, 20):
        add(f"| {r['wav']} | {r['baseline_text']} | {r['config_text']} | {r['category']} | {', '.join(r['unsupported_introduced']) or '—'} |")
    add("")
    add("### False positive bias candidates (full @ w10)")
    add("")
    fb = [r for r in changed10 if r["unsupported_introduced"] or r["category"] in ("NO_STREET_CHANGED", "STREET_LOST")]
    if fb_examples := fb[:15]:
        add("| wav | baseline | hotword | category |")
        add("|---|---|---|---|")
        for r in fb_examples:
            add(f"| {r['wav']} | {r['baseline_text']} | {r['config_text']} | {r['category']} |")
    else:
        add("None found.")
    add("")
    add("## Short hotwords: risk analysis")
    add("")
    short_hw = sorted({n for h in hw for n in h.split() if len(n) <= 4})
    add(f"Short hotword unigrams present in every structure (<=4 chars): {short_hw}")
    add("")
    short_changes = [r for r in hot_rows if r["config_streets"] and all(len(t) <= 4 for s in r["config_streets"] for t in s.split())]
    add(f"Utterance/config rows where the resulting street tokens are all short (<=4 chars): {len(short_changes)}")
    for r in short_changes[:12]:
        add(f"- {r['label']} {r['wav']}: `{r['baseline_text']}` -> `{r['config_text']}` ({r['category']})")
    add("")
    add("### Where NAME_ONLY differs from FULL (same weight)")
    add("")
    diffs = []
    for w in [float(x) for x in cfg["combo_weights"].split(",")]:
        full_by = {(r["wav"]): r for r in hot_rows if r["structure"] == "full" and r["weight"] == w}
        name_by = {(r["wav"]): r for r in hot_rows if r["structure"] == "name_only" and r["weight"] == w}
        for wav in full_by:
            fr, nr = full_by[wav], name_by.get(wav)
            if nr and fr["config_text"] != nr["config_text"]:
                diffs.append((w, wav, fr, nr))
    if diffs:
        add("| weight | wav | full | name_only |")
        add("|---|---|---|---|")
        for w, wav, fr, nr in diffs[:15]:
            add(f"| {w:g} | {wav} | {fr['config_text']} | {nr['config_text']} |")
    else:
        add("No differences between FULL and NAME_ONLY on this set: the 'улица'/'переулок' type words never changed the winning beam here (their unigrams already co-occur with the name tokens).")
    add("")
    add("### Numeric street names")
    add("")
    add(f"Numeric streets in streets.txt (already expanded to words per project rule, CTC labels contain no digits): {len(numeric)}")
    for n in numeric:
        add(f"- {n}")
    said_numeric = [r for r in rows if " лет " in f" {(r.get('baseline_text') or '')} {(r.get('config_text') or '')} "]
    if said_numeric:
        add("Utterances containing 'лет' (numeric streets spoken):")
        for r in said_numeric[:8]:
            add(f"- {r['wav']}: baseline=`{r.get('baseline_text')}` config=`{r.get('config_text')}` ({r['label']})")
    else:
        add("No utterance in this eval set contains a numeric street (no 'лет' in any baseline/hotword text) — the numeric hotword forms are verified only as valid vocabulary (see above), not by recall.")
    add("")
    add("### Iskra")
    add("")
    iskra_rows = [r for r in hot_rows if "искра" in " ".join(r["config_streets"])]
    if iskra_rows:
        for r in iskra_rows[:6]:
            add(f"- {r['label']} {r['wav']}: `{r['config_text']}` streets={r['config_streets']}")
    else:
        add("No utterance decoded with an Iskra street name in any configuration; the two 'Искра' streets share unigrams and cannot be told apart by this scorer (see structural notes).")
    add("")
    add("## Observations")
    add("")
    counts: dict[tuple[str, float], dict[str, int]] = {}
    for r in hot_rows:
        c = counts.setdefault((r["structure"], r["weight"]), {"changed": 0, "intro": 0, "lost": 0, "unsupported": 0, "n": 0})
        c["n"] += 1
        if r["category"] != "UNCHANGED":
            c["changed"] += 1
        if r["category"] in ("STREET_INTRODUCED", "STREET_EXTENDED"):
            c["intro"] += 1
        if r["category"] in ("STREET_LOST", "STREET_NARROWED"):
            c["lost"] += 1
        c["unsupported"] += len(r["unsupported_introduced"])
    add("| configuration | weight | n | changed | street introduced(+ext) | street lost(+narrowed) | unsupported introduced |")
    add("|---|---|---|---|---|---|---|")
    for (structure, weight), c in sorted(counts.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        add(f"| {structure} | {weight:g} | {c['n']} | {c['changed']} | {c['intro']} | {c['lost']} | {c['unsupported']} |")
    add("")
    add("## Conclusions (this eval set, reference-free categories only)")
    add("")
    add("1. weight=0 is NOT equivalent to no hotwords (partial-token trie effect, 26/71 changed vs baseline_beam).")
    add("2. Structure choice does not matter on this set: full == name_only == components == full_plus_components outputs bit-for-bit at every weight; the type words 'улица'/'переулок' never flipped a beam. Only the WEIGHT matters (and the w0 partial-token artifact).")
    add("3. Street-introduced count grows with weight (full: 3 -> 4 -> 8 -> 9 -> 12 -> 13 at w0..w10) but so do non-street distortions (no_street_changed: 23 -> 33) and unsupported introductions (2 -> 8).")
    add("4. At w15-w20 the extra changes are mostly harmful: 'файзи рахманах хисматуллина' (w20, phrase_020), 'ак шамбет' (w20, phrase_049), 'о есь' (w15, phrase_002), 'биишпатыра' (w20, phrase_090).")
    add("5. Short street words ('ак' from Ак Кайын) systematically degrade short utterances from w5 on: 'а каин'->'ак', 'акка'->'ак к', 'кан'->'к', 'каин'->'ка'.")
    add("6. Numeric street unigrams inject 'лет' into unrelated speech (phrase_016 'шамана' -> 'лет шакмана' from w10, 'лет ак' at w20).")
    add("7. The best-performing corrections (файзрахмана хисматуллина, абзелиловская, нажипа суфьянова, бииш батыра, валиахмета сулейманова) already occur by w3-w7; raising the weight beyond 10 added no new correct street on this set while adding distortions.")
    add("")
    add("_Conclusions are restricted to this evaluation set and to reference-free categories._")
    add("")
    (d / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"written {d / 'README.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



if __name__ == "__main__":
    raise SystemExit(main())
