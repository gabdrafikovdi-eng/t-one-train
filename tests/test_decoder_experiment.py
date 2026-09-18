"""Tests for the decoder experiment's own logic (no T-one/pyctcdecode tests)."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.test_decoders import MIC_CONFIG_WEIGHTS, parse_args  # noqa: E402
from src.t_one_train.decoder_experiment import (  # noqa: E402
    MIC_RESULT_FIELDS,
    SESSION_FIELDS,
    TARGET_RATE,
    build_hotwords_json,
    build_mic_config_plan,
    build_mic_phrase_record,
    build_session_summary,
    evaluate_mic_texts,
    load_wav_any,
    mic_config_dicts,
    mic_config_rows,
    mic_display_label,
    run_dir,
    save_wav,
    session_summary_rows,
    write_rows_csv,
)
from src.t_one_train.forms import NUMERALS as FORMS_NUMERALS
from src.t_one_train.hotwords import NUMERALS, load_streets
from src.t_one_train.street_forms_experiment import (
    HARMED,
    HELPED,
    build_canonical_index,
    build_empty_index,
    load_street_names,
    resolve_expected_street,
)

STREETS = ROOT / "streets.txt"


@pytest.fixture(scope="module")
def loaded():
    return load_streets(STREETS)


def test_streets_unique_hotwords(loaded):
    hotwords, audit = loaded
    # Текущий streets.txt: 109 «голых» названий без префиксов и без заголовков.
    assert len(hotwords) == len(audit) == 109
    assert len(set(hotwords)) == 109


def test_no_empty_or_page_heading_hotwords(loaded):
    hotwords, audit = loaded
    assert all(h.strip() for h in hotwords)
    assert not any("Страница" in h for h in hotwords)
    # ни одна строка файла не отбрасывается
    assert not [e for e in audit if e.excluded]


def test_iskra_hotword(loaded):
    hotwords, _ = loaded
    # В новом списке «Искра» — одно название, один hotword.
    assert "искра" in hotwords
    assert hotwords.count("искра") == 1


def test_all_hotwords_lowercase_no_digits(loaded):
    hotwords, _ = loaded
    for h in hotwords:
        assert h == h.lower()
        assert not re.search(r"\d", h), h
        assert re.fullmatch(r"[а-яё ]+", h), h


def test_numeric_streets_expanded_to_words(loaded):
    hotwords, _ = loaded
    assert "сорок лет октября" in hotwords
    assert "сорок лет победы" in hotwords
    assert "пятьдесят лет победы" in hotwords
    assert "шестьдесят лет победы" in hotwords
    assert "шестьдесят пять лет победы" in hotwords
    assert "семьдесят лет октября" in hotwords
    assert "семьдесят лет победы" in hotwords
    assert len([h for h in hotwords if re.search(r"\bлет\b", h)]) == 7


def test_numerals_match_forms_module():
    assert NUMERALS == FORMS_NUMERALS


def test_unknown_numeral_rejected(tmp_path):
    p = tmp_path / "s.txt"
    p.write_text("Улица 33 Героев\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unreviewed numeral"):
        load_streets(p)


def test_hotwords_json_structure(loaded):
    hotwords, audit = loaded
    data = build_hotwords_json(audit, "streets.txt")
    assert data["total_lines"] == 109
    assert data["hotwords_used"] == hotwords
    assert len(data["entries"]) == 109
    iskra = [e for e in data["entries"] if "искра" in (e["hotword"] or "")]
    assert len(iskra) == 1 and iskra[0]["source"] == "Искра"


# --- run dir / storage --------------------------------------------------------


def test_run_dir_unique_and_named(tmp_path):
    d1 = run_dir(tmp_path)
    d2 = run_dir(tmp_path)
    assert d1 != d2
    assert d1.exists() and d2.exists()
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}(-\d+)?", d1.name)
    assert d2.name.startswith(d1.name[:19])  # same stamp, suffixed


def test_wav_roundtrip_and_resample(tmp_path):
    pcm = (np.sin(np.linspace(0, 100, 8000)) * 10000).astype(np.int16)
    out = tmp_path / "p.wav"
    save_wav(out, pcm, TARGET_RATE)
    audio, sr = load_wav_any(out)
    assert sr == TARGET_RATE
    assert audio.dtype == np.int32
    assert len(audio) == 8000

    out16 = tmp_path / "p16.wav"
    save_wav(out16, pcm, 16_000)
    audio2, sr2 = load_wav_any(out16)
    assert sr2 == TARGET_RATE
    assert abs(len(audio2) - 4000) <= 2


def test_mic_config_plan_is_fixed_four_configs():
    plan = build_mic_config_plan(MIC_CONFIG_WEIGHTS)
    assert plan == [
        ("greedy", "greedy", None),
        ("beam_no_hotwords", "beam_no_hotwords", None),
        ("canonical@w10", "canonical", 10.0),
        ("canonical@w15", "canonical", 15.0),
    ]
    weights = [w for _label, _kind, w in plan if w is not None]
    assert weights == [10.0, 15.0]  # no canonical@3/5/7/20
    assert mic_display_label("canonical@w10") == "Canonical @10"
    assert mic_display_label("beam_no_hotwords") == "Beam"
    assert mic_display_label("greedy") == "Greedy"


def test_mic_config_dicts_hotword_count_only_for_canonical():
    plan = build_mic_config_plan(MIC_CONFIG_WEIGHTS)
    cfgs = mic_config_dicts(plan, 109)
    assert [c["display"] for c in cfgs] == [
        "Greedy", "Beam", "Canonical @10", "Canonical @15",
    ]
    assert [c["hotword_count"] for c in cfgs] == [0, 0, 109, 109]
    assert [c["weight"] for c in cfgs] == [None, None, 10.0, 15.0]


class _CountingGreedy:
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls = 0

    def forward(self, logprobs):
        self.calls += 1
        return self.text


class _CountingBeam:
    def __init__(self, plain: str, hw: str) -> None:
        self.plain = plain
        self.hw = hw
        self.calls_plain = 0
        self.calls_hw = 0
        self.hotwords_seen = []

    def decode(self, logprobs, beam_width=200, hotwords=None, hotword_weight=None):
        if hotwords is None:
            self.calls_plain += 1
            return self.plain
        self.calls_hw += 1
        self.hotwords_seen.append((list(hotwords), hotword_weight))
        return self.hw


def test_decode_plan_runs_each_decoder_once_on_shared_logprobs():
    from src.t_one_train.test_decoders_core import DecoderTrio

    greedy = _CountingGreedy("лесная")
    beam = _CountingBeam("садовой", "советской")
    trio = DecoderTrio(greedy=greedy, beam=beam, beam_width=200)
    logprobs = np.zeros((10, 32), dtype=np.float32)
    plan = build_mic_config_plan(MIC_CONFIG_WEIGHTS)

    texts = trio.decode_plan(logprobs, plan, ["улица лесная"])

    assert texts == {
        "greedy": "лесная",
        "beam_no_hotwords": "садовой",
        "canonical@w10": "советской",
        "canonical@w15": "советской",
    }
    assert greedy.calls == 1  # acoustic forward happens once upstream
    assert beam.calls_plain == 1
    assert beam.calls_hw == 2  # one decode per canonical weight
    assert beam.hotwords_seen[0][1] == 10.0
    assert beam.hotwords_seen[1][1] == 15.0
    assert beam.hotwords_seen[0][0] == ["улица лесная"]


def test_decode_all_legacy_trio_maps_to_plan():
    from src.t_one_train.test_decoders_core import DecoderTrio

    trio = DecoderTrio(greedy=_CountingGreedy("g"), beam=_CountingBeam("b", "h"))
    texts = trio.decode_all(np.zeros((5, 32), dtype=np.float32), ["улица а"], 10.0)
    assert texts == {"greedy": "g", "beam_kenlm": "b", "beam_kenlm_hotwords": "h"}


def test_resolve_expected_street():
    names = load_street_names(STREETS)
    exact = resolve_expected_street("Лесная", names)
    assert exact.resolved and exact.street
    assert exact.user_input == "Лесная"
    assert exact.reference

    # падеж/предлог/«улица» — резолв в каноническое имя, оценка по канону
    inflected = resolve_expected_street("на Лесной", names)
    assert inflected.street == exact.street

    unknown = resolve_expected_street("Такой Улицы Нет", names)
    assert not unknown.resolved and unknown.street is None
    assert unknown.reference == "такой улицы нет"

    empty = resolve_expected_street("", names)
    assert empty.user_input == "" and not empty.resolved


def test_evaluate_mic_texts_and_phrase_record():
    names = load_street_names(STREETS)
    canon = build_canonical_index(names)
    empty = build_empty_index("additional")
    plan = build_mic_config_plan(MIC_CONFIG_WEIGHTS)
    expected = resolve_expected_street("Лесная", names)
    assert expected.resolved

    texts = {
        "greedy": "на советской",
        "beam_no_hotwords": "на садовой",
        "canonical@w10": "на лесной",
        "canonical@w15": "на советской",
    }
    ev = evaluate_mic_texts(
        texts, plan, expected_street=expected.street, reference=expected.reference,
        canonical_index=canon, additional_index=empty, combined_index=canon,
        street_names=names,
    )
    assert ev["beam_no_hotwords"]["found"] is False
    assert ev["canonical@w10"]["found"] is True
    assert ev["canonical@w10"]["changed_from_baseline"] is True
    assert ev["canonical@w10"]["classification"] == HELPED
    assert ev["canonical@w15"]["classification"] == HARMED
    assert ev["greedy"]["changed_from_baseline"] is False

    rec = build_mic_phrase_record(
        phrase_id="phrase_001",
        source="mic",
        wav_name="phrase_001.wav",
        expected_street=expected.street,
        expected_street_input=expected.user_input,
        evaluated=True,
        configs=mic_config_dicts(plan, 109),
        decoded=texts,
        evaluation=ev,
        sample_rate=8000,
        duration_seconds=1.5,
    )
    assert rec["id"] == "phrase_001"
    assert rec["evaluated"] is True
    assert rec["evaluation"]["canonical@w10"]["detected_streets"] == ["Лесная"]
    assert rec["evaluation"]["canonical@w10"]["changed_from_beam"] is True
    assert rec["evaluation"]["canonical@w10"]["helped"] is True
    assert rec["evaluation"]["canonical@w15"]["harmed"] is True


def test_mic_config_rows_and_session_summary():
    plan = build_mic_config_plan(MIC_CONFIG_WEIGHTS)
    configs = mic_config_dicts(plan, 109)
    good = {"changed_from_baseline": True, "expected_reference_match": True,
            "classification": HELPED, "detected_canonical": ["Лесная"]}
    bad = {"changed_from_baseline": True, "expected_reference_match": False,
           "classification": HARMED, "detected_canonical": ["Садовая"]}
    recs = []
    for i in (1, 2):
        texts = {"greedy": "", "beam_no_hotwords": "садовая",
                 "canonical@w10": "лесная", "canonical@w15": "садовая"}
        ev = {label: (good if label == "canonical@w10" else bad)
              for label, _k, _w in plan}
        recs.append(build_mic_phrase_record(
            phrase_id=f"phrase_{i:03d}", source="mic", wav_name=f"phrase_{i:03d}.wav",
            expected_street="Лесная", expected_street_input="Лесная", evaluated=True,
            configs=configs, decoded=texts, evaluation=ev,
            sample_rate=8000, duration_seconds=1.0,
        ))
    recs.append(build_mic_phrase_record(
        phrase_id="phrase_003", source="mic", wav_name="phrase_003.wav",
        expected_street=None, expected_street_input=None, evaluated=False,
        configs=configs,
        decoded={"greedy": "x", "beam_no_hotwords": "x",
                 "canonical@w10": "x", "canonical@w15": "x"},
        evaluation={label: {} for label, _k, _w in plan},
        sample_rate=8000, duration_seconds=1.0,
    ))

    rows = [row for rec in recs for row in mic_config_rows(rec)]
    assert len(rows) == 12  # 4 configs x 3 phrases
    assert set(rows[0]) == set(MIC_RESULT_FIELDS)
    assert rows[0]["found"] == "NOT FOUND" and rows[2]["found"] == "FOUND"
    assert rows[2]["changed_from_beam"] == "YES"

    summary = build_session_summary(recs, configs)
    assert summary["tests"] == 3 and summary["evaluated"] == 2
    by_label = {row["label"]: row for row in summary["configs"]}
    assert by_label["greedy"]["correct"] == 0
    assert by_label["beam_no_hotwords"]["correct"] == 0
    assert by_label["canonical@w10"]["correct"] == 2
    assert by_label["canonical@w10"]["helped"] == 2
    assert by_label["canonical@w15"]["harmed"] == 2
    assert session_summary_rows(summary)[0]["configuration"] == "Greedy"


def test_phrase_record_without_expected_stays_unscored():
    plan = build_mic_config_plan(MIC_CONFIG_WEIGHTS)
    names = load_street_names(STREETS)
    canon = build_canonical_index(names)
    rec = build_mic_phrase_record(
        phrase_id="phrase_009", source="mic", wav_name="phrase_009.wav",
        expected_street=None, expected_street_input=None, evaluated=False,
        configs=mic_config_dicts(plan, 109),
        decoded={label: "текст" for label, _k, _w in plan},
        evaluation=evaluate_mic_texts(
            {label: "текст" for label, _k, _w in plan}, plan, "", "",
            canon, build_empty_index("additional"), canon, names,
        ),
        sample_rate=8000, duration_seconds=2.0,
    )
    assert rec["evaluated"] is False
    assert all(v["found"] is None for v in rec["evaluation"].values())
    assert all(v["helped"] is False for v in rec["evaluation"].values())


def test_results_csv_written(tmp_path):
    plan = build_mic_config_plan(MIC_CONFIG_WEIGHTS)
    rec = build_mic_phrase_record(
        phrase_id="phrase_001", source="x.wav", wav_name="phrase_001.wav",
        expected_street=None, expected_street_input=None, evaluated=False,
        configs=mic_config_dicts(plan, 109),
        decoded={label: "t" for label, _k, _w in plan},
        evaluation={label: {} for label, _k, _w in plan},
        sample_rate=8000, duration_seconds=1.0,
    )
    rows = mic_config_rows(rec)
    write_rows_csv(tmp_path / "results.csv", MIC_RESULT_FIELDS, rows)
    text = (tmp_path / "results.csv").read_text(encoding="utf-8")
    header = ",".join(MIC_RESULT_FIELDS)
    assert text.splitlines()[0] == header
    assert "phrase_001.wav" in text


# --- CLI parsing --------------------------------------------------------------


def test_cli_parsing_defaults():
    args = parse_args(["--audio", "x.wav"])
    assert args.audio == Path("x.wav")
    assert not args.mic
    assert args.audio_dir is None
    assert args.beam_width == 200
    assert args.threshold == 0.01
    assert args.max_record_seconds == 30.0
    assert args.kenlm_path is None
    assert args.model_path is None
    assert args.streets == ROOT / "streets.txt"
    assert args.results_dir == ROOT / "results"


def test_cli_parsing_mic_and_options():
    args = parse_args(
        ["--mic", "--kenlm-path", "/tmp/kenlm.bin",
         "--threshold", "0.02", "--beam-width", "100", "--max-record-seconds", "25"]
    )
    assert args.mic
    assert args.kenlm_path == Path("/tmp/kenlm.bin")
    assert args.threshold == 0.02
    assert args.beam_width == 100
    assert args.max_record_seconds == 25.0


def test_cli_mutually_exclusive_modes():
    with pytest.raises(SystemExit):
        parse_args(["--mic", "--audio", "x.wav"])
    with pytest.raises(SystemExit):
        parse_args([])

