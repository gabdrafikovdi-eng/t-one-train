"""Tests for the real-world evaluation dataset logic.

No microphone and no ASR models are touched here: only parsing streets.txt,
reference normalization, ids/plan, manifest, resume and validation.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.t_one_train.decoder_experiment import save_wav  # noqa: E402
from src.t_one_train.forms import NUMERALS as FORMS_NUMERALS  # noqa: E402
from src.t_one_train.real_dataset import (  # noqa: E402
    PROMPT_NATURAL,
    PROMPT_SHORT,
    SKIP_TOKENS,
    TARGET_RATE,
    VARIANTS,
    append_manifest,
    audio_relpath,
    build_dataset_meta,
    build_plan,
    create_dataset_dir,
    digits_to_words,
    find_latest_dataset,
    manifest_state,
    normalize_reference,
    number_to_words,
    pending_records,
    read_manifest,
    read_streets_file,
    record_id,
    sha256_file,
    validate_dataset,
    wav_info,
    write_dataset_json,
    write_manifest_csv,
)

STREETS_FILE = ROOT / "streets.txt"


# --- helpers ------------------------------------------------------------------


def make_wav(path: Path, seconds: float = 1.0, rate: int = TARGET_RATE,
             freq: float = 200.0) -> float:
    """Write a synthetic tone as mono PCM16 WAV; return its duration."""
    n = int(seconds * rate)
    t = np.arange(n) / rate
    pcm = np.clip(np.sin(2 * np.pi * freq * t) * 0.3 * 32767, -32768, 32767).astype(np.int16)
    save_wav(path, pcm, rate)
    return round(n / rate, 3)


def saved_record(i: int, street: str, variant: str, text: str, duration: float) -> dict:
    rid = record_id(i, variant)
    return {
        "id": rid,
        "street_index": i,
        "street": street,
        "variant": variant,
        "status": "saved",
        "reference_raw": text,
        "reference": normalize_reference(text),
        "audio": audio_relpath(rid),
        "sample_rate": TARGET_RATE,
        "channels": 1,
        "duration_sec": duration,
    }


def skipped_record(i: int, street: str, variant: str) -> dict:
    return {
        "id": record_id(i, variant),
        "street_index": i,
        "street": street,
        "variant": variant,
        "status": "skipped",
        "reference_raw": "",
        "reference": "",
        "audio": None,
        "sample_rate": None,
        "channels": None,
        "duration_sec": None,
    }


def make_dataset(tmp_path: Path, streets: tuple[str, ...] = ("Ленина", "Гагарина"),
                 n_saved: int = 3, n_skipped: int = 1) -> tuple[Path, list[dict]]:
    """Synthetic dataset: streets x (full, short, natural), first ones saved."""
    d = create_dataset_dir(tmp_path)
    records: list[dict] = []
    made = 0
    for i, street in enumerate(streets, 1):
        for variant in VARIANTS:
            if made < n_saved:
                duration = make_wav(d / audio_relpath(record_id(i, variant)))
                rec = saved_record(i, street, variant, street, duration)
            elif made < n_saved + n_skipped:
                rec = skipped_record(i, street, variant)
            else:
                break
            append_manifest(d / "manifest.jsonl", rec)
            records.append(rec)
            made += 1
    write_dataset_json(d, build_dataset_meta(STREETS_FILE, list(streets), records))
    return d, records


# --- streets.txt parsing ------------------------------------------------------


def test_current_streets_file_is_parsed_in_file_order():
    streets, excluded = read_streets_file(STREETS_FILE)
    assert streets, "streets.txt пуст"
    assert all(s.strip() == s and s for s in streets)
    assert excluded == [] or all(e["reason"] for e in excluded)
    # order and spelling are taken verbatim from the file
    raw = [ln.strip() for ln in STREETS_FILE.read_text(encoding="utf-8").splitlines()]
    expected = [ln for ln in raw if ln and not ln.startswith("Страница")]
    assert streets == expected
    assert len(streets) == len(set(streets))
    assert not any(s.startswith("Страница") for s in streets)


def test_parser_skips_blank_lines_and_page_heading(tmp_path):
    f = tmp_path / "streets.txt"
    f.write_text("Ленина\n\nСтраница 2 (улицы 91–107)\n\nГагарина\n", encoding="utf-8")
    streets, excluded = read_streets_file(f)
    assert streets == ["Ленина", "Гагарина"]
    reasons = {e["reason"] for e in excluded}
    assert reasons == {"empty line", "page heading"}


def test_parser_rejects_duplicates(tmp_path):
    f = tmp_path / "streets.txt"
    f.write_text("Искра\nИскра\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Дубликат"):
        read_streets_file(f)


def test_parser_rejects_invalid_characters(tmp_path):
    f = tmp_path / "streets.txt"
    f.write_text("Ленина\nЛенина 2-я; drop table\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Недопустимые символы"):
        read_streets_file(f)


def test_parser_rejects_empty_file(tmp_path):
    f = tmp_path / "streets.txt"
    f.write_text("\n\n", encoding="utf-8")
    with pytest.raises(ValueError, match="пуст"):
        read_streets_file(f)


def test_type_words_are_never_added_to_full_reference():
    streets = ["Искра", "Файзрахмана Хисматуллина"]
    plan = build_plan(streets)
    full = [p for p in plan if p.variant == "full"]
    assert [p.reference_raw for p in full] == streets
    assert not any("улица" in p.reference_raw.lower() for p in full)


# --- normalization ------------------------------------------------------------


def test_normalize_reference_lowercase_and_punctuation():
    assert normalize_reference("До Хисматуллина, дом 31!") == "до хисматуллина дом тридцать один"


def test_normalize_reference_reviewed_street_numerals():
    assert normalize_reference("улица 40 лет Победы") == "улица сорок лет победы"
    assert normalize_reference("65 лет Победы") == normalize_reference(
        FORMS_NUMERALS["65"] + " лет Победы")


def test_normalize_reference_keeps_russian_numerals_as_words():
    assert normalize_reference("Сорок лет Победы") == "сорок лет победы"


def test_normalize_reference_keeps_yo():
    assert normalize_reference("Весёлая") == "весёлая"


@pytest.mark.parametrize("n, expected", [
    (0, "ноль"),
    (7, "семь"),
    (15, "пятнадцать"),
    (31, "тридцать один"),
    (40, "сорок"),
    (65, "шестьдесят пять"),
    (100, "сто"),
    (107, "сто семь"),
    (1000, "одна тысяча"),
    (2000, "две тысячи"),
    (5000, "пять тысяч"),
    (12000, "двенадцать тысяч"),
    (21500, "двадцать одна тысяча пятьсот"),
])
def test_number_to_words(n, expected):
    assert number_to_words(n) == expected


def test_number_to_words_range_guard():
    with pytest.raises(ValueError):
        number_to_words(1_000_000)


def test_digits_to_words_prefers_reviewed_street_numerals():
    assert digits_to_words("40") == FORMS_NUMERALS["40"]
    assert digits_to_words("дом 31") == "дом тридцать один"

# --- ids / plan / short suggestion -------------------------------------------


def test_record_id_and_audio_relpath():
    assert record_id(1, "full") == "street_001_full"
    assert record_id(107, "natural") == "street_107_natural"
    assert audio_relpath("street_001_full") == "audio/street_001_full.wav"
    assert audio_relpath("street_001_full").isascii()


def test_build_plan_matches_streets_order_and_variants():
    streets = ["Абзелиловская", "Ак Кайын", "Файзрахмана Хисматуллина"]
    plan = build_plan(streets)
    assert len(plan) == 3 * len(VARIANTS)
    assert [p.variant for p in plan[:3]] == list(VARIANTS)
    assert [p.street_index for p in plan[:3]] == [1, 1, 1]
    assert [p.street for p in plan] == [s for s in streets for _ in VARIANTS]
    assert plan[3].id == "street_002_full"


def test_plan_has_no_generated_reference_for_short_and_natural():
    """SHORT/NATURAL must not be generated: no text is derived from the street."""
    plan = build_plan(["Абзелиловская", "Файзрахмана Хисматуллина"])
    full = [p for p in plan if p.variant == "full"]
    short = [p for p in plan if p.variant == "short"]
    natural = [p for p in plan if p.variant == "natural"]
    assert [p.reference_raw for p in full] == ["Абзелиловская", "Файзрахмана Хисматуллина"]
    assert all(p.reference_raw == "" for p in short + natural)


def test_plan_never_contains_grammatically_generated_phrases():
    """Regression: "До Абзелиловская" and similar forms must never be produced."""
    streets = ["Абзелиловская", "Ленина", "Искра", "Сорок лет Победы"]
    for p in build_plan(streets):
        assert not p.reference_raw.lower().startswith("до ")
        assert p.reference_raw == "" or p.reference_raw == p.street
        assert "абзелиловская" not in p.reference_raw.lower() or p.reference_raw == p.street


def test_prompt_constants_match_spec():
    assert PROMPT_SHORT == "Введите короткий вариант или - чтобы пропустить:"
    assert PROMPT_NATURAL == "Введите реальную фразу, которую вы будете произносить:"
    assert SKIP_TOKENS == ("", "-")


@pytest.mark.parametrize("street", ["Сорок лет Победы", "Шестьдесят пять лет Победы",
                                    "Семьдесят лет Октября", "Файзрахмана Хисматуллина"])
def test_full_reference_is_the_verbatim_street_line(street):
    plan = build_plan([street])
    assert plan[0].reference_raw == street
    assert plan[0].reference_raw == plan[0].street


def test_iskra_and_iskra_lane_are_distinct():
    # the current file has one "Искра"; if a lane appears later both stay distinct
    streets = ["Искра", "Переулок Искра"]
    plan = build_plan(streets)
    assert plan[0].reference_raw == "Искра"
    assert plan[3].reference_raw == "Переулок Искра"
    assert plan[0].id != plan[3].id


# --- manifest / resume --------------------------------------------------------


def test_manifest_roundtrip_and_csv(tmp_path):
    d, records = make_dataset(tmp_path, n_saved=2, n_skipped=1)
    assert read_manifest(d / "manifest.jsonl") == records
    write_manifest_csv(d / "manifest.csv", records)
    rows = list(csv.DictReader((d / "manifest.csv").open(encoding="utf-8")))
    assert len(rows) == len(records)
    assert rows[0]["id"] == "street_001_full"
    assert set(rows[0]) >= {"id", "street", "variant", "reference", "audio"}


def test_manifest_state_detects_duplicate_ids(tmp_path):
    d, records = make_dataset(tmp_path, n_saved=1, n_skipped=0)
    append_manifest(d / "manifest.jsonl", records[0])
    with pytest.raises(ValueError, match="дубликат id"):
        manifest_state(read_manifest(d / "manifest.jsonl"))


def test_pending_records_resume_from_first_missing():
    streets = ["Ленина", "Гагарина"]
    plan = build_plan(streets)
    state = manifest_state([r for r in [
        saved_record(1, "Ленина", "full", "Ленина", 1.0),
        skipped_record(1, "Ленина", "short"),
    ]])
    pending = pending_records(plan, state)
    assert pending[0].id == "street_001_natural"
    assert len(pending) == len(plan) - 2


def test_pending_records_is_empty_when_all_handled():
    plan = build_plan(["Искра"])
    records = [saved_record(1, "Искра", "full", "Искра", 1.0),
               skipped_record(1, "Искра", "short"),
               skipped_record(1, "Искра", "natural")]
    assert pending_records(plan, manifest_state(records)) == []


# --- dataset dirs -------------------------------------------------------------


def test_create_dataset_dir_is_unique(tmp_path):
    a = create_dataset_dir(tmp_path)
    b = create_dataset_dir(tmp_path)
    assert a != b
    assert a.name.startswith("real_dataset_") and b.name.startswith("real_dataset_")
    assert (a / "audio").is_dir() and (b / "audio").is_dir()


def test_find_latest_dataset(tmp_path):
    assert find_latest_dataset(tmp_path) is None
    first = create_dataset_dir(tmp_path)
    (tmp_path / "unrelated").mkdir()
    assert find_latest_dataset(tmp_path) == first
    newer = create_dataset_dir(tmp_path)
    assert find_latest_dataset(tmp_path) == newer


def test_sha256_file(tmp_path):
    f = tmp_path / "streets.txt"
    f.write_text("Ленина\n", encoding="utf-8")
    assert len(sha256_file(f)) == 64
    assert sha256_file(STREETS_FILE) == sha256_file(STREETS_FILE)

# --- validation ---------------------------------------------------------------


def test_validate_ok_dataset(tmp_path):
    d, records = make_dataset(tmp_path, n_saved=3, n_skipped=1)
    report = validate_dataset(d)
    assert report["ok"], report["errors"]
    assert report["stats"] == {"expected": 6, "saved": 3, "skipped": 1, "pending": 2}


def test_validate_skipped_records_are_not_errors(tmp_path):
    d, _ = make_dataset(tmp_path, n_saved=1, n_skipped=1)
    report = validate_dataset(d)
    assert report["ok"], report["errors"]
    assert report["stats"]["skipped"] == 1


def test_validate_detects_wrong_sample_rate(tmp_path):
    d, records = make_dataset(tmp_path, n_saved=1, n_skipped=0)
    target = d / records[0]["audio"]
    make_wav(target, rate=16_000)  # same duration, wrong format
    report = validate_dataset(d)
    assert not report["ok"]
    assert any("sample_rate" in e and "16000" in e for e in report["errors"])


def test_validate_detects_missing_and_unreadable_wav(tmp_path):
    d, records = make_dataset(tmp_path, n_saved=2, n_skipped=0)
    (d / records[0]["audio"]).unlink()
    (d / records[1]["audio"]).write_bytes(b"not a wav")
    report = validate_dataset(d)
    assert not report["ok"]
    assert any("не найден" in e for e in report["errors"])
    assert any("не читается" in e for e in report["errors"])


def test_validate_detects_truncated_wav(tmp_path):
    """A truncated WAV keeps the original length in its header, so the real
    file size must be checked, not only the header fields."""
    d, records = make_dataset(tmp_path, n_saved=1, n_skipped=0)
    wav = d / records[0]["audio"]
    wav.write_bytes(wav.read_bytes()[:44])
    report = validate_dataset(d)
    assert not report["ok"]
    assert any("обрезан" in e for e in report["errors"])


def test_validate_detects_empty_wav(tmp_path):
    """Header declares zero-length audio data (silent corruption)."""
    d, records = make_dataset(tmp_path, n_saved=1, n_skipped=0)
    wav = d / records[0]["audio"]
    header = bytearray(wav.read_bytes()[:44])
    header[40:44] = (0).to_bytes(4, "little")  # data chunk size = 0
    wav.write_bytes(bytes(header))
    report = validate_dataset(d)
    assert not report["ok"]
    assert any("нулевая длительность" in e for e in report["errors"]), report["errors"]


def test_validate_detects_duplicate_id(tmp_path):
    d, records = make_dataset(tmp_path, n_saved=1, n_skipped=0)
    append_manifest(d / "manifest.jsonl", records[0])
    report = validate_dataset(d)
    assert not report["ok"]
    assert any("дубликат id" in e for e in report["errors"])


def test_validate_detects_empty_reference(tmp_path):
    d, records = make_dataset(tmp_path, n_saved=1, n_skipped=0)
    bad = dict(records[0], reference_raw="", reference="")
    (d / "manifest.jsonl").unlink()
    append_manifest(d / "manifest.jsonl", bad)
    report = validate_dataset(d)
    assert not report["ok"]
    assert any("пустой reference_raw" in e for e in report["errors"])
    assert any("пустой reference" in e for e in report["errors"])


def test_validate_detects_skipped_with_audio(tmp_path):
    d, records = make_dataset(tmp_path, n_saved=1, n_skipped=0)
    bad = dict(skipped_record(1, "Ленина", "short"), audio="audio/street_001_full.wav")
    append_manifest(d / "manifest.jsonl", bad)
    report = validate_dataset(d)
    assert not report["ok"]
    assert any("skipped-запись со ссылкой" in e for e in report["errors"])


def test_validate_detects_out_of_plan_record(tmp_path):
    d, records = make_dataset(tmp_path, n_saved=1, n_skipped=0)
    append_manifest(d / "manifest.jsonl", saved_record(999, "Неизвестная", "full",
                                                       "Неизвестная", 1.0))
    report = validate_dataset(d)
    assert not report["ok"]
    assert any("вне плана" in e for e in report["errors"])


def test_validate_detects_plan_mismatch(tmp_path):
    d, records = make_dataset(tmp_path, n_saved=1, n_skipped=0)
    bad = dict(records[0], street="Другая улица")
    (d / "manifest.jsonl").unlink()
    append_manifest(d / "manifest.jsonl", bad)
    report = validate_dataset(d)
    assert not report["ok"]
    assert any("не соответствуют плану" in e for e in report["errors"])


def test_validate_warns_on_extra_wav(tmp_path):
    d, _ = make_dataset(tmp_path, n_saved=1, n_skipped=0)
    make_wav(d / "audio" / "street_042_full.wav")
    report = validate_dataset(d)
    assert report["ok"], report["errors"]
    assert any("лишний WAV" in w for w in report["warnings"])


def test_validate_missing_manifest_and_dir(tmp_path):
    d = create_dataset_dir(tmp_path)
    report = validate_dataset(d)
    assert not report["ok"]
    assert any("manifest.jsonl не найден" in e for e in report["errors"])
    assert any("dataset.json не найден" in w for w in report["warnings"])

    missing = validate_dataset(tmp_path / "nope")
    assert not missing["ok"]
    assert any("каталог не найден" in e for e in missing["errors"])


def test_validate_detects_bad_json_line(tmp_path):
    d, _ = make_dataset(tmp_path, n_saved=1, n_skipped=0)
    with (d / "manifest.jsonl").open("a", encoding="utf-8") as f:
        f.write("{not json}\n")
    report = validate_dataset(d)
    assert not report["ok"]
    assert any("невалидный JSON" in e for e in report["errors"])


def test_wav_info_reads_saved_format(tmp_path):
    path = tmp_path / "a.wav"
    duration = make_wav(path, seconds=0.5)
    info = wav_info(path)
    assert info["sample_rate"] == TARGET_RATE
    assert info["channels"] == 1
    assert info["sample_width_bytes"] == 2
    assert info["comptype"] == "NONE"
    assert round(info["frames"] / info["sample_rate"], 3) == duration


def test_dataset_json_has_provenance(tmp_path):
    d, _ = make_dataset(tmp_path, n_saved=1, n_skipped=0)
    meta = json.loads((d / "dataset.json").read_text(encoding="utf-8"))
    assert meta["streets_sha256"] == sha256_file(STREETS_FILE)
    assert meta["source_streets"] == ["Ленина", "Гагарина"]
    assert meta["expected_records"] == 6
    assert meta["audio_format"]["sample_rate"] == TARGET_RATE
    assert "NOT for fine-tuning" in meta["purpose"]
    assert "hotword" in meta["purpose"]



# --- CLI parsing --------------------------------------------------------------


def test_recorder_cli_defaults():
    from scripts.record_real_dataset import DEFAULT_STREETS, parse_args

    args = parse_args([])
    assert args.new is False
    assert args.dataset is None
    assert args.streets == DEFAULT_STREETS
    assert args.input_rate == 48000
    assert args.limit is None


def test_recorder_cli_flags(tmp_path):
    from scripts.record_real_dataset import parse_args

    args = parse_args(["--new", "--limit", "2", "--input-rate", "16000",
                       "--dataset", str(tmp_path), "--streets", str(tmp_path / "s.txt")])
    assert args.new is True and args.limit == 2 and args.input_rate == 16000
    assert args.dataset == tmp_path


def test_recorder_resolve_dataset_new_and_resume(tmp_path):
    from scripts.record_real_dataset import parse_args, resolve_dataset

    streets = tmp_path / "streets.txt"
    streets.write_text("Ленина\nГагарина\n", encoding="utf-8")
    results = tmp_path / "results"

    args = parse_args(["--streets", str(streets), "--results-dir", str(results)])
    ctx = resolve_dataset(args)
    assert ctx.dataset_dir.is_dir() and ctx.dataset_dir.parent == results
    assert ctx.streets == ["Ленина", "Гагарина"]

    # second run without --new resumes the same dir
    assert resolve_dataset(args).dataset_dir == ctx.dataset_dir

    # --new forces a fresh dir
    fresh = resolve_dataset(parse_args(["--new", "--streets", str(streets),
                                        "--results-dir", str(results)]))
    assert fresh.dataset_dir != ctx.dataset_dir


def test_recorder_resolve_dataset_limit(tmp_path):
    from scripts.record_real_dataset import parse_args, resolve_dataset

    streets = tmp_path / "streets.txt"
    streets.write_text("Ленина\nГагарина\nИскра\n", encoding="utf-8")
    args = parse_args(["--streets", str(streets), "--results-dir", str(tmp_path / "r"),
                       "--limit", "2"])
    assert resolve_dataset(args).streets == ["Ленина", "Гагарина"]


def test_validator_cli_targets(tmp_path):
    from scripts.validate_real_dataset import parse_args, resolve_targets

    args = parse_args(["--json"])
    assert args.datasets == [] and args.json is True

    explicit = tmp_path / "real_dataset_x"
    assert resolve_targets(parse_args([str(explicit)])) == [explicit]

    results = tmp_path / "results"
    d = create_dataset_dir(results)
    assert resolve_targets(parse_args(["--results-dir", str(results)])) == [d]


def test_validator_main_exit_codes(tmp_path, capsys):
    from scripts.validate_real_dataset import main

    results = tmp_path / "results"
    d, _ = make_dataset(results, n_saved=2, n_skipped=1)
    assert main([str(d)]) == 0
    assert "Статус  : OK" in capsys.readouterr().out

    (d / "audio" / "street_001_full.wav").unlink()
    assert main([str(d)]) == 1

    empty = tmp_path / "empty"
    empty.mkdir()
    assert main(["--results-dir", str(empty)]) == 1


def test_validator_main_json_report(tmp_path, capsys):
    from scripts.validate_real_dataset import main

    d, _ = make_dataset(tmp_path / "results", n_saved=1, n_skipped=1)
    assert main([str(d), "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report[0]["ok"] is True
    assert report[0]["stats"]["saved"] == 1


def test_tone_audio_format_matches_pcm16_mono_8k(tmp_path):
    """Saved audio must be exactly what T-one expects (mono 8 kHz PCM16)."""
    d, records = make_dataset(tmp_path, n_saved=1, n_skipped=0)
    info = wav_info(d / records[0]["audio"])
    assert (info["channels"], info["sample_rate"], info["sample_width_bytes"]) == (1, 8000, 2)
    assert records[0]["duration_sec"] == round(info["frames"] / info["sample_rate"], 3)


# --- recorder flow (no real microphone) ---------------------------------------


class FakeRecorder:
    """Stands in for ManualRecorder: returns a synthetic tone, no audio devices."""

    def __init__(self, seconds: float = 0.8, rate: int = 48_000) -> None:
        self.rate = rate
        self._seconds = seconds
        self.calls = 0

    def start(self) -> None:
        pass

    def stop(self) -> np.ndarray:
        self.calls += 1
        n = int(self._seconds * self.rate)
        t = np.arange(n) / self.rate
        return (np.sin(2 * np.pi * 300 * t) * 0.4).astype(np.float32)

    def close(self) -> None:
        pass


def make_ctx(tmp_path: Path, streets: list[str]):
    from scripts.record_real_dataset import DatasetContext

    return DatasetContext(dataset_dir=create_dataset_dir(tmp_path),
                          streets_path=STREETS_FILE, streets=streets)


def feed_inputs(monkeypatch, answers: list[str]):
    """Feed scripted answers to input(); fail if the flow asks for more."""
    from scripts import record_real_dataset as mod

    it = iter(answers)

    def fake_input(prompt: str = "") -> str:
        try:
            return next(it)
        except StopIteration:  # pragma: no cover - guards against flow changes
            raise AssertionError(f"лишний input(): {prompt!r}")

    monkeypatch.setattr("builtins.input", fake_input)
    monkeypatch.setattr(mod, "play", lambda pcm: None)


def test_recorder_flow_saves_full_record(tmp_path, monkeypatch):
    from scripts.record_real_dataset import process_plan_record

    ctx = make_ctx(tmp_path, ["Ленина"])
    rec = build_plan(["Ленина"])[0]
    feed_inputs(monkeypatch, ["", "", ""])  # start, stop, save
    process_plan_record(FakeRecorder(), ctx, rec)

    record = read_manifest(ctx.manifest_path)[0]
    assert record["id"] == "street_001_full"
    assert record["status"] == "saved"
    assert record["reference_raw"] == "Ленина"
    assert record["reference"] == "ленина"
    assert record["sample_rate"] == TARGET_RATE and record["channels"] == 1
    assert record["duration_sec"] > 0
    info = wav_info(ctx.dataset_dir / record["audio"])
    assert (info["sample_rate"], info["channels"], info["sample_width_bytes"]) == (8000, 1, 2)
    meta = json.loads((ctx.dataset_dir / "dataset.json").read_text(encoding="utf-8"))
    assert meta["saved"] == 1 and meta["pending"] == 2


def test_recorder_flow_skips_natural_on_dash(tmp_path, monkeypatch):
    from scripts.record_real_dataset import process_plan_record

    ctx = make_ctx(tmp_path, ["Ленина"])
    rec = build_plan(["Ленина"])[2]  # natural
    feed_inputs(monkeypatch, ["-"])
    process_plan_record(FakeRecorder(), ctx, rec)

    record = read_manifest(ctx.manifest_path)[0]
    assert record["status"] == "skipped" and record["audio"] is None
    assert not (ctx.dataset_dir / "audio" / "street_001_natural.wav").exists()


def test_recorder_flow_skips_short_on_enter(tmp_path, monkeypatch):
    """ENTER at the SHORT prompt means skip (no auto-generated short variant)."""
    from scripts.record_real_dataset import process_plan_record

    ctx = make_ctx(tmp_path, ["Ленина"])
    rec = build_plan(["Ленина"])[1]  # short
    feed_inputs(monkeypatch, [""])
    process_plan_record(FakeRecorder(), ctx, rec)

    record = read_manifest(ctx.manifest_path)[0]
    assert record["status"] == "skipped"
    assert record["audio"] is None
    assert record["reference_raw"] == ""


def test_recorder_flow_skips_natural_on_enter(tmp_path, monkeypatch):
    from scripts.record_real_dataset import process_plan_record

    ctx = make_ctx(tmp_path, ["Ленина"])
    rec = build_plan(["Ленина"])[2]  # natural
    feed_inputs(monkeypatch, [""])
    process_plan_record(FakeRecorder(), ctx, rec)

    record = read_manifest(ctx.manifest_path)[0]
    assert record["status"] == "skipped" and record["reference"] == ""


def test_recorder_flow_uses_typed_short_reference(tmp_path, monkeypatch):
    from scripts.record_real_dataset import process_plan_record

    ctx = make_ctx(tmp_path, ["Файзрахмана Хисматуллина"])
    rec = [p for p in build_plan(["Файзрахмана Хисматуллина"]) if p.variant == "short"][0]
    feed_inputs(monkeypatch, ["Хисматуллина", "", "", ""])  # text, start, stop, save
    process_plan_record(FakeRecorder(), ctx, rec)

    record = read_manifest(ctx.manifest_path)[0]
    assert record["variant"] == "short"
    assert record["reference_raw"] == "Хисматуллина"
    assert record["reference"] == "хисматуллина"


def test_recorder_flow_accepts_typed_natural_reference(tmp_path, monkeypatch, capsys):
    from scripts.record_real_dataset import process_plan_record

    ctx = make_ctx(tmp_path, ["Ленина"])
    rec = [p for p in build_plan(["Ленина"]) if p.variant == "natural"][0]
    feed_inputs(monkeypatch, ["До Ленина дом 31", "", "", ""])  # text, start, stop, save
    process_plan_record(FakeRecorder(), ctx, rec)

    record = read_manifest(ctx.manifest_path)[0]
    assert record["variant"] == "natural"
    assert record["reference_raw"] == "До Ленина дом 31"
    assert record["reference"] == "до ленина дом тридцать один"
    out = capsys.readouterr().out
    assert "NATURAL: До Ленина дом 31" in out
    assert "Произнесите ровно эту фразу." in out
    assert "Проверьте падеж" in out


def test_recorder_flow_natural_reference_is_never_the_street_name(tmp_path, monkeypatch):
    """The stored reference must be what the user typed, not the street line."""
    from scripts.record_real_dataset import process_plan_record

    ctx = make_ctx(tmp_path, ["Абзелиловская"])
    rec = [p for p in build_plan(["Абзелиловская"]) if p.variant == "natural"][0]
    feed_inputs(monkeypatch, ["До Абзелиловской", "", "", ""])
    process_plan_record(FakeRecorder(), ctx, rec)

    record = read_manifest(ctx.manifest_path)[0]
    assert record["reference_raw"] == "До Абзелиловской"
    assert record["reference"] == "до абзелиловской"
    assert record["street"] == "Абзелиловская"


def test_recorder_flow_rerecords_then_edits_then_saves(tmp_path, monkeypatch):
    from scripts.record_real_dataset import process_plan_record

    ctx = make_ctx(tmp_path, ["Ленина"])
    rec = [p for p in build_plan(["Ленина"]) if p.variant == "short"][0]
    # text, start, stop, r, start, stop, e, new text, start, stop, save
    feed_inputs(monkeypatch, ["Ленинградская", "", "", "r", "", "",
                              "e", "Ленинская", "", "", ""])
    recorder = FakeRecorder()
    process_plan_record(recorder, ctx, rec)

    assert recorder.calls == 3  # re-record + edited take
    record = read_manifest(ctx.manifest_path)[0]
    assert record["reference_raw"] == "Ленинская"
    assert record["status"] == "saved"


def test_recorder_flow_full_reference_cannot_be_edited(tmp_path, monkeypatch, capsys):
    """FULL keeps the verbatim streets.txt line even if 'e' is pressed."""
    from scripts.record_real_dataset import process_plan_record

    ctx = make_ctx(tmp_path, ["Ленина"])
    rec = build_plan(["Ленина"])[0]  # full
    feed_inputs(monkeypatch, ["", "", "e", "", "", ""])  # start, stop, e, start, stop, save
    process_plan_record(FakeRecorder(), ctx, rec)

    record = read_manifest(ctx.manifest_path)[0]
    assert record["reference_raw"] == "Ленина"
    assert "не редактируется" in capsys.readouterr().out


def test_recorder_flow_quit_saves_nothing(tmp_path, monkeypatch):
    from scripts.record_real_dataset import QuitSession, process_plan_record

    ctx = make_ctx(tmp_path, ["Ленина"])
    rec = build_plan(["Ленина"])[0]
    feed_inputs(monkeypatch, ["", "", "q"])
    with pytest.raises(QuitSession):
        process_plan_record(FakeRecorder(), ctx, rec)
    assert not ctx.manifest_path.exists()


def test_recorder_main_without_mic_when_all_done(tmp_path, capsys):
    """main() must not touch the microphone when nothing is pending."""
    from scripts.record_real_dataset import main

    streets = tmp_path / "streets.txt"
    streets.write_text("Ленина\n", encoding="utf-8")
    results = tmp_path / "results"
    d = create_dataset_dir(results)
    for rec in build_plan(["Ленина"]):
        append_manifest(d / "manifest.jsonl",
                        skipped_record(rec.street_index, rec.street, rec.variant))
    write_dataset_json(d, build_dataset_meta(streets, ["Ленина"],
                                             read_manifest(d / "manifest.jsonl")))

    assert main(["--streets", str(streets), "--results-dir", str(results)]) == 0
    assert "Все записи уже сделаны" in capsys.readouterr().out


def test_recorder_main_resumes_without_duplicating(tmp_path, monkeypatch):
    """Second run continues from the first missing record (resume)."""
    from scripts import record_real_dataset as mod

    streets = tmp_path / "streets.txt"
    streets.write_text("Ленина\nГагарина\n", encoding="utf-8")
    results = tmp_path / "results"
    argv = ["--streets", str(streets), "--results-dir", str(results)]

    monkeypatch.setattr(mod, "ManualRecorder", lambda rate: FakeRecorder())
    feed_inputs(monkeypatch, ["", "", "s",   # FULL: start, stop, skip
                              "",            # SHORT: ENTER -> skip (no auto text)
                              "q"])          # NATURAL: quit at the text prompt
    assert mod.main(argv) == 0

    ctx = mod.resolve_dataset(mod.parse_args(argv))
    first = read_manifest(ctx.manifest_path)
    assert [r["id"] for r in first] == ["street_001_full", "street_001_short"]
    assert [r["status"] for r in first] == ["skipped", "skipped"]

    feed_inputs(monkeypatch, ["-",                   # NATURAL -> skip
                              "", "", "q"])          # FULL(2): start, stop, quit
    assert mod.main(argv) == 0
    ctx2 = mod.resolve_dataset(mod.parse_args(argv))
    assert ctx2.dataset_dir == ctx.dataset_dir           # same dataset, not a new one
    ids = [r["id"] for r in read_manifest(ctx2.manifest_path)]
    assert ids == ["street_001_full", "street_001_short", "street_001_natural"]
    assert len(ids) == len(set(ids))


def test_recorder_flow_quit_at_start_prompt(tmp_path, monkeypatch):
    from scripts.record_real_dataset import QuitSession, process_plan_record

    ctx = make_ctx(tmp_path, ["Ленина"])
    rec = build_plan(["Ленина"])[0]
    feed_inputs(monkeypatch, ["q"])
    with pytest.raises(QuitSession):
        process_plan_record(FakeRecorder(), ctx, rec)
    assert not ctx.manifest_path.exists()


def test_recorder_flow_cancel_take_with_q(tmp_path, monkeypatch):
    from scripts.record_real_dataset import QuitSession, process_plan_record

    ctx = make_ctx(tmp_path, ["Ленина"])
    rec = build_plan(["Ленина"])[0]
    feed_inputs(monkeypatch, ["", "q"])  # start, then q instead of stop
    with pytest.raises(QuitSession):
        process_plan_record(FakeRecorder(), ctx, rec)
    assert not ctx.manifest_path.exists()


def test_recorder_flow_quit_at_text_prompt(tmp_path, monkeypatch):
    from scripts.record_real_dataset import QuitSession, process_plan_record

    ctx = make_ctx(tmp_path, ["Ленина"])
    rec = [p for p in build_plan(["Ленина"]) if p.variant == "short"][0]
    feed_inputs(monkeypatch, ["q"])
    with pytest.raises(QuitSession):
        process_plan_record(FakeRecorder(), ctx, rec)
    assert not ctx.manifest_path.exists()


def test_recorder_main_handles_closed_stdin(tmp_path, monkeypatch, capsys):
    """EOFError (piped stdin) must end the session gracefully, not crash."""
    from scripts import record_real_dataset as mod

    streets = tmp_path / "streets.txt"
    streets.write_text("Ленина\n", encoding="utf-8")
    results = tmp_path / "results"
    monkeypatch.setattr(mod, "ManualRecorder", lambda rate: FakeRecorder())

    def eof(prompt: str = "") -> str:
        raise EOFError

    monkeypatch.setattr("builtins.input", eof)
    assert mod.main(["--streets", str(streets), "--results-dir", str(results)]) == 0
    assert "Осталось" in capsys.readouterr().out


def test_pcm16_8k_conversion():
    from scripts.record_real_dataset import to_pcm16_8k

    audio = FakeRecorder(seconds=0.5, rate=48_000).stop()
    pcm = to_pcm16_8k(audio, 48_000)
    assert pcm.dtype == np.int16
    assert abs(len(pcm) - 4000) < 10  # 0.5 s at 8 kHz
    assert np.abs(pcm).max() > 1000

    native = FakeRecorder(seconds=0.5, rate=TARGET_RATE).stop()
    assert len(to_pcm16_8k(native, TARGET_RATE)) == len(native)
