"""Pure logic for the manual real-world evaluation dataset (t-one-train).

streets.txt is the ONLY source of streets: a FULL reference is the verbatim
line from the file, in file order; type words ("улица"/"переулок"/...) are
NEVER added automatically. SHORT and NATURAL references are NOT generated in
any way (no abbreviations, no declension) — the user types them before
recording, so a reference always matches the phrase actually spoken. The
dataset is an evaluation-only set: it is not used for fine-tuning and it is NOT
a hotword source (hotwords are built from streets.txt separately and evaluated
against this dataset).

This module has no microphone/sounddevice imports so tests can run headless.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import wave
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from src.t_one_train.forms import NUMERALS, normalize

VARIANTS = ("full", "short", "natural")
TYPE_WORDS = ("улица", "переулок")
TARGET_RATE = 8_000
CHANNELS = 1
SAMPLE_WIDTH_BYTES = 2  # 16-bit PCM

PAGE_HEADING_RE = re.compile(r"Страница \d+ \(улицы \d+[–-]\d+\)")
VALID_CHARS_RE = re.compile(r"[А-Яа-яЁё0-9 -]+")

# SHORT/NATURAL prompts (shared by the recorder and its tests). The user types a
# real phrase; ENTER or "-" skips the record. Nothing is pre-filled.
PROMPT_SHORT = "Введите короткий вариант или - чтобы пропустить:"
PROMPT_NATURAL = "Введите реальную фразу, которую вы будете произносить:"
SKIP_TOKENS = ("", "-")

MANIFEST_FIELDS = [
    "id",
    "street_index",
    "street",
    "variant",
    "status",
    "reference_raw",
    "reference",
    "audio",
    "sample_rate",
    "channels",
    "duration_sec",
]

_DIGIT_RE = re.compile(r"\d+")

_UNITS_M = ["", "один", "два", "три", "четыре", "пять", "шесть", "семь", "восемь", "девять"]
_UNITS_F = ["", "одна", "две", "три", "четыре", "пять", "шесть", "семь", "восемь", "девять"]
_TEENS = ["десять", "одиннадцать", "двенадцать", "тринадцать", "четырнадцать",
          "пятнадцать", "шестнадцать", "семнадцать", "восемнадцать", "девятнадцать"]
_TENS = ["", "", "двадцать", "тридцать", "сорок", "пятьдесят",
         "шестьдесят", "семьдесят", "восемьдесят", "девяносто"]
_HUNDREDS = ["", "сто", "двести", "триста", "четыреста", "пятьсот",
             "шестьсот", "семьсот", "восемьсот", "девятьсот"]
# тысяча: 1 -> тысяча, 2-4 -> тысячи, 0/5+ -> тысяч
_THOUSAND_FORMS = {1: "тысяча", 2: "тысячи", 3: "тысячи", 4: "тысячи"}


def read_streets_file(path: str | Path) -> tuple[list[str], list[dict]]:
    """Parse the CURRENT streets.txt: verbatim street names in file order.

    - UTF-8, per-line strip(), blank lines skipped;
    - page headings ("Страница N (улицы M–K)") are skipped and reported;
    - only Russian/Latin digits/spaces/hyphen characters are allowed;
    - duplicates raise ValueError.
    Returns (streets, excluded) where excluded = [{"line", "text", "reason"}].
    """
    streets: list[str] = []
    excluded: list[dict] = []
    seen: set[str] = set()
    for lineno, raw in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line:
            excluded.append({"line": lineno, "text": raw, "reason": "empty line"})
            continue
        if PAGE_HEADING_RE.fullmatch(line):
            excluded.append({"line": lineno, "text": line, "reason": "page heading"})
            continue
        if not VALID_CHARS_RE.fullmatch(line):
            raise ValueError(f"Недопустимые символы в streets.txt строка {lineno}: {line!r}")
        if line in seen:
            raise ValueError(f"Дубликат улицы в streets.txt строка {lineno}: {line!r}")
        seen.add(line)
        streets.append(line)
    if not streets:
        raise ValueError("streets.txt пуст")
    return streets, excluded


def record_id(street_index: int, variant: str) -> str:
    """Stable ascii id: street_001_full."""
    return f"street_{street_index:03d}_{variant}"


def audio_relpath(rid: str) -> str:
    return f"audio/{rid}.wav"


def number_to_words(n: int) -> str:
    """Russian words for 0..999999 (enough for house numbers and street numerals).

    Reviewed street numerals from forms.NUMERALS are handled by digits_to_words()
    before this general fallback, so street spellings stay exactly as reviewed.
    """
    if n < 0 or n > 999_999:
        raise ValueError(f"Число вне поддерживаемого диапазона: {n}")
    if n == 0:
        return "ноль"

    def triple(x: int, feminine: bool) -> list[str]:
        words: list[str] = []
        hundreds, rest = divmod(x, 100)
        if hundreds:
            words.append(_HUNDREDS[hundreds])
        if rest >= 10 and rest < 20:
            words.append(_TEENS[rest - 10])
            return words
        tens, unit = divmod(rest, 10)
        if tens:
            words.append(_TENS[tens])
        if unit:
            words.append((_UNITS_F if feminine else _UNITS_M)[unit])
        return words

    words: list[str] = []
    thousands, rest = divmod(n, 1000)
    if thousands:
        words += triple(thousands, feminine=True)
        words.append(_THOUSAND_FORMS.get(thousands % 10, "тысяч")
                     if thousands % 100 not in range(11, 15) else "тысяч")
    words += triple(rest, feminine=False)
    return " ".join(words)


def digits_to_words(text: str) -> str:
    """Digits -> Russian words, T-one compatible (CTC labels contain no digits).

    Reviewed street numerals (forms.NUMERALS: 40/50/60/65/70) are mapped to
    their exact reviewed spellings; any other number (e.g. a house number in a
    NATURAL phrase: "До Хисматуллина дом 31") is expanded by number_to_words().
    """

    def repl(m: re.Match[str]) -> str:
        token = m[0]
        if token in NUMERALS:
            return NUMERALS[token]
        return number_to_words(int(token))

    return _DIGIT_RE.sub(repl, text)


def normalize_reference(text: str) -> str:
    """Normalize a reference for WER/CER comparison, T-one compatible:
    lowercase, digits -> words (reviewed street numerals first, general Russian
    numerals otherwise), punctuation removed, only [а-яё] + spaces.
    Russian numerals are never converted back to digits."""
    return normalize(digits_to_words(text))


@dataclass
class PlannedRecord:
    id: str
    street_index: int
    street: str
    variant: str
    reference_raw: str = ""  # "" = no text known yet; FULL carries the street line


def build_plan(streets: list[str]) -> list[PlannedRecord]:
    """Plan in streets.txt order: per street -> full, short, natural.

    Only FULL has a reference upfront (the verbatim streets.txt line).
    SHORT/NATURAL plans carry no text at all: the recorder asks the user for a
    real phrase before recording. Nothing is derived from the street name —
    automatic proposals produced ungrammatical phrases ("До Абзелиловская") and
    artificial abbreviations, so this module deliberately has no such logic.
    """
    plan: list[PlannedRecord] = []
    for i, street in enumerate(streets, 1):
        plan.append(PlannedRecord(record_id(i, "full"), i, street, "full", street))
        plan.append(PlannedRecord(record_id(i, "short"), i, street, "short"))
        plan.append(PlannedRecord(record_id(i, "natural"), i, street, "natural"))
    return plan


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# --- dataset directories ------------------------------------------------------

DATASET_PREFIX = "real_dataset_"


def find_latest_dataset(results_dir: str | Path) -> Path | None:
    """Latest results/real_dataset_<timestamp>/ dir, or None."""
    base = Path(results_dir)
    if not base.is_dir():
        return None
    dirs = [d for d in base.iterdir()
            if d.is_dir() and d.name.startswith(DATASET_PREFIX)]
    return max(dirs, key=lambda x: x.name) if dirs else None


def create_dataset_dir(results_dir: str | Path) -> Path:
    """Create a brand new unique dataset dir with an empty audio/ subdir."""
    base = Path(results_dir)
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    d = base / f"{DATASET_PREFIX}{stamp}"
    n = 1
    while d.exists():
        n += 1
        d = base / f"{DATASET_PREFIX}{stamp}-{n}"
    (d / "audio").mkdir(parents=True)
    return d


# --- manifest -----------------------------------------------------------------


def read_manifest(path: str | Path) -> list[dict]:
    records: list[dict] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def append_manifest(path: str | Path, record: dict) -> None:
    with Path(path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_manifest_csv(path: str | Path, records: list[dict]) -> None:
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for r in records:
            writer.writerow({k: r.get(k, "") for k in MANIFEST_FIELDS})


def manifest_state(records: list[dict]) -> dict[str, dict]:
    """id -> record; raises on missing or duplicate ids."""
    state: dict[str, dict] = {}
    for r in records:
        rid = r.get("id")
        if not rid:
            raise ValueError("manifest: запись без id")
        if rid in state:
            raise ValueError(f"manifest: дубликат id {rid!r}")
        state[rid] = r
    return state


def pending_records(plan: list[PlannedRecord], state: dict[str, dict]) -> list[PlannedRecord]:
    """Records not yet saved/skipped; resume starts at the first one."""
    done = {rid for rid, r in state.items() if r.get("status") in ("saved", "skipped")}
    return [p for p in plan if p.id not in done]


# --- dataset meta / README ----------------------------------------------------


def build_dataset_meta(
    streets_path: str | Path,
    streets: list[str],
    records: list[dict],
    *,
    created: str | None = None,
    excluded: list[dict] | None = None,
) -> dict:
    now = datetime.now().isoformat(timespec="seconds")
    saved = sum(1 for r in records if r.get("status") == "saved")
    skipped = sum(1 for r in records if r.get("status") == "skipped")
    expected = len(streets) * len(VARIANTS)
    return {
        "created": created or now,
        "updated": now,
        "purpose": (
            "independent ASR evaluation set; NOT for fine-tuning; "
            "NOT a hotword source (hotwords are built from streets.txt separately)"
        ),
        "streets_file": str(Path(streets_path).resolve()),
        "streets_sha256": sha256_file(streets_path),
        "street_count": len(streets),
        "excluded_street_lines": excluded or [],
        "variants": list(VARIANTS),
        "variant_reference_source": {
            "full": "verbatim streets.txt line; never modified, type words never added",
            "short": "typed by the user before recording; ENTER/'-' skips the record",
            "natural": "typed by the user before recording; ENTER/'-' skips the record",
        },
        "reference_generation": ("SHORT/NATURAL are never generated automatically "
                                 "(no abbreviation, no declension of street names)"),
        "expected_records": expected,
        "saved": saved,
        "skipped": skipped,
        "pending": expected - saved - skipped,
        "audio_format": {
            "sample_rate": TARGET_RATE,
            "channels": CHANNELS,
            "encoding": "PCM16",
            "capture": "manual ENTER start/stop, native mic rate resampled to 8 kHz, no augmentation",
        },
        "normalization": ("digits->words (reviewed street numerals from forms.NUMERALS, "
                          "general Russian numerals otherwise), lowercase, punctuation "
                          "removed, only [а-яё ]"),
        "source_streets": streets,
    }


def write_dataset_json(dataset_dir: Path, meta: dict) -> None:
    (dataset_dir / "dataset.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def write_dataset_readme(dataset_dir: Path, meta: dict) -> None:
    text = (
        f"# Real-world evaluation dataset ({meta['created']})\n"
        "\n"
        "Назначение: независимый ручной evaluation set для объективной оценки качества ASR (T-one).\n"
        "\n"
        "- НЕ используется для fine-tuning T-one.\n"
        "- НЕ является источником hotwords: hotwords строятся только из streets.txt.\n"
        "- Используется только для сравнения декодеров/моделей на одних и тех же WAV + reference;\n"
        "  WAV и reference остаются неизменными между экспериментами.\n"
        "\n"
        f"Источник улиц: {meta['streets_file']} (sha256 {meta['streets_sha256'][:12]}…), "
        f"улиц: {meta['street_count']}.\n"
        "Порядок и написание — ровно как в streets.txt; слова «улица»/«переулок» не добавляются.\n"
        "\n"
        "Варианты записей:\n"
        "- FULL — дословная строка из streets.txt; ничего не добавляется и не склоняется.\n"
        "- SHORT — короткий вариант, который пользователь вводит сам перед записью\n"
        "  (например «Хисматуллина»); ENTER или «-» — пропустить запись.\n"
        "- NATURAL — реальная фраза, которую пользователь вводит сам перед записью\n"
        "  (например «До Абзелиловской»); ENTER или «-» — пропустить запись.\n"
        "SHORT/NATURAL не генерируются программой: ни сокращений, ни склонений названий.\n"
        "reference всегда соответствует реально произнесённой фразе;\n"
        "пропущенные записи сохраняются со status=\"skipped\".\n"
        "\n"
        "Аудио: mono, 8000 Hz, PCM16 WAV; запись на native sample rate микрофона,\n"
        "перед сохранением resample в 8 kHz; без шумов, кодеков и усиления.\n"
        "Запись ручная: ENTER старт/стоп, без VAD.\n"
        "\n"
        f"Статистика: сохранено {meta['saved']}, пропущено {meta['skipped']}, "
        f"осталось {meta['pending']} из {meta['expected_records']}.\n"
        "\n"
        f"Валидация: uv run python scripts/validate_real_dataset.py {Path('results') / dataset_dir.name}\n"
    )
    (dataset_dir / "README.md").write_text(text, encoding="utf-8")


# --- validation ---------------------------------------------------------------


def wav_info(path: str | Path) -> dict:
    """Format info from the WAV header plus the real file size.

    `frames` comes from the header, so `file_bytes` is needed to detect
    truncated files (a truncated header still declares the original length).
    """
    p = Path(path)
    with wave.open(str(p), "rb") as f:
        return {
            "sample_rate": f.getframerate(),
            "channels": f.getnchannels(),
            "sample_width_bytes": f.getsampwidth(),
            "comptype": f.getcomptype(),
            "frames": f.getnframes(),
            "file_bytes": p.stat().st_size,
        }


def validate_dataset(dataset_dir: str | Path) -> dict:
    """Validate a real_dataset directory. Returns {"ok", "errors", "warnings", "stats"}.

    Skipped entries are not errors. Errors: manifest problems, duplicate ids,
    empty references, missing/corrupt WAVs, wrong format, plan mismatches.
    """
    d = Path(dataset_dir)
    errors: list[str] = []
    warnings: list[str] = []
    stats: dict[str, int | None] = {"expected": None, "saved": 0, "skipped": 0, "pending": None}

    if not d.is_dir():
        return {"ok": False, "errors": [f"каталог не найден: {d}"], "warnings": [], "stats": stats}

    records: list[dict] = []
    manifest_path = d / "manifest.jsonl"
    if manifest_path.exists():
        for n, line in enumerate(manifest_path.read_text(encoding="utf-8").splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                errors.append(f"manifest.jsonl строка {n}: невалидный JSON ({e})")
    else:
        errors.append("manifest.jsonl не найден")

    try:
        state = manifest_state(records)
    except ValueError as e:
        errors.append(str(e))
        state = {}

    # expected plan from dataset.json (self-contained provenance)
    expected_ids: dict[str, tuple[str, str]] = {}
    dataset_json = d / "dataset.json"
    meta: dict = {}
    if dataset_json.exists():
        try:
            meta = json.loads(dataset_json.read_text(encoding="utf-8"))
            for i, s in enumerate(meta.get("source_streets", []), 1):
                for v in VARIANTS:
                    expected_ids[record_id(i, v)] = (s, v)
        except json.JSONDecodeError as e:
            errors.append(f"dataset.json: невалидный JSON ({e})")
    else:
        warnings.append("dataset.json не найден — проверка плана и счётчиков пропущена")

    for rid in state:
        if expected_ids and rid not in expected_ids:
            errors.append(f"{rid}: запись вне плана dataset.json")

    saved = skipped = 0
    for rid, r in state.items():
        variant = r.get("variant")
        if variant not in VARIANTS:
            errors.append(f"{rid}: неизвестный variant {variant!r}")
        if expected_ids and variant in VARIANTS and rid in expected_ids:
            exp_street, exp_variant = expected_ids[rid]
            if exp_variant != variant or r.get("street") != exp_street:
                errors.append(f"{rid}: street/variant не соответствуют плану")
        raw = (r.get("reference_raw") or "").strip()
        status = r.get("status")
        if status == "saved":
            if not raw:
                errors.append(f"{rid}: пустой reference_raw")
            if not (r.get("reference") or "").strip():
                errors.append(f"{rid}: пустой reference")
            audio = r.get("audio")
            if not audio:
                errors.append(f"{rid}: сохранённая запись без пути к audio")
                continue
            apath = d / audio
            if not apath.exists():
                errors.append(f"{rid}: файл не найден: {audio}")
                continue
            try:
                info = wav_info(apath)
            except Exception as e:
                errors.append(f"{rid}: WAV не читается ({e})")
                continue
            if info["sample_rate"] != TARGET_RATE:
                errors.append(f"{rid}: sample_rate {info['sample_rate']} != 8000")
            if info["channels"] != CHANNELS:
                errors.append(f"{rid}: channels {info['channels']} != 1")
            if info["sample_width_bytes"] != SAMPLE_WIDTH_BYTES:
                errors.append(f"{rid}: разрядность {info['sample_width_bytes'] * 8} bit != 16 bit")
            if info["comptype"] != "NONE":
                errors.append(f"{rid}: WAV сжат ({info['comptype']}), ожидался PCM")
            dur = info["frames"] / info["sample_rate"] if info["sample_rate"] else 0.0
            if dur <= 0:
                errors.append(f"{rid}: нулевая длительность")
            data_bytes = info["frames"] * info["channels"] * info["sample_width_bytes"]
            if data_bytes > info["file_bytes"]:
                errors.append(
                    f"{rid}: WAV обрезан — заголовок заявляет {data_bytes} байт данных, "
                    f"в файле всего {info['file_bytes']}")
            m_dur = r.get("duration_sec")
            if m_dur is None or abs(float(m_dur) - dur) > 0.05:
                errors.append(f"{rid}: duration_sec {m_dur} != {round(dur, 3)}")
            if r.get("sample_rate") not in (None, TARGET_RATE):
                errors.append(f"{rid}: manifest sample_rate != 8000")
            if r.get("channels") not in (None, CHANNELS):
                errors.append(f"{rid}: manifest channels != 1")
            saved += 1
        elif status == "skipped":
            skipped += 1
            if r.get("audio"):
                errors.append(f"{rid}: skipped-запись со ссылкой на audio")
        else:
            errors.append(f"{rid}: неизвестный status {status!r}")

    # wav files on disk vs manifest
    audio_dir = d / "audio"
    disk = {p.name for p in audio_dir.glob("*.wav")} if audio_dir.is_dir() else set()
    referenced = {Path(r["audio"]).name for r in state.values()
                  if r.get("status") == "saved" and r.get("audio")}
    for extra in sorted(disk - referenced):
        warnings.append(f"лишний WAV без записи в manifest: {extra}")
    for missing in sorted(referenced - disk):
        errors.append(f"WAV указан в manifest, но отсутствует: {missing}")

    pending = (len(expected_ids) - saved - skipped) if expected_ids else None
    stats = {"expected": len(expected_ids) or None, "saved": saved,
             "skipped": skipped, "pending": pending}
    return {"ok": not errors, "errors": errors, "warnings": warnings, "stats": stats}
