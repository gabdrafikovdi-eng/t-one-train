"""Load streets.txt and prepare hotwords for pyctcdecode 0.5.0.

Format decision (verified against installed pyctcdecode 0.5.0 source):
- HotwordScorer.build_scorer splits every hotword on whitespace into unigrams
  and matches whole words inside decoded text (regex `(?<!\\S)word(?!\\S)`).
- Multi-word hotwords therefore work: each of their words gets +weight when
  present in a beam. Partial (in-progress) words get a proportional score
  via the char trie.
- T-one transcripts are lowercase Russian letters only (label alphabet
  "абвгдеёжзийклмнопрстуфхцчшщъыьэюя " + blank); digits cannot appear in
  decoder output, so numeric street names are expanded to words
  (same mapping as src/t_one_train/forms.py: 40 -> сорок ...).
- "улица"/"переулок" type words are kept: they are real spoken words and
  give the decoder a small bias toward the correct street type too.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

NUMERALS = {"40": "сорок", "50": "пятьдесят", "60": "шестьдесят", "65": "шестьдесят пять", "70": "семьдесят"}

_PAGE_HEADING_RE = re.compile(r"Страница \d+ \(улицы \d+[–-]\d+\)")
_VALID_CHARS_RE = re.compile(r"[А-Яа-яЁё0-9 -]+")


@dataclass
class HotwordEntry:
    source: str
    hotword: str
    excluded: bool = False
    reason: str | None = None


def _digits_to_words(text: str) -> str:
    def repl(m: re.Match[str]) -> str:
        if m[0] not in NUMERALS:
            raise ValueError(f"Unreviewed numeral {m[0]!r} in street {text!r}")
        return NUMERALS[m[0]]

    return re.sub(r"\d+", repl, text)


def load_streets(path: str | Path) -> tuple[list[str], list[HotwordEntry]]:
    """Read streets.txt; return (hotwords, per-line audit entries)."""
    entries: list[HotwordEntry] = []
    hotwords: list[str] = []
    seen: set[str] = set()
    text = Path(path).read_text(encoding="utf-8")
    for lineno, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line:
            entries.append(HotwordEntry(raw.strip(), "", excluded=True, reason="empty line"))
            continue
        if _PAGE_HEADING_RE.fullmatch(line):
            entries.append(HotwordEntry(line, "", excluded=True, reason="page heading"))
            continue
        if not _VALID_CHARS_RE.fullmatch(line):
            entries.append(HotwordEntry(line, "", excluded=True, reason="invalid characters"))
            continue
        lower = line.lower()
        expanded = _digits_to_words(lower)
        if expanded in seen:
            entries.append(HotwordEntry(line, expanded, excluded=True, reason="duplicate"))
            continue
        seen.add(expanded)
        hotwords.append(expanded)
        entries.append(HotwordEntry(line, expanded))
    return hotwords, entries
