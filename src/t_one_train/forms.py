"""Curated forms: never decline personal names that are already genitive."""
import re
from pathlib import Path

NUMERALS = {"40": "сорок", "50": "пятьдесят", "60": "шестьдесят", "65": "шестьдесят пять", "70": "семьдесят"}


def normalize(text):
    """Exactly the Russian-word extraction used by the official notebook; keeps ё."""
    return " ".join(re.findall("[а-яё]+", text.lower()))


def read_streets(path):
    streets, excluded = [], []
    for i, raw in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        s = raw.strip()
        if not s:
            continue
        if re.fullmatch(r"Страница \d+ \(улицы \d+[–-]\d+\)", s):
            excluded.append({"line": i, "text": s, "reason": "page heading"})
            continue
        if not re.fullmatch(r"[А-Яа-яЁё0-9 -]+", s):
            raise ValueError(f"Invalid street line {i}: {s}")
        if s in streets:
            raise ValueError(f"Duplicate street: {s}")
        forms(s)  # reject unknown structure / digits before generation
        streets.append(s)
    if not streets:
        raise ValueError("Empty streets file")
    return streets, excluded


def forms(street):
    s = street.lower()
    def number(m):
        if m[0] not in NUMERALS:
            raise ValueError(f"Unreviewed numeral in {street}")
        return NUMERALS[m[0]]
    s = re.sub(r"\d+", number, s)
    if s.endswith(" улица"):
        adjective = s.removesuffix(" улица")
        if not adjective.endswith(("ая", "яя")) or " " in adjective:
            raise ValueError(f"Unreviewed adjective: {street}")
        stem = adjective[:-2]
        oblique, accusative = ("ей", "юю") if adjective.endswith("яя") else ("ой", "ую")
        return dict(nom=f"{adjective} улица", pre=f"{stem}{oblique} улице",
                    acc=f"{stem}{accusative} улицу", gen=f"{stem}{oblique} улицы",
                    bare=adjective, at=stem+oblique, to=stem+accusative)
    if s.startswith("улица "):
        name = s.removeprefix("улица ")
        # Name is already a genitive anthroponym or a quoted toponym.
        # All non-adjectival names stay unchanged WITH a type word.
        return dict(nom=s, pre="улице "+name, acc="улицу "+name,
                    gen="улицы "+name, bare=name, at=None, to=None)
    if s.startswith("переулок "):
        name = s.removeprefix("переулок ")
        return dict(nom=s, pre="переулке "+name, acc=s,
                    gen="переулка "+name, bare=None, at=None, to=None)
    raise ValueError(f"Unreviewed street structure: {street}")


def render(template, street):
    f = forms(street)
    keys = re.findall(r"\{(\w+)\}", template)
    if len(keys) != 1 or keys[0] not in f or f[keys[0]] is None:
        return None
    return template.format(**f)
