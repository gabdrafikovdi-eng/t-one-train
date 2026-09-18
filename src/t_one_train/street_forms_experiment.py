"""Pure logic for the experimental street hotword-forms study.

This module deliberately has NO tone/pyctcdecode imports, so the experiment's
own logic (dictionary loading/validation, hotword assembly, street detection,
change classification, reproducible config) is testable headless.

Design constraints (see src/t_one_train/street_hotword_forms_experiment.json):
- the production hotword source stays ``streets.txt`` (``hotwords.load_streets``);
  this module only ADDS an experimental, separate set of extra surface forms;
- anthroponym (personal-name) streets are never declined;
- ``улица``/``переулок`` type words are never hotwords here;
- the additional forms are only real toponym forms used when the word
  "улица" is dropped in colloquial speech ("на Советской", "с Садовой").
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

EXPERIMENT_DICT_PATH = Path(__file__).with_name("street_hotword_forms_experiment.json")

PAGE_HEADING_RE = re.compile(r"Страница \d+ \(улицы \d+[–-]\d+\)")
VALID_FORM_RE = re.compile(r"[А-Яа-яЁё][А-Яа-яЁё -]*")

# Classification labels for a configuration's output relative to the
# no-hotword beam baseline. NONE of these is WER/CER: there is no ground-truth
# transcript, only the expected street from the dataset manifest.
UNCHANGED = "UNCHANGED"
HELPED = "HELPED"
HARMED = "HARMED"
EXPECTED_KEPT = "EXPECTED_KEPT"
NEITHER = "NEITHER"


def normalize_tokens(text: str) -> tuple[str, ...]:
    """Lowercase and keep only Russian word tokens (same as forms.normalize)."""
    return tuple(re.findall(r"[а-яё]+", text.lower()))


def normalize_text(text: str) -> str:
    return " ".join(normalize_tokens(text))


def load_street_names(path: str | Path) -> list[str]:
    """Read streets.txt verbatim (original case), skipping blanks/page headings.

    Unlike forms.read_streets this does NOT call forms() (which requires the old
    "Улица ..."/"... улица" structure) — it just returns the file lines in order,
    exactly like the real-world dataset references.
    """
    names: list[str] = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or PAGE_HEADING_RE.fullmatch(line):
            continue
        names.append(line)
    return names


@dataclass(frozen=True)
class ExperimentForms:
    """Experimental additional forms: canonical street -> extra surface forms."""

    additional_forms: dict[str, tuple[str, ...]]
    no_additional_forms: tuple[str, ...]
    reason: str
    version: int
    path: str

    def as_dict(self) -> dict:
        return {
            "path": self.path,
            "version": self.version,
            "reason": self.reason,
            "additional_forms": {k: list(v) for k, v in self.additional_forms.items()},
            "no_additional_forms": list(self.no_additional_forms),
        }


def load_experiment_forms(path: str | Path = EXPERIMENT_DICT_PATH) -> ExperimentForms:
    """Load and structurally validate the experimental dictionary."""
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top-level JSON must be an object")
    raw_forms = data.get("additional_forms")
    if not isinstance(raw_forms, dict) or not raw_forms:
        raise ValueError(f"{path}: 'additional_forms' must be a non-empty object")
    additional: dict[str, tuple[str, ...]] = {}
    for street, forms in raw_forms.items():
        if not isinstance(street, str) or not street.strip():
            raise ValueError(f"{path}: empty street key {street!r}")
        if not isinstance(forms, list) or not forms:
            raise ValueError(f"{path}: street {street!r} must map to a non-empty list")
        cleaned: list[str] = []
        for form in forms:
            if not isinstance(form, str) or not form.strip():
                raise ValueError(f"{path}: empty form for street {street!r}")
            cleaned.append(form.strip())
        additional[street.strip()] = tuple(cleaned)
    no_forms_block = data.get("documented_no_additional_forms") or {}
    no_forms = tuple((no_forms_block.get("streets") or []))
    return ExperimentForms(
        additional_forms=additional,
        no_additional_forms=no_forms,
        reason=no_forms_block.get("reason", ""),
        version=int(data.get("version", 1)),
        path=str(path),
    )


def validate_experiment_forms(exp: ExperimentForms, street_names: list[str]) -> list[str]:
    """Return a list of human-readable validation errors (empty list == valid)."""
    errors: list[str] = []
    known = set(street_names)
    canonical_tokens = {normalize_text(s): s for s in street_names}
    seen_forms: dict[str, str] = {}

    for street, forms in exp.additional_forms.items():
        if street not in known:
            errors.append(f"unknown street {street!r} (not in streets.txt)")
            continue
        for form in forms:
            if not VALID_FORM_RE.fullmatch(form):
                errors.append(f"invalid characters in form {form!r} for street {street!r}")
            norm = normalize_text(form)
            if not norm:
                errors.append(f"empty normalized form {form!r} for street {street!r}")
                continue
            if len(norm.replace(" ", "")) < 4:
                errors.append(
                    f"too-short form {form!r} for street {street!r} "
                    "(short unigrams cause false bias)"
                )
            if norm in canonical_tokens and canonical_tokens[norm] != street:
                errors.append(
                    f"form {form!r} duplicates canonical name of "
                    f"{canonical_tokens[norm]!r} (ambiguous hotword for {street!r})"
                )
            if norm in seen_forms:
                errors.append(f"duplicate form {form!r} used for {seen_forms[norm]!r} and {street!r}")
            else:
                seen_forms[norm] = street

    for street in exp.no_additional_forms:
        if street not in known:
            errors.append(f"unknown street {street!r} in no_additional_forms")
        if street in exp.additional_forms:
            errors.append(f"street {street!r} appears both in additional_forms and no_additional_forms")
    if not exp.no_additional_forms:
        errors.append("no_additional_forms is empty (documented exclusions are expected)")
    return errors


def _add_unique(out: list[str], seen: set[str], value: str) -> None:
    norm = normalize_text(value)
    if norm and norm not in seen:
        seen.add(norm)
        out.append(norm)


def build_experiment_hotwords(street_names: list[str], exp: ExperimentForms) -> list[str]:
    """Canonical hotwords (file order) followed by the extra forms, deterministic.

    A street's extra forms are emitted right after its canonical form; duplicates
    are dropped while preserving first occurrence. Output is lowercase and has no
    'улица'/'переулок' type words (streets.txt is already bare names).
    """
    out: list[str] = []
    seen: set[str] = set()
    for street in street_names:
        _add_unique(out, seen, street)
        for form in exp.additional_forms.get(street, ()):
            _add_unique(out, seen, form)
    return out


def build_canonical_hotwords(street_names: list[str]) -> list[str]:
    """Canonical hotwords only (lowercase, dedup), i.e. the production view."""
    return build_experiment_hotwords(
        street_names, ExperimentForms({}, (), "", 1, ""),
    )


@dataclass(frozen=True)
class StreetFormIndex:
    """Token-sequence -> street matcher over canonical and/or additional forms."""

    entries: tuple[tuple[tuple[str, ...], str, str], ...]  # (tokens, street, surface)
    kind: str  # "canonical" | "additional" | "combined"

    def match(self, text: str) -> dict[str, tuple[str, ...]]:
        """Return {street: (matched normalized surface forms...)} for every form present."""
        words = normalize_tokens(text)
        found: dict[str, list[str]] = {}
        for tokens, street, surface in self.entries:
            n = len(tokens)
            if n == 0 or len(words) < n:
                continue
            for i in range(len(words) - n + 1):
                if words[i : i + n] == tokens:
                    found.setdefault(street, [])
                    if surface not in found[street]:
                        found[street].append(surface)
                    break
        return {k: tuple(v) for k, v in found.items()}

    def detect(self, text: str) -> set[str]:
        return set(self.match(text))


def _index_entries(
    street_names: list[str], forms: dict[str, tuple[str, ...]], kind: str
) -> StreetFormIndex:
    entries: list[tuple[tuple[str, ...], str, str]] = []
    for street in street_names:
        for surface in forms.get(street, ()):
            tokens = normalize_tokens(surface)
            if tokens:
                entries.append((tokens, street, normalize_text(surface)))
    # Longest forms first, then stable by (street, surface) for reproducibility.
    entries.sort(key=lambda e: (-len(e[0]), e[1], e[2]))
    return StreetFormIndex(tuple(entries), kind)


def build_canonical_index(street_names: list[str]) -> StreetFormIndex:
    return _index_entries(street_names, {s: (s,) for s in street_names}, "canonical")


def build_empty_index(kind: str = "additional") -> StreetFormIndex:
    """Index matching nothing; for canonical-only scoring (no experimental forms)."""
    return StreetFormIndex((), kind)


def build_additional_index(street_names: list[str], exp: ExperimentForms) -> StreetFormIndex:
    return _index_entries(street_names, exp.additional_forms, "additional")


def build_combined_index(street_names: list[str], exp: ExperimentForms) -> StreetFormIndex:
    forms = {s: (s,) + tuple(exp.additional_forms.get(s, ())) for s in street_names}
    return _index_entries(street_names, forms, "combined")


def contains_form(text: str, surface: str) -> bool:
    """Contiguous token-sequence match of a surface form inside text."""
    words = normalize_tokens(text)
    tokens = normalize_tokens(surface)
    n = len(tokens)
    if n == 0 or len(words) < n:
        return False
    return any(words[i : i + n] == tokens for i in range(len(words) - n + 1))


def partial_candidates(
    text: str, street_names: list[str], min_token_len: int = 4
) -> list[str]:
    """Streets whose longest token (>= min_token_len) appears but full form does not.

    Weak signal used only to flag 'requires manual review', never for the metric.
    """
    words = set(normalize_tokens(text))
    out: list[str] = []
    for street in street_names:
        tokens = normalize_tokens(street)
        longest = max(tokens, key=len, default="")
        if len(longest) >= min_token_len and longest in words and not contains_form(text, street):
            out.append(street)
    return out


def build_config_plan(weights: list[float]) -> list[tuple[str, float | None]]:
    """Deterministic experiment plan: greedy, beam, canonical@w, canonical+forms@w."""
    plan: list[tuple[str, float | None]] = [("greedy", None), ("beam_no_hotwords", None)]
    plan += [("canonical", float(w)) for w in weights]
    plan += [("canonical_plus_forms", float(w)) for w in weights]
    return plan


def config_label(kind: str, weight: float | None) -> str:
    return kind if weight is None else f"{kind}@w{weight:g}"


def classify_change(
    baseline_text: str,
    config_text: str,
    reference: str,
    expected_street: str,
    canonical_index: StreetFormIndex,
    additional_index: StreetFormIndex,
    combined_index: StreetFormIndex,
    greedy_text: str,
    street_names: list[str],
) -> dict:
    """Classify one config output against the no-hotword beam baseline.

    The objective anchor is the dataset reference (the street actually spoken)
    and the expected street from the manifest. Everything is reference-relative,
    so a change toward a random dictionary street is NOT an improvement.
    """
    changed = config_text != baseline_text
    expected_before = contains_form(baseline_text, reference)
    expected_after = contains_form(config_text, reference)

    detected_canonical = sorted(canonical_index.detect(config_text))
    detected_additional = sorted(additional_index.detect(config_text))
    detected_combined = sorted(combined_index.detect(config_text))
    other_streets = sorted(s for s in detected_combined if s != expected_street)

    baseline_tokens = set(normalize_tokens(baseline_text))
    greedy_tokens = set(normalize_tokens(greedy_text))
    unsupported: list[str] = []
    for street in other_streets:
        tokens = normalize_tokens(street)
        if not any(t in baseline_tokens or t in greedy_tokens for t in tokens):
            unsupported.append(street)

    baseline_combined = combined_index.detect(baseline_text)
    additionally_detected = sorted(s for s in detected_combined if s not in baseline_combined)

    if not changed:
        category = UNCHANGED
    elif not expected_before and expected_after:
        category = HELPED
    elif expected_before and not expected_after:
        category = HARMED
    elif expected_after:
        category = EXPECTED_KEPT
    else:
        category = NEITHER

    partials = partial_candidates(config_text, street_names) if not expected_after else []
    review_reasons: list[str] = []
    if category == NEITHER:
        review_reasons.append("changed but expected street absent in baseline and config")
    if unsupported:
        review_reasons.append("unsupported street introduced: " + ", ".join(unsupported))
    if partials and not expected_after:
        review_reasons.append("partial street-word match (ambiguous): " + ", ".join(partials))
    if detected_additional and not expected_after:
        review_reasons.append("oblique experimental form present but reference is canonical")

    return {
        "text": config_text,
        "expected_reference_match": expected_after,
        "expected_reference_match_baseline": expected_before,
        "changed_from_baseline": changed,
        "classification": category,
        "expected_street": expected_street,
        "detected_canonical": detected_canonical,
        "detected_additional_forms": detected_additional,
        "detected_combined": detected_combined,
        "other_streets_detected": other_streets,
        "additionally_detected_vs_baseline": additionally_detected,
        "unsupported_street_introduced": unsupported,
        "partial_candidates": partials,
        "no_street_detected": not detected_combined and not partials,
        "requires_manual_review": bool(review_reasons),
        "review_reasons": review_reasons,
    }


def build_experiment_config(
    *,
    dataset_dir: str,
    streets_file: str,
    street_count: int,
    n_records: int,
    weights: list[float],
    beam_width: int,
    decoder_params: dict,
    canonical_hotwords: list[str],
    canonical_plus_forms_hotwords: list[str],
    experiment_forms: dict,
    env: dict,
) -> dict:
    """Reproducible description of the experiment (written to the results dir)."""
    return {
        "dataset_dir": dataset_dir,
        "streets_file": streets_file,
        "street_count": street_count,
        "n_records": n_records,
        "weights": weights,
        "beam_width": beam_width,
        "decoder_params": decoder_params,
        "acoustic_forwards": n_records,
        "canonical_hotword_count": len(canonical_hotwords),
        "canonical_plus_forms_hotword_count": len(canonical_plus_forms_hotwords),
        "canonical_hotwords": canonical_hotwords,
        "canonical_plus_forms_hotwords": canonical_plus_forms_hotwords,
        "experiment_forms": experiment_forms,
        "env": env,
        "note": (
            "no ground-truth transcript; 'expected_reference_match' compares the decoder "
            "output with the dataset reference (the street actually spoken). WER/CER are "
            "NOT computed."
        ),
    }


# --- expected street for the interactive real-voice test ----------------------

PREPOSITIONS = frozenset(
    {
        "в", "во", "на", "до", "по", "с", "со", "у", "к", "ко", "от", "из",
        "около", "возле", "под", "над", "за", "перед", "про", "через",
    }
)
STREET_TYPE_WORDS = frozenset(
    {
        "улица", "улице", "улицу", "улицы", "улицей",
        "переулок", "переулке", "переулка", "переулком", "переулки",
    }
)
_MIN_PREFIX = 3
_PREFIX_SLACK = 2


def _common_prefix_len(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def _tokens_match(a: str, b: str) -> bool:
    """Inflected form match: same token with a long common prefix ('лесной'/'лесная')."""
    n = min(len(a), len(b))
    if n == 0:
        return False
    need = min(n, max(_MIN_PREFIX, n - _PREFIX_SLACK))
    return _common_prefix_len(a, b) >= need


def tokens_match(a: str, b: str) -> bool:
    """Public inflection-tolerant token comparison (see ``_tokens_match``)."""
    return _tokens_match(a, b)


@dataclass(frozen=True)
class ExpectedStreet:
    """Expected street typed by the operator: used ONLY for scoring, never decoded.

    ``street`` is the canonical ``streets.txt`` name when the typed input could be
    resolved; ``reference`` is the normalized text looked up in decoder output.
    """

    user_input: str
    street: str | None
    candidates: tuple[str, ...] = ()

    @property
    def resolved(self) -> bool:
        return self.street is not None

    @property
    def reference(self) -> str:
        return normalize_text(self.street) if self.street else normalize_text(self.user_input)


def resolve_expected_street(text: str, street_names: list[str]) -> ExpectedStreet:
    """Map a freely typed expected street ('Лесная', 'на Лесной') to a streets.txt name.

    Type words ('улица'/'переулок') and prepositions are dropped; the remaining
    tokens are matched against the canonical names, allowing for the same
    inflection ('Лесной' -> 'Лесная', 'Искры' -> 'Искра'). Ambiguity resolves to
    ``street=None`` so the caller can decide; nothing is ever guessed silently.
    """
    raw = (text or "").strip()
    if not raw:
        return ExpectedStreet(raw, None)
    normalized = normalize_text(raw)
    exact = [s for s in street_names if normalize_text(s) == normalized]
    if len(exact) == 1:
        return ExpectedStreet(raw, exact[0], tuple(exact))
    core = tuple(
        t for t in normalize_tokens(raw)
        if t not in PREPOSITIONS and t not in STREET_TYPE_WORDS
    )
    if not core:
        return ExpectedStreet(raw, None)
    core_exact = [s for s in street_names if normalize_tokens(s) == core]
    if len(core_exact) == 1:
        return ExpectedStreet(raw, core_exact[0], tuple(core_exact))
    hits = [
        s for s in street_names
        if len(normalize_tokens(s)) == len(core)
        and all(_tokens_match(a, b) for a, b in zip(normalize_tokens(s), core))
    ]
    if len(hits) == 1:
        return ExpectedStreet(raw, hits[0], tuple(hits))
    return ExpectedStreet(raw, None, tuple(hits))
