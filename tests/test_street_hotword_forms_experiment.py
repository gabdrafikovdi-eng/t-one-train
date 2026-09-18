"""Unit tests for the experimental street hotword-forms layer.

No T-one/pyctcdecode imports: only dictionary loading/validation, canonical +
experimental hotword assembly, street-form detection, change classification,
reproducible config and dataset loading.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.run_street_hotword_experiment import (  # noqa: E402
    DEFAULT_DATASET,
    load_records,
    parse_weights,
)
from src.t_one_train.hotwords import load_streets  # noqa: E402
from src.t_one_train.street_forms_experiment import (  # noqa: E402
    ExperimentForms,
    HELPED, HARMED, NEITHER, UNCHANGED,
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
    normalize_text,
    partial_candidates,
    validate_experiment_forms,
)

STREETS = ROOT / "streets.txt"

EXPECTED_EXTRA_FORMS = {
    "весенней", "горной", "колхозной", "коммунистической", "комсомольской",
    "лесной", "луговой", "магнитогорской", "молодежной", "партизанской",
    "первомайской", "пионерской", "пятой", "садовой", "советской", "солнечной",
    "сосновой", "тангатарской", "уральской", "целлинной", "школьной",
    "юбилейной", "южной", "северной",
    "искры", "искре", "караташа", "караташе", "сабантуя", "сабантуе",
}


@pytest.fixture(scope="module")
def street_names() -> list[str]:
    return load_street_names(STREETS)


@pytest.fixture(scope="module")
def exp() -> ExperimentForms:
    return load_experiment_forms()


@pytest.fixture(scope="module")
def canonical(street_names) -> list[str]:
    return build_canonical_hotwords(street_names)


# --- dictionary loading / validation -----------------------------------------


def test_experiment_dictionary_loads(exp):
    assert exp.version == 1
    assert len(exp.additional_forms) == 27
    total_forms = sum(len(v) for v in exp.additional_forms.values())
    assert total_forms == 30
    assert exp.no_additional_forms
    assert exp.path.endswith("street_hotword_forms_experiment.json")


def test_all_form_keys_exist_in_streets(exp, street_names):
    unknown = [s for s in exp.additional_forms if s not in set(street_names)]
    assert unknown == []


def test_all_documented_exclusions_exist_in_streets(exp, street_names):
    unknown = [s for s in exp.no_additional_forms if s not in set(street_names)]
    assert unknown == []


def test_validation_passes_on_real_dictionary(exp, street_names):
    assert validate_experiment_forms(exp, street_names) == []


def test_no_duplicate_or_empty_forms(exp):
    all_forms = [f for forms in exp.additional_forms.values() for f in forms]
    assert all(f.strip() for f in all_forms)
    normalized = [normalize_text(f) for f in all_forms]
    assert len(normalized) == len(set(normalized))


def test_no_extra_forms_for_anthroponyms_and_numerals(exp):
    """Personal-name streets must never be declined; numeric streets stay canonical."""
    anthroponyms = [
        "Александра Пушкина", "Ахмет Заки Валиди", "Мусы Муртазина",
        "Мустая Карима", "Николая Гоголя", "Тамерлана Ильгамова",
        "Файзрахмана Хисматуллина", "Шайхизады Бабича",
    ]
    numerals = [
        "Сорок лет Октября", "Сорок лет Победы", "Пятьдесят лет Победы",
        "Шестьдесят лет Победы", "Шестьдесят пять лет Победы",
        "Семьдесят лет Октября", "Семьдесят лет Победы",
    ]
    for street in anthroponyms + numerals:
        assert street not in exp.additional_forms, street
        assert street in exp.no_additional_forms, street
    assert len(anthroponyms + numerals) == 15


def test_targeted_streets_have_extra_forms(exp):
    assert exp.additional_forms["Советская"] == ("Советской",)
    assert exp.additional_forms["Садовая"] == ("Садовой",)
    assert exp.additional_forms["Лесная"] == ("Лесной",)
    assert exp.additional_forms["Пятая"] == ("Пятой",)
    assert exp.additional_forms["Искра"] == ("Искры", "Искре")
    assert exp.additional_forms["Караташ"] == ("Караташа", "Караташе")
    assert exp.additional_forms["Сабантуй"] == ("Сабантуя", "Сабантуе")


def test_validate_rejects_unknown_street(street_names):
    bad = ExperimentForms({"Несуществующая": ("Несуществующей",)}, ("Ленина",), "", 1, "x")
    errors = validate_experiment_forms(bad, street_names)
    assert any("unknown street" in e for e in errors)


def test_validate_rejects_short_and_duplicate_forms(street_names):
    bad = ExperimentForms(
        {"Советская": ("Сов", "Садовой"), "Садовая": ("Садовой",)},
        ("Ленина",), "", 1, "x",
    )
    errors = validate_experiment_forms(bad, street_names)
    assert any("too-short" in e for e in errors)
    assert any("duplicate form" in e for e in errors)


def test_validate_rejects_street_in_both_sections(street_names):
    bad = ExperimentForms({"Советская": ("Советской",)}, ("Советская",), "", 1, "x")
    errors = validate_experiment_forms(bad, street_names)
    assert any("both" in e for e in errors)


# --- hotword assembly ---------------------------------------------------------


def test_streets_file_has_109_unique_names(street_names):
    assert len(street_names) == 109
    assert len(set(street_names)) == 109


def test_canonical_hotwords_match_production_loader(street_names, canonical):
    production, _audit = load_streets(STREETS)
    assert canonical == production
    assert len(canonical) == 109
    assert all(h == h.lower() for h in canonical)


def test_experiment_hotwords_are_canonical_plus_extra_forms(street_names, exp, canonical):
    combined = build_experiment_hotwords(street_names, exp)
    assert len(combined) == 139
    # canonical part is unchanged and keeps file order once the extra forms are removed
    canonical_part = [h for h in combined if h not in EXPECTED_EXTRA_FORMS]
    assert canonical_part == canonical
    extra = [h for h in combined if h in EXPECTED_EXTRA_FORMS]
    assert set(extra) == EXPECTED_EXTRA_FORMS
    assert len(extra) == 30


def test_extra_forms_follow_their_canonical_form(street_names, exp):
    combined = build_experiment_hotwords(street_names, exp)
    i = combined.index("советская")
    assert combined[i + 1] == "советской"
    j = combined.index("искра")
    assert combined[j + 1 : j + 3] == ["искры", "искре"]


def test_hotword_assembly_is_deterministic_and_no_type_words(street_names, exp, canonical):
    first = build_experiment_hotwords(street_names, exp)
    second = build_experiment_hotwords(street_names, exp)
    assert first == second
    assert build_canonical_hotwords(street_names) == canonical
    for hotword in first:
        assert "улица" not in hotword.split()
        assert "переулок" not in hotword.split()
        assert hotword == hotword.lower()


def test_hotwords_count_matches_streets_plus_forms(street_names, exp):
    combined = build_experiment_hotwords(street_names, exp)
    assert len(combined) == len(street_names) + sum(len(v) for v in exp.additional_forms.values())


# --- street detection ---------------------------------------------------------


def test_canonical_index_detects_nominative(street_names, exp):
    idx = build_canonical_index(street_names)
    assert idx.detect("на советской") == set()
    assert idx.detect("советская") == {"Советская"}
    assert idx.detect("мусы муртазина") == {"Мусы Муртазина"}
    assert idx.detect("совершенно пустая фраза") == set()


def test_additional_index_detects_colloquial_forms(street_names, exp):
    idx = build_additional_index(street_names, exp)
    assert idx.detect("на советской") == {"Советская"}
    assert idx.detect("с садовой") == {"Садовая"}
    assert idx.detect("до искры") == {"Искра"}
    assert idx.detect("караташе") == {"Караташ"}
    assert idx.detect("сабантуе") == {"Сабантуй"}
    assert idx.detect("советская") == set()


def test_combined_index_handles_both(street_names, exp):
    idx = build_combined_index(street_names, exp)
    assert idx.detect("советская") == {"Советская"}
    assert idx.detect("советской") == {"Советская"}
    assert len(idx.entries) == 139


def test_additional_forms_do_not_collide_with_other_streets(exp, street_names):
    canon = {normalize_text(s) for s in street_names}
    for forms in exp.additional_forms.values():
        for form in forms:
            assert normalize_text(form) not in canon


def test_partial_candidates_flags_only_incomplete(street_names):
    assert partial_candidates("гагарина", street_names) == []
    assert "Файзрахмана Хисматуллина" in partial_candidates("хисматуллина", street_names)
# --- change classification ----------------------------------------------------


@pytest.fixture(scope="module")
def indexes(street_names, exp):
    return {
        "canonical": build_canonical_index(street_names),
        "additional": build_additional_index(street_names, exp),
        "combined": build_combined_index(street_names, exp),
    }


def classify(baseline, config, reference, street, greedy, indexes, street_names):
    return classify_change(
        baseline_text=baseline, config_text=config, reference=reference,
        expected_street=street, canonical_index=indexes["canonical"],
        additional_index=indexes["additional"], combined_index=indexes["combined"],
        greedy_text=greedy, street_names=street_names,
    )


def test_classify_helped(indexes, street_names):
    info = classify("апкн", "абзелиловская", "абзелиловская", "Абзелиловская",
                    "апкн", indexes, street_names)
    assert info["classification"] == HELPED
    assert info["expected_reference_match"] is True
    assert info["changed_from_baseline"] is True


def test_classify_harmed_by_oblique_form(indexes, street_names):
    """Nominative was correct in baseline; the extra oblique form must count as HARMED."""
    info = classify("советская", "советской", "советская", "Советская",
                    "советская", indexes, street_names)
    assert info["classification"] == HARMED
    assert info["expected_reference_match"] is False
    assert info["detected_additional_forms"] == ["Советская"]
    assert info["requires_manual_review"] is True


def test_classify_unchanged(indexes, street_names):
    info = classify("садовая", "садовая", "садовая", "Садовая",
                    "садовая", indexes, street_names)
    assert info["classification"] == UNCHANGED
    assert info["changed_from_baseline"] is False
    assert info["requires_manual_review"] is False


def test_classify_expected_kept(indexes, street_names):
    info = classify("садовая", "на садовой садовая", "садовая", "Садовая",
                    "садовая", indexes, street_names)
    assert info["classification"] in ("EXPECTED_KEPT", "NEITHER")
    # "садовая" is still present -> EXPECTED_KEPT
    assert info["classification"] == "EXPECTED_KEPT"
    assert info["expected_reference_match"] is True


def test_classify_neither_and_unsupported(indexes, street_names):
    info = classify("галина", "колхозной", "гагарина", "Гагарина",
                    "галина", indexes, street_names)
    assert info["classification"] == NEITHER
    assert info["changed_from_baseline"] is True
    assert info["unsupported_street_introduced"] == ["Колхозная"]
    assert info["requires_manual_review"] is True


def test_classify_no_street_detected(indexes, street_names):
    info = classify("апкн", "апкн", "абзелиловская", "Абзелиловская",
                    "апкн", indexes, street_names)
    assert info["classification"] == UNCHANGED
    assert info["no_street_detected"] is True


def test_change_toward_wrong_dictionary_street_is_not_improvement(indexes, street_names):
    """baseline 'галина' -> config 'горная' is NOT an improvement when expected is Садовая."""
    info = classify("галина", "горная", "садовая", "Садовая",
                    "галина", indexes, street_names)
    assert info["expected_reference_match"] is False
    assert info["classification"] == NEITHER


# --- reproducible config / plan ----------------------------------------------


def test_config_plan_is_deterministic():
    plan = build_config_plan([1, 3, 5, 7, 10])
    assert plan == [
        ("greedy", None), ("beam_no_hotwords", None),
        ("canonical", 1.0), ("canonical", 3.0), ("canonical", 5.0),
        ("canonical", 7.0), ("canonical", 10.0),
        ("canonical_plus_forms", 1.0), ("canonical_plus_forms", 3.0),
        ("canonical_plus_forms", 5.0), ("canonical_plus_forms", 7.0),
        ("canonical_plus_forms", 10.0),
    ]
    assert build_config_plan([5]) == build_config_plan([5])
    assert config_label("canonical", 5.0) == "canonical@w5"
    assert config_label("greedy", None) == "greedy"


def test_experiment_config_reproducible(street_names, exp, canonical):
    combined = build_experiment_hotwords(street_names, exp)
    kwargs = dict(
        dataset_dir="d", streets_file="s.txt", street_count=len(street_names),
        n_records=109, weights=[1.0, 3.0], beam_width=200,
        decoder_params={"alpha": 0.4, "beta": 0.9},
        canonical_hotwords=canonical, canonical_plus_forms_hotwords=combined,
        experiment_forms=exp.as_dict(), env={"python": "3.12"},
    )
    first = build_experiment_config(**kwargs)
    second = build_experiment_config(**kwargs)
    assert first == second
    assert first["canonical_hotword_count"] == 109
    assert first["canonical_plus_forms_hotword_count"] == 139
    assert first["beam_width"] == 200


# --- dataset / script helpers -------------------------------------------------


def test_parse_weights():
    assert parse_weights("1,3,5,7,10") == [1.0, 3.0, 5.0, 7.0, 10.0]
    assert parse_weights(" 5 ") == [5.0]
    with pytest.raises(SystemExit):
        parse_weights(" , ")


@pytest.mark.skipif(not DEFAULT_DATASET.exists(), reason="real dataset not present")
def test_load_records_finds_109_full_recordings():
    records = load_records(DEFAULT_DATASET, "full")
    assert len(records) == 109
    assert all(r["variant"] == "full" and r["status"] == "saved" for r in records)
    assert all(r["_wav"].exists() for r in records)
    assert len({r["street"] for r in records}) == 109
    assert len({r["reference"] for r in records}) == 109
    assert all(r["reference"] for r in records)


@pytest.mark.skipif(not DEFAULT_DATASET.exists(), reason="real dataset not present")
def test_dataset_references_match_streets_file(street_names):
    records = load_records(DEFAULT_DATASET, "full")
    assert {r["street"] for r in records} == set(street_names)