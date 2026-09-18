"""Street Accuracy metrics are validated on real streets from streets.txt."""
import json
import pytest

from src.t_one_train.forms import read_streets
from src.t_one_train.tts import ROOT


def test_street_metrics(tmp_path):
    from src.t_one_train.street_metrics import build_street_dictionary, evaluate_predictions, extract_street
    streets, _ = read_streets(ROOT / "streets.txt")
    table = build_street_dictionary(streets)
    assert extract_street("заберите меня на улице Ленина", table) == ("Улица Ленина", False)
    # "улице искра" is unique to Улица Искра; bare "искра" alone would be ambiguous
    # only if it mapped to several streets (it does not in this corpus).
    found, ambiguous = extract_street("я нахожусь на улице искра", table)
    assert found == "Улица Искра" and not ambiguous
    found, ambiguous = extract_street("переулок искра рядом", table)
    assert found == "Переулок Искра" and not ambiguous
    assert extract_street("позвоните когда подъедете", table) == (None, False)
    # Genuine ambiguity: same longest form mapped to two streets.
    found, ambiguous = extract_street("мы на искра", {"искра": {"Улица Искра", "Переулок Искра"}})
    assert found is None and ambiguous
    test_rows = [
        {"audio": "a1.wav", "text": "заберите меня на улице ленина", "street": "Улица Ленина"},
        {"audio": "a2.wav", "text": "мне нужна машина до улицы гагарина", "street": "Улица Гагарина"},
    ]
    pred_rows = [
        {"prediction": "улица Ленина, заберите меня", "street": "Улица Ленина"},
        {"prediction": "улица Гагарина и всё", "street": "Улица Гагарина"},
    ]
    test_path = tmp_path / "test.jsonl"
    pred_path = tmp_path / "predictions.jsonl"
    test_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in test_rows))
    pred_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in pred_rows))
    m = evaluate_predictions(test_path, pred_path)
    assert m["street_accuracy"] == 1.0 and m["exact_street_accuracy"] == 1.0
    # Predictions differ verbatim from references, though both streets are correct.
    assert m["exact_utterance_accuracy"] == 0.0 and m["no_street_match"] == 0
