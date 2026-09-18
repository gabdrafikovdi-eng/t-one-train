"""Street Accuracy metrics for A/B comparison. Runs on the PC; jiwer only."""
import argparse
import json
import re
import jiwer
from src.t_one_train.forms import forms, normalize, read_streets
from src.t_one_train.tts import ROOT


def build_street_dictionary(streets):
    """Normalized allowed surface forms -> canonical street name."""
    table = {}
    for street in streets:
        f = forms(street)
        variants = {f["nom"], f["pre"], f["acc"], f["gen"]} | ({f["bare"]} if f["bare"] else set())
        for form in variants:
            table.setdefault(form, set()).add(street)
    return table


def extract_street(text, table):
    """Longest dictionary form with word boundaries; ambiguity -> longest match."""
    normalized = " ".join(re.findall("[а-яё]+", text.lower()))
    matches = []
    for form, streets in table.items():
        if re.search(rf"(?<![а-яё]){re.escape(form)}(?![а-яё])", normalized):
            matches.append((len(form), streets))
    if not matches:
        return None, False
    matches.sort(key=lambda m: m[0], reverse=True)
    longest = matches[0][0]
    best = set()
    for length, street_names in matches:
        if length == longest:
            best |= street_names
    canonical = next(iter(best)) if len(best) == 1 else None
    return canonical, canonical is None and len(best) > 1


def evaluate_predictions(test_path, predictions_path, streets_path=None):
    streets, _ = read_streets(streets_path or ROOT / "streets.txt")
    table = build_street_dictionary(streets)
    refs = [json.loads(l) for l in open(test_path, encoding="utf-8")]
    hyps = [json.loads(l) for l in open(predictions_path, encoding="utf-8")]
    if len(refs) != len(hyps):
        raise ValueError("Prediction count differs from test manifest")
    references, predictions = [], []
    street_hits = exact_hits = ambiguous = no_match = 0
    for row, hyp in zip(refs, hyps):
        text_ref = " ".join(re.findall("[а-яё]+", row["text"].lower()))
        text_hyp = " ".join(re.findall("[а-яё]+", hyp["prediction"].lower()))
        references.append(text_ref)
        predictions.append(text_hyp)
        street = hyp.get("street") or row.get("street")
        if street is None:
            raise ValueError("Predictions/manifest must carry street")
        found, was_ambiguous = extract_street(text_hyp, table)
        ambiguous += was_ambiguous
        no_match += found is None and not was_ambiguous
        street_hits += found == street
        canonical_full = " ".join(re.findall("[а-яё]+", forms(street)["nom"]))
        exact_hits += canonical_full in text_hyp
    return {
        "wer": jiwer.wer(references, predictions),
        "cer": jiwer.cer(references, predictions),
        "street_accuracy": street_hits / len(refs),
        "exact_street_accuracy": exact_hits / len(refs),
        "exact_utterance_accuracy": sum(a == b for a, b in zip(references, predictions)) / len(refs),
        "ambiguous_street_matches": ambiguous,
        "no_street_match": no_match,
        "samples": len(refs),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", required=True, help="dataset/test.jsonl")
    parser.add_argument("--predictions", required=True, help="predictions.jsonl from train_tone.py")
    parser.add_argument("--streets", default=str(ROOT / "streets.txt"))
    args = parser.parse_args()
    print(json.dumps(evaluate_predictions(args.test, args.predictions, args.streets),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
