"""Recalculate summary.csv from detailed_results.jsonl (fixes changed/unchanged counts)."""

import csv
import json
import sys
from pathlib import Path

d = Path(sys.argv[1])
rows = [json.loads(l) for l in (d / "detailed_results.jsonl").open(encoding="utf-8")]
n_speech = sum(1 for r in rows if r["structure"] == "baseline_beam")
keys = ["changed", "unchanged", "changed_gt30pct", "street_introduced", "street_lost",
        "street_replaced", "street_extended", "street_narrowed", "same_street_kept",
        "no_street_changed", "unsupported_introduced"]
counters: dict[tuple[str, float], dict[str, int]] = {}
for r in rows:
    if r["structure"].startswith("baseline"):
        continue
    key = (r["structure"], r["weight"])
    c = counters.setdefault(key, {k: 0 for k in keys})
    cat = r["category"]
    if cat == "UNCHANGED":
        c["unchanged"] += 1
    else:
        c["changed"] += 1
    if r["tokens_changed_ratio"] > 0.3:
        c["changed_gt30pct"] += 1
    if cat != "UNCHANGED":
        c[cat.lower()] += 1
    c["unsupported_introduced"] += len(r["unsupported_introduced"])

fields = ["configuration", "weight", "n", "changed", "unchanged", "changed_gt30pct",
          "street_introduced", "street_lost", "street_replaced", "street_extended",
          "street_narrowed", "same_street_kept", "no_street_changed", "unsupported_introduced"]
with (d / "summary.csv").open("w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerow({"configuration": "baseline_beam", "weight": 0, "n": n_speech, **{k: 0 for k in fields[3:]}})
    for (s, wt), c in sorted(counters.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        w.writerow({"configuration": s, "weight": wt, "n": n_speech, **c})
print("summary.csv recalculated OK")
