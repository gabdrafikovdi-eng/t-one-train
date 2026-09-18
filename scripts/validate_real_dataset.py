"""Validate a real-world evaluation dataset produced by record_real_dataset.py.

Checks manifest integrity (no duplicate id, plan match, non-empty references),
WAV format (mono, 8000 Hz, PCM16), file existence and duration consistency.
Skipped records are NOT errors.

Usage:
    uv run python scripts/validate_real_dataset.py                       # latest dataset
    uv run python scripts/validate_real_dataset.py results/real_dataset_XXX/
    uv run python scripts/validate_real_dataset.py --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.t_one_train.real_dataset import (  # noqa: E402
    find_latest_dataset,
    validate_dataset,
)

DEFAULT_RESULTS = ROOT / "results"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="validate_real_dataset",
        description="Validate real-world evaluation dataset(s).",
    )
    p.add_argument("datasets", nargs="*", type=Path,
                   help="dataset dirs (default: latest real_dataset_* in --results-dir)")
    p.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    p.add_argument("--json", action="store_true", help="print a JSON report")
    return p.parse_args(argv)


def resolve_targets(args: argparse.Namespace) -> list[Path]:
    if args.datasets:
        return list(args.datasets)
    latest = find_latest_dataset(Path(args.results_dir))
    return [latest] if latest else []


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    targets = resolve_targets(args)
    if not targets:
        print(f"Не найдено ни одного каталога real_dataset_* в {args.results_dir}")
        return 1

    reports = []
    failed = False
    for d in targets:
        report = validate_dataset(d)
        reports.append({"dataset": str(d), **report})
        failed = failed or not report["ok"]

    if args.json:
        print(json.dumps(reports, ensure_ascii=False, indent=2))
        return 1 if failed else 0

    for r in reports:
        s = r["stats"]
        print("=" * 60)
        print(f"Dataset : {r['dataset']}")
        print(f"Статус  : {'OK' if r['ok'] else 'ОШИБКИ'}")
        print(f"План    : {s['expected']}, сохранено {s['saved']}, "
              f"пропущено {s['skipped']}, осталось {s['pending']}")
        for e in r["errors"]:
            print(f"  ✗ {e}")
        for w in r["warnings"]:
            print(f"  ⚠ {w}")
        if not r["errors"] and not r["warnings"]:
            print("  Замечаний нет.")
    print("=" * 60)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
