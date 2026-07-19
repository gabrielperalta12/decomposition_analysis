"""Command-line interface for the Notebook 01 production methods."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from scripts.rate_decomposition import (
    chevan_categorical_report,
    kitagawa_two_period,
    multiperiod_kitagawa,
)


def _write(frame: pd.DataFrame, destination: Path | None) -> None:
    if destination is None:
        print(frame.to_string())
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination)
    print(f"Wrote {destination}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validated descriptive rate decompositions (not causal effects)."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    two = subparsers.add_parser("kitagawa", help="Two-period Kitagawa decomposition")
    two.add_argument("csv", type=Path, help="CSV with segment,w0,w1,r0,r1")
    two.add_argument("--output", type=Path, help="Destination CSV for category detail")
    two.add_argument(
        "--missing-rate-policy",
        choices=["error", "separate", "reference"],
        default="error",
    )
    two.add_argument("--reference-rate", type=float)

    multi = subparsers.add_parser(
        "multiperiod", help="Direct and adjacent-period Kitagawa comparison"
    )
    multi.add_argument("csv", type=Path, help="CSV with period,segment,weight,rate")
    multi.add_argument("--output", type=Path, help="Destination CSV for link detail")
    multi.add_argument(
        "--missing-rate-policy",
        choices=["error", "separate", "reference"],
        default="error",
    )
    multi.add_argument("--reference-rate", type=float)
    return parser


def main() -> None:
    args = _parser().parse_args()
    data = pd.read_csv(args.csv)
    if args.command == "kitagawa":
        result = kitagawa_two_period(
            data,
            missing_rate_policy=args.missing_rate_policy,
            reference_rate=args.reference_rate,
        )
        print("Summary")
        print(result.summary.to_string())
        print("\nCategory detail")
        report = chevan_categorical_report(result)
        _write(report, args.output)
    else:
        result = multiperiod_kitagawa(
            data,
            missing_rate_policy=args.missing_rate_policy,
            reference_rate=args.reference_rate,
        )
        print("Direct versus chained")
        print(result.comparison.to_string())
        print("\nAdjacent links")
        _write(result.links, args.output)


if __name__ == "__main__":
    main()

