"""Command line: replay frugal-eval on a CSV of scores you already have.

    frugal-eval replay scores.csv --delta 0.05 --epsilon 0.01

The CSV has one column per candidate (header = candidate name) and one row per
example, with scores in [0, 1].
"""
from __future__ import annotations

import argparse
import csv
import sys

import numpy as np

from .core import ALGORITHMS, find_best


def _read_csv(path: str) -> tuple[np.ndarray, list[str]]:
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    if len(rows) < 2:
        raise SystemExit("CSV needs a header row and at least one row of scores")
    names, data = rows[0], rows[1:]
    try:
        matrix = np.array([[float(x) for x in row] for row in data]).T
    except ValueError as exc:
        raise SystemExit(f"every cell must be a number: {exc}")
    return matrix, names


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="frugal-eval", description="Find the best candidate with fewer evaluations.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    rp = sub.add_parser("replay", help="simulate on a CSV of known scores (rows = examples, columns = candidates)")
    rp.add_argument("csv")
    rp.add_argument("--algorithm", default="lucb", choices=ALGORITHMS)
    rp.add_argument("--delta", type=float, default=0.05, help="allowed error probability (default 0.05)")
    rp.add_argument("--epsilon", type=float, default=0.0, help="accept any candidate within this of the best")
    rp.add_argument("--budget", type=int, default=None, help="maximum number of calls")
    rp.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    scores, names = _read_csv(args.csv)
    result = find_best(scores, names=names, algorithm=args.algorithm, delta=args.delta,
                       epsilon=args.epsilon, budget=args.budget, seed=args.seed)
    print(result.summary())
    true = scores.mean(1)
    exact = int(np.argmax(true))
    print(f"\nFull evaluation would pick: {names[exact]} ({true[exact]:.3f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
