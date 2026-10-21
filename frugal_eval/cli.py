"""Command line interface.

    frugal-eval methods                       # list every method with its paper
    frugal-eval replay scores.csv             # find the best column of a CSV with few "calls"
    frugal-eval demo --task gsm8k             # run every method on real leaderboard data
    frugal-eval bench identify --trials 20    # benchmark a problem across tasks
"""
from __future__ import annotations

import argparse
import csv
import sys

import numpy as np

from .bench import DEFAULT_TASKS, run, to_markdown
from .estimate import estimate
from .identify import identify
from .online import run_online
from .registry import PROBLEMS, list_methods
from .tasks import LEADERBOARD_TASKS, leaderboard


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


def cmd_methods(args) -> int:
    for problem in PROBLEMS:
        print(f"\n{problem}")
        for m in list_methods(problem):
            print(f"  {m.name:<24} {m.kind:<22} {m.paper}")
    return 0


def cmd_replay(args) -> int:
    scores, names = _read_csv(args.csv)
    result = identify(scores, method=args.method, names=names, delta=args.delta, epsilon=args.epsilon,
                      budget=args.budget, seed=args.seed)
    print(result.summary())
    true = scores.mean(1)
    print(f"\nFull evaluation would pick: {names[int(np.argmax(true))]} ({true.max():.3f})")
    return 0


def cmd_demo(args) -> int:
    task = leaderboard(args.task, n_candidates=args.k, seed=args.seed)
    print(f"Task: {args.task}, {task.n_candidates} real leaderboard models, {task.n_examples} questions\n")
    print("== Find the best (calls saved, right answer?) ==")
    best = int(np.argmax(task.truth))
    for m in list_methods("identify"):
        budget = int(0.3 * task.n_candidates * task.n_examples) if m.kind == "fixed budget" else None
        r = identify(task, method=m.name, budget=budget, seed=args.seed)
        print(f"  {m.name:<24} {r.savings:5.0%} saved   {'right' if r.best == best else 'WRONG'}")
    print("\n== Estimate every score from 100 questions (mean abs error) ==")
    for m in list_methods("estimate"):
        r = estimate(task, method=m.name, budget=100, seed=args.seed)
        print(f"  {m.name:<24} {r.mae * 100:5.1f} points")
    print("\n== Online: pick one model per question (average score vs best single model) ==")
    for m in list_methods("online"):
        r = run_online(task, method=m.name, seed=args.seed)
        print(f"  {m.name:<24} {r.average_reward:.3f}  ({r.total_reward / r.best_fixed_reward:.0%})")
    return 0


def cmd_bench(args) -> int:
    rows = run(args.problem, methods=args.methods, tasks=args.tasks, trials=args.trials)
    print(to_markdown(args.problem, rows))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="frugal-eval", description="Cheaper LLM evaluation and online model selection.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("methods", help="list every method and its paper").set_defaults(fn=cmd_methods)

    rp = sub.add_parser("replay", help="find the best column of a CSV (rows = examples, columns = candidates)")
    rp.add_argument("csv")
    rp.add_argument("--method", "--algorithm", dest="method", default="lucb",
                    choices=[m.name for m in list_methods("identify")])
    rp.add_argument("--delta", type=float, default=0.05)
    rp.add_argument("--epsilon", type=float, default=0.0)
    rp.add_argument("--budget", type=int, default=None)
    rp.add_argument("--seed", type=int, default=0)
    rp.set_defaults(fn=cmd_replay)

    dp = sub.add_parser("demo", help="run every method on real leaderboard data (no API key needed)")
    dp.add_argument("--task", default="gsm8k", choices=LEADERBOARD_TASKS)
    dp.add_argument("--k", type=int, default=10, help="number of models to compare")
    dp.add_argument("--seed", type=int, default=0)
    dp.set_defaults(fn=cmd_demo)

    bp = sub.add_parser("bench", help="benchmark all methods for one problem")
    bp.add_argument("problem", choices=PROBLEMS)
    bp.add_argument("--methods", nargs="+", default=None)
    bp.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS), choices=LEADERBOARD_TASKS)
    bp.add_argument("--trials", type=int, default=10)
    bp.set_defaults(fn=cmd_bench)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
