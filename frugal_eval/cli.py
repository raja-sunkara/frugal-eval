"""Command line interface: one command per job, all methods selectable by name.

    frugal-eval methods                                          # every method with its paper
    frugal-eval demo                                             # run everything on real data, no API key

    frugal-eval identify --task leaderboard/gsm8k --method lucb  # find the best with few calls
    frugal-eval identify --csv scores.csv                        # ... on your own scores
    frugal-eval estimate --task leaderboard/mmlu --method irt --budget 100
    frugal-eval route    --task routerbench --method linucb --cost-weight 0.5
    frugal-eval duel     --task arena --method borda_lucb --epsilon 0.02

    frugal-eval bench identify --trials 20                       # benchmark every method of a problem
"""
from __future__ import annotations

import argparse
import csv
import sys

import numpy as np

from .bench import DEFAULT_TASKS, run, to_markdown
from .duel import arena, find_best_pairwise, from_votes, synthetic_duel
from .estimate import estimate
from .identify import identify
from .online import run_online
from .registry import PROBLEMS, list_methods
from .tasks import LEADERBOARD_TASKS, from_scores, leaderboard, list_tasks, make


def _read_csv(path: str) -> tuple[np.ndarray, list[str]]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if len(rows) < 2:
        raise SystemExit("CSV needs a header row and at least one row of scores")
    names, data = rows[0], rows[1:]
    try:
        matrix = np.array([[float(x) for x in row] for row in data]).T
    except ValueError as exc:
        raise SystemExit(f"every cell must be a number: {exc}")
    return matrix, names


def _task(args, **kw):
    if getattr(args, "csv", None):
        scores, names = _read_csv(args.csv)
        return from_scores(scores, names)
    if args.task.startswith("leaderboard/"):
        kw.setdefault("n_candidates", args.k)
    return make(args.task, seed=args.seed, **kw)


def cmd_methods(args) -> int:
    for problem in PROBLEMS:
        print(f"\n{problem}")
        for m in list_methods(problem):
            print(f"  {m.name:<24} {m.kind:<22} {m.paper}")
    print("\ntasks:", ", ".join(list_tasks()), "(and for `duel`: arena, arena/top, synthetic)")
    return 0


def cmd_identify(args) -> int:
    task = _task(args)
    r = identify(task, method=args.method, delta=args.delta, epsilon=args.epsilon, budget=args.budget, seed=args.seed)
    print(r.summary())
    if task.truth is not None:
        print(f"\nFull evaluation would pick: {task.names[int(np.argmax(task.truth))]}")
    return 0


def cmd_estimate(args) -> int:
    print(estimate(_task(args), method=args.method, budget=args.budget, seed=args.seed).summary())
    return 0


def cmd_route(args) -> int:
    kw = {"n_examples": args.n} if args.task == "routerbench" and args.n else {}
    task = _task(args, **kw)
    print(run_online(task, method=args.method, cost_weight=args.cost_weight, seed=args.seed).summary())
    return 0


def cmd_duel(args) -> int:
    if args.csv:  # square CSV of wins: row model beat column model, header = model names
        wins, names = _read_csv(args.csv)
        task = from_votes(wins.T, np.zeros_like(wins), names)
    elif args.task == "synthetic":
        task = synthetic_duel(args.k, seed=args.seed)
    else:
        task = arena(args.k, seed=args.seed, pick="top" if args.task.endswith("top") else "random")
    r = find_best_pairwise(task, method=args.method, delta=args.delta, epsilon=args.epsilon,
                           budget=args.budget, seed=args.seed)
    print(r.summary())
    if task.borda is not None:
        print(f"\nTrue best (average win rate): {task.names[int(np.argmax(task.borda))]}")
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
    print("\n== Pairwise: Chatbot Arena human votes, 8 models (votes to certify within 2 points) ==")
    duel_task = arena(8, seed=args.seed)
    for m in list_methods("duel"):
        r = find_best_pairwise(duel_task, method=m.name, epsilon=0.02, budget=300_000, seed=args.seed)
        print(f"  {m.name:<24} {r.comparisons:>9,} votes  {'certified' if r.certified else 'not certified'}")
    return 0


def cmd_bench(args) -> int:
    rows = run(args.problem, methods=args.methods, tasks=args.tasks, trials=args.trials)
    print(to_markdown(args.problem, rows))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="frugal-eval", description="Cheaper LLM evaluation and online model selection.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("methods", help="list every method and its paper").set_defaults(fn=cmd_methods)

    def common(p, problem, default_method, task_default):
        p.add_argument("--method", default=default_method, choices=[m.name for m in list_methods(problem)])
        p.add_argument("--task", default=task_default, help="built-in task (see `frugal-eval methods`)")
        p.add_argument("--k", type=int, default=10 if problem != "duel" else 8, help="number of candidates")
        p.add_argument("--seed", type=int, default=0)

    p = sub.add_parser("identify", help="find the best candidate with as few calls as possible")
    common(p, "identify", "lucb", "leaderboard/gsm8k")
    p.add_argument("--csv", help="your own scores: rows = examples, columns = candidates (header = names)")
    p.add_argument("--delta", type=float, default=0.05)
    p.add_argument("--epsilon", type=float, default=0.0)
    p.add_argument("--budget", type=int, default=None)
    p.set_defaults(fn=cmd_identify)

    p = sub.add_parser("replay", help="alias of `identify --csv`")
    p.add_argument("csv")
    p.add_argument("--method", "--algorithm", dest="method", default="lucb",
                   choices=[m.name for m in list_methods("identify")])
    p.add_argument("--delta", type=float, default=0.05)
    p.add_argument("--epsilon", type=float, default=0.0)
    p.add_argument("--budget", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--task", default="custom")
    p.add_argument("--k", type=int, default=10)
    p.set_defaults(fn=cmd_identify)

    p = sub.add_parser("estimate", help="estimate every candidate's score from a few examples")
    common(p, "estimate", "irt", "leaderboard/gsm8k")
    p.add_argument("--csv")
    p.add_argument("--budget", type=int, default=100, help="examples per candidate")
    p.set_defaults(fn=cmd_estimate)

    p = sub.add_parser("route", help="pick a model per request while learning (optionally cost-aware)")
    common(p, "online", "linucb", "leaderboard/gsm8k")
    p.add_argument("--csv")
    p.add_argument("--cost-weight", type=float, default=0.0)
    p.add_argument("--n", type=int, default=None, help="number of requests (routerbench)")
    p.set_defaults(fn=cmd_route)

    p = sub.add_parser("duel", help="find the best from pairwise comparisons (human or LLM-judge votes)")
    common(p, "duel", "borda_lucb", "arena")
    p.add_argument("--csv", help="square CSV of wins (row beat column), header = model names")
    p.add_argument("--delta", type=float, default=0.05)
    p.add_argument("--epsilon", type=float, default=0.02, help="accept a model within this win rate of the best")
    p.add_argument("--budget", type=int, default=300_000)
    p.set_defaults(fn=cmd_duel)

    p = sub.add_parser("demo", help="run every method on real data (no API key needed)")
    p.add_argument("--task", default="gsm8k", choices=LEADERBOARD_TASKS)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=cmd_demo)

    p = sub.add_parser("bench", help="benchmark all methods for one problem")
    p.add_argument("problem", choices=("identify", "estimate", "online"))
    p.add_argument("--methods", nargs="+", default=None)
    p.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS), choices=LEADERBOARD_TASKS)
    p.add_argument("--trials", type=int, default=10)
    p.set_defaults(fn=cmd_bench)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
