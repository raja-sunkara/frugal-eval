"""Benchmark every method on real Open LLM Leaderboard results and write BENCHMARKS.md.

    python benchmarks/run_all.py              # 20 trials per task (about 30 minutes on 2 cores)
    python benchmarks/run_all.py --trials 3   # quick check
    python benchmarks/run_all.py --tasks mmlu --raw-only out/mmlu.json   # one slice (used by CI)
    python benchmarks/run_all.py --merge out/*.json                      # combine slices into the reports
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from frugal_eval.bench import DEFAULT_TASKS, run, to_markdown  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INTRO = {
    "identify": ("Find the best candidate",
                 "10 random leaderboard models per trial; confidence 95%. Fixed-confidence methods stop when "
                 "the winner is certified; fixed-budget methods get 30% of the full cost. Cells: calls saved "
                 "versus running everything, and how often the true best model was returned."),
    "estimate": ("Estimate every score from 100 examples",
                 "10 random leaderboard models per trial, 100 examples each; methods that learn from history "
                 "use 300 other leaderboard models. Cells: mean absolute error in percentage points and rank "
                 "correlation with the true scores."),
    "online": ("Online selection and routing",
               "10 random leaderboard models; every question in the benchmark arrives once in random order "
               "and the method picks one model per question. Cells: average score and its share of what "
               "the best single model would score. Above 100% means routing beat every single model."),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    ap.add_argument("--problems", nargs="+", default=["identify", "estimate", "online"])
    ap.add_argument("--raw-only", metavar="PATH", help="only run and save raw rows to PATH (for parallel jobs)")
    ap.add_argument("--merge", nargs="+", metavar="RAW", help="merge raw files from parallel jobs and write reports")
    args = ap.parse_args()

    if args.merge:
        raw: dict = {}
        trials = set()
        for path in args.merge:
            with open(path) as f:
                part = json.load(f)
            trials.add(part.pop("_trials", None))
            for problem, rows in part.items():
                raw.setdefault(problem, []).extend(rows)
        write_reports(raw, trials=max(t for t in trials if t is not None))
        return

    raw = {}
    for problem in args.problems:
        t0 = time.time()
        raw[problem] = run(problem, tasks=args.tasks, trials=args.trials)
        print(f"{problem}: {len(raw[problem])} runs in {time.time() - t0:.0f}s", flush=True)
    if args.raw_only:
        os.makedirs(os.path.dirname(os.path.abspath(args.raw_only)), exist_ok=True)
        with open(args.raw_only, "w") as f:
            json.dump({**raw, "_trials": args.trials}, f)
        print(f"wrote {args.raw_only}")
        return
    out_dir = os.path.join(ROOT, "benchmarks", "results")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "raw.json"), "w") as f:
        json.dump(raw, f)
    write_reports(raw, trials=args.trials)


def write_reports(raw: dict, trials: int) -> None:
    order = [p for p in ("identify", "estimate", "online") if p in raw]
    tasks = list(dict.fromkeys(r["task"] for p in order for r in raw[p]))
    sections = []
    for problem in order:
        title, text = INTRO[problem]
        sections.append(f"## {title}\n\n{text}\n\n{to_markdown(problem, raw[problem])}\n")
    header = ("# Benchmarks\n\nEvery method on the same real data: per-question results of 395 models from the "
              "Open LLM Leaderboard (via tinyBenchmarks, MIT license). "
              f"Tasks: {', '.join(tasks)}. {trials} trials per task. "
              "Reproduce with `python benchmarks/run_all.py`, or run the Benchmark workflow on GitHub.\n\n")
    with open(os.path.join(ROOT, "BENCHMARKS.md"), "w") as f:
        f.write(header + "\n".join(sections))
    write_summary(raw, tasks=tasks, trials=trials)
    print("wrote BENCHMARKS.md and benchmarks/summary.json")
    try:
        import plot
        plot.main()
    except ImportError:
        print("matplotlib not installed: skipped docs/benchmark.png")


def write_summary(raw: dict, tasks: list | None = None, trials: int | None = None) -> None:
    """Per-method averages across tasks, used for the README figure."""
    from frugal_eval.bench import summarize
    from frugal_eval.registry import list_methods

    metric = {"identify": ("savings", 100), "estimate": ("mae", 100), "online": (None, 100)}
    out = {}
    for problem, rows in raw.items():
        s = summarize(rows)
        kinds = {m.name: m.kind for m in list_methods(problem)}
        methods = dict.fromkeys(r["method"] for r in rows)
        tasks = list(dict.fromkeys(r["task"] for r in rows))
        out[problem] = {}
        for m in methods:
            if problem == "online":
                v = np.mean([s[(m, t)]["avg"] / s[(m, t)]["best_fixed"] for t in tasks])
            else:
                v = np.mean([s[(m, t)][metric[problem][0]] for t in tasks])
            out[problem][m] = {"kind": kinds[m], "value": round(float(v) * 100, 1)}
    out["_meta"] = {"tasks": tasks, "trials": trials}
    with open(os.path.join(ROOT, "benchmarks", "summary.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
