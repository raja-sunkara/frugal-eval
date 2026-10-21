"""Benchmark every method on real Open LLM Leaderboard results and write BENCHMARKS.md.

    python benchmarks/run_all.py              # 20 trials per task (about 15 minutes on 2 cores)
    python benchmarks/run_all.py --trials 3   # quick check
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
    args = ap.parse_args()
    sections, raw = [], {}
    for problem in args.problems:
        t0 = time.time()
        rows = run(problem, tasks=args.tasks, trials=args.trials)
        raw[problem] = rows
        title, text = INTRO[problem]
        sections.append(f"## {title}\n\n{text}\n\n{to_markdown(problem, rows)}\n")
        print(f"{problem}: {len(rows)} runs in {time.time() - t0:.0f}s", flush=True)
    out_dir = os.path.join(ROOT, "benchmarks", "results")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "raw.json"), "w") as f:
        json.dump(raw, f)
    header = ("# Benchmarks\n\nEvery method on the same real data: per-question results of 395 models from the "
              "Open LLM Leaderboard (via tinyBenchmarks, MIT license). "
              f"{args.trials} trials per task. Reproduce with `python benchmarks/run_all.py`.\n\n")
    with open(os.path.join(ROOT, "BENCHMARKS.md"), "w") as f:
        f.write(header + "\n".join(sections))
    print("wrote BENCHMARKS.md")


if __name__ == "__main__":
    main()
