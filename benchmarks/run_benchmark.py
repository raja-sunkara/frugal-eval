"""Replay frugal-eval on real Open LLM Leaderboard results.

For each task we repeatedly pick a group of models, hide their scores, and let
each method decide which examples to "evaluate". We record how many calls it
used and whether it picked the model a full evaluation would pick.

    python benchmarks/run_benchmark.py            # full run (a few minutes)
    python benchmarks/run_benchmark.py --quick    # smoke test
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from frugal_eval import find_best  # noqa: E402
from leaderboard import TASKS, load  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
K = 10
DELTA = 0.05


def groups(task: str, setting: str, trials: int, seed: int) -> list[np.ndarray]:
    scores, _ = load(task)
    rng = np.random.default_rng(seed)
    if setting == "random":
        return [rng.choice(len(scores), K, replace=False) for _ in range(trials)]
    # "close": the 10 strongest models on the task. Their scores are close together,
    # so this is the hard case. Trials differ by the random example order.
    top = np.argsort(-scores.mean(1))[:K]
    return [top for _ in range(trials)]


def one_trial(args):
    task, setting, trial, idx, method, epsilon, budget = args
    scores, _ = load(task)
    M = scores[idx]
    true = M.mean(1)
    algo = {"uniform": "uniform", "lucb": "lucb", "halving": "successive_halving"}[method]
    r = find_best(M, algorithm=algo, delta=DELTA, epsilon=epsilon, budget=budget, seed=trial)
    correct = bool(true[r.best] >= true.max() - epsilon - 1e-12)
    return dict(task=task, setting=setting, trial=trial, method=method, epsilon=epsilon,
                calls=r.total_calls, full=r.full_calls, correct=correct)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--tasks", nargs="+", default=["gsm8k", "arc_challenge", "winogrande", "truthfulqa"],
                    choices=TASKS, help="hellaswag and mmlu are large (10k+ examples) and slow to replay")
    args = ap.parse_args()
    tasks = args.tasks
    trials = 5 if args.quick else args.trials

    t0 = time.time()
    jobs = []
    for task in tasks:
        for setting in ("random", "close"):
            for trial, idx in enumerate(groups(task, setting, trials, seed=hash(task) % 2**32)):
                for epsilon in (0.0, 0.01):
                    for method in ("uniform", "lucb"):
                        jobs.append((task, setting, trial, idx, method, epsilon, None))
    with Pool(args.workers) as pool:
        rows = pool.map(one_trial, jobs, chunksize=4)

    # Successive halving gets LUCB's average budget, to show what you lose without a guarantee.
    sh_jobs = []
    for task in tasks:
        for setting in ("random", "close"):
            lucb = [r for r in rows if r["task"] == task and r["setting"] == setting
                    and r["method"] == "lucb" and r["epsilon"] == 0.0]
            budget = int(np.mean([r["calls"] for r in lucb]))
            for trial, idx in enumerate(groups(task, setting, trials, seed=hash(task) % 2**32)):
                sh_jobs.append((task, setting, trial, idx, "halving", 0.0, budget))
    with Pool(args.workers) as pool:
        rows += pool.map(one_trial, sh_jobs, chunksize=4)

    summary = []
    for task in tasks:
        for setting in ("random", "close"):
            for method, epsilon in (("uniform", 0.0), ("lucb", 0.0), ("lucb", 0.01), ("uniform", 0.01),
                                    ("halving", 0.0)):
                sel = [r for r in rows if r["task"] == task and r["setting"] == setting
                       and r["method"] == method and r["epsilon"] == epsilon]
                if not sel:
                    continue
                frac = np.array([r["calls"] / r["full"] for r in sel])
                summary.append(dict(task=task, setting=setting, method=method, epsilon=epsilon,
                                    trials=len(sel), calls_fraction=float(frac.mean()),
                                    calls_fraction_median=float(np.median(frac)),
                                    accuracy=float(np.mean([r["correct"] for r in sel])),
                                    n_examples=load(task)[0].shape[1]))
    out = os.path.join(HERE, "results")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(dict(k=K, delta=DELTA, trials=trials, rows=summary), f, indent=1)
    print(f"done in {time.time() - t0:.0f}s, {len(rows)} runs")
    for s in summary:
        print(f"{s['task']:<14}{s['setting']:<7}{s['method']:<8}eps={s['epsilon']:<5}"
              f"calls={s['calls_fraction']:6.1%}  picked best={s['accuracy']:6.1%}")


if __name__ == "__main__":
    main()
