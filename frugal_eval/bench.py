"""Benchmark every registered method on the same tasks.

    from frugal_eval.bench import run, to_markdown
    rows = run("identify", tasks=["gsm8k", "arc_challenge"], trials=20)
    print(to_markdown("identify", rows))

Each trial draws a fresh group of real leaderboard models (and, for methods
that need it, a disjoint set of prior models), so every method sees exactly the
same problems.
"""
from __future__ import annotations

import os
from multiprocessing import Pool
from typing import Any, Sequence

import numpy as np

from .estimate import estimate
from .identify import identify
from .online import run_online
from .registry import list_methods
from .tasks import leaderboard

DEFAULT_TASKS = ("gsm8k", "arc_challenge", "winogrande", "truthfulqa")


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def _one(job: tuple) -> dict:
    problem, method, task_name, trial, cfg = job
    task = leaderboard(task_name, n_candidates=cfg["k"], seed=trial, pick=cfg.get("pick", "random"))
    truth = task.truth
    row = dict(problem=problem, method=method, task=task_name, trial=trial)
    if problem == "identify":
        eps = cfg.get("epsilon", 0.0)
        kind = next(m.kind for m in list_methods("identify") if m.name == method)
        budget = int(cfg.get("budget_fraction", 0.3) * task.n_candidates * task.n_examples) \
            if kind == "fixed budget" else None
        r = identify(task, method=method, delta=cfg.get("delta", 0.05), epsilon=eps, budget=budget, seed=trial)
        row.update(correct=bool(truth[r.best] >= truth.max() - eps - 1e-12), savings=r.savings)
    elif problem == "estimate":
        r = estimate(task, method=method, budget=cfg.get("budget", 100), seed=trial)
        row.update(mae=r.mae, spearman=_spearman(r.estimates, truth),
                   top1=bool(np.argmax(r.estimates) == np.argmax(truth)))
    else:
        r = run_online(task, method=method, seed=trial)
        T = len(r.choices)
        row.update(avg=r.average_reward, best_fixed=r.best_fixed_reward / T, oracle=r.oracle_reward / T,
                   regret_per_request=r.regret / T)
    return row


def run(problem: str, methods: Sequence[str] | None = None, tasks: Sequence[str] = DEFAULT_TASKS,
        trials: int = 20, workers: int | None = None, **cfg: Any) -> list[dict]:
    """Run ``trials`` trials of every method (default: all registered for the problem) on every task."""
    cfg.setdefault("k", 10)
    methods = list(methods) if methods else [m.name for m in list_methods(problem)]
    jobs = [(problem, m, t, i, cfg) for m in methods for t in tasks for i in range(trials)]
    workers = workers or os.cpu_count() or 1
    if workers == 1:
        return [_one(j) for j in jobs]
    with Pool(workers) as pool:
        return pool.map(_one, jobs, chunksize=2)


def summarize(rows: list[dict]) -> dict[tuple[str, str], dict]:
    out: dict[tuple[str, str], dict] = {}
    for r in rows:
        out.setdefault((r["method"], r["task"]), []).append(r)
    summary = {}
    for key, rs in out.items():
        metrics = [k for k in rs[0] if k not in {"problem", "method", "task", "trial"}]
        summary[key] = {m: float(np.nanmean([float(r[m]) for r in rs])) for m in metrics}
    return summary


def to_markdown(problem: str, rows: list[dict]) -> str:
    s = summarize(rows)
    methods = list(dict.fromkeys(r["method"] for r in rows))
    tasks = list(dict.fromkeys(r["task"] for r in rows))
    info = {m.name: m for m in list_methods(problem)}
    if problem == "identify":
        head = "| Method | Type | " + " | ".join(tasks) + " | Avg saved |"

        def cell(d):
            return f"{d['savings']:.0%} saved, {d['correct']:.0%} right"

        def avg(m):
            # fixed-budget methods save a preset amount by design; rank them after the others
            base = np.mean([s[(m, t)]["savings"] for t in tasks])
            return base - (10.0 if info[m].kind == "fixed budget" else 0.0)

        def fmt_avg(v):
            return f"**{v:.0%}**" if v > -5 else f"{v + 10:.0%} (preset)"
    elif problem == "estimate":
        head = "| Method | Type | " + " | ".join(tasks) + " | Avg error |"
        cell = lambda d: f"{d['mae'] * 100:.1f} pts (rank ρ {d['spearman']:.2f})"
        avg = lambda m: np.mean([s[(m, t)]["mae"] for t in tasks])
        fmt_avg = lambda v: f"**{v * 100:.1f} pts**"
    else:
        head = "| Method | Type | " + " | ".join(tasks) + " | Avg vs best single |"
        cell = lambda d: f"{d['avg']:.3f} ({d['avg'] / d['best_fixed']:.0%})"
        avg = lambda m: np.mean([s[(m, t)]["avg"] / s[(m, t)]["best_fixed"] for t in tasks])
        fmt_avg = lambda v: f"**{v:.0%}**"
    lines = [head, "|" + "---|" * (len(tasks) + 3)]
    order = sorted(methods, key=avg, reverse=(problem != "estimate"))
    for m in order:
        cells = " | ".join(cell(s[(m, t)]) for t in tasks)
        lines.append(f"| `{m}` | {info[m].kind} | {cells} | {fmt_avg(avg(m))} |")
    return "\n".join(lines)


# ---------------------------------------------------------------- routing with costs
ROUTING_WEIGHTS = (0.0, 0.5, 1.0)


_TASK_CACHE: dict = {}


def _routerbench_task():
    if "routerbench" not in _TASK_CACHE:  # build once per worker process
        from .tasks import make

        _TASK_CACHE["routerbench"] = make("routerbench")
    return _TASK_CACHE["routerbench"]


def _routing_one(job: tuple) -> dict:
    method, weight, trial = job
    task = _routerbench_task()
    S, C = task.source.scores, task.costs
    ref = int(np.argmax(S.mean(1)))  # best single model (GPT-4 on RouterBench)
    r = run_online(task, method=method, seed=trial, cost_weight=weight)
    stream = r.examples
    return dict(problem="routing", method=method, task=f"cost weight {weight:g}", trial=trial,
                score_vs_best=r.average_score / S[ref, stream].mean(),
                cost_vs_best=r.total_cost / C[ref, stream].sum(), score=r.average_score)


def run_routing(methods: Sequence[str] | None = None, weights: Sequence[float] = ROUTING_WEIGHTS,
                trials: int = 5, workers: int | None = None) -> list[dict]:
    """Online methods on RouterBench, trading answer quality against dollar cost."""
    from .datasets import load_routerbench

    load_routerbench()  # download and embed once, before the worker processes start
    methods = list(methods) if methods else [m.name for m in list_methods("online")]
    jobs = [(m, w, i) for m in methods for w in weights for i in range(trials)]
    workers = workers or os.cpu_count() or 1
    if workers == 1:
        return [_routing_one(j) for j in jobs]
    with Pool(workers) as pool:
        return pool.map(_routing_one, jobs, chunksize=1)


def routing_markdown(rows: list[dict]) -> str:
    s = summarize(rows)
    methods = list(dict.fromkeys(r["method"] for r in rows))
    settings = list(dict.fromkeys(r["task"] for r in rows))
    info = {m.name: m for m in list_methods("online")}
    lines = ["| Method | Type | " + " | ".join(settings) + " |", "|" + "---|" * (len(settings) + 2)]
    key = settings[1] if len(settings) > 1 else settings[0]
    for m in sorted(methods, key=lambda m: -(s[(m, key)]["score_vs_best"] - s[(m, key)]["cost_vs_best"])):
        cells = " | ".join(f"{s[(m, t)]['score_vs_best']:.0%} quality, {s[(m, t)]['cost_vs_best']:.0%} cost"
                           for t in settings)
        lines.append(f"| `{m}` | {info[m].kind} | {cells} |")
    return "\n".join(lines)
