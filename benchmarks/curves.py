"""Trade-off curves: how good each method's answer is at every budget.

    python benchmarks/curves.py                 # run all four experiments, write docs/curves*.png
    python benchmarks/curves.py --trials 3      # quick check
    python benchmarks/curves.py --only identify estimate --raw-only out/curves.json   # one slice (CI)
    python benchmarks/curves.py --merge out/curves-*.json                             # combine and plot

Experiments
  identify  P(the method's current pick is the true best) vs % of the full evaluation cost
  estimate  mean absolute error of the score estimates vs questions per model
  routing   RouterBench quality vs dollar cost as the cost weight is swept (Pareto curve)
  scaling   calls saved vs number of candidates (5, 10, 20, 50)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from frugal_eval import estimate, identify, run_online  # noqa: E402
from frugal_eval.tasks import leaderboard  # noqa: E402

TASKS = ("gsm8k", "arc_challenge", "winogrande", "truthfulqa")
FRACTIONS = (0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
ANYTIME = ("uniform", "lucb", "track_and_stop", "eb_tc", "top_two_ts")
BUDGET_METHODS = ("successive_halving",)
EST_BUDGETS = (10, 25, 50, 100, 200, 400)
EST_METHODS = ("random", "stratified", "anchor_points", "irt")
ROUTE_METHODS = ("linucb", "lin_ts", "thompson", "kl_ucb", "epsilon_greedy")
COST_WEIGHTS = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0)
SCALE_K = (5, 10, 20, 50)
SCALE_METHODS = ("uniform", "lucb", "track_and_stop", "eb_tc")


# ------------------------------------------------------------------ experiments
def _identify_job(job):
    method, task_name, trial = job
    task = leaderboard(task_name, n_candidates=10, seed=trial)
    truth = task.truth
    full = task.n_candidates * task.n_examples
    best = int(np.argmax(truth))
    out = {}
    if method in BUDGET_METHODS:
        for f in FRACTIONS:
            r = identify(task, method=method, budget=max(task.n_candidates, int(f * full)), seed=trial)
            out[f] = bool(r.best == best)
    else:
        r = identify(task, method=method, seed=trial, record_history=True)
        calls = np.array([h[0] for h in r.history])
        picks = np.array([h[1] for h in r.history])
        for f in FRACTIONS:
            i = np.searchsorted(calls, f * full, side="right") - 1
            pick = picks[i] if i >= 0 else -1
            out[f] = bool(pick == best)
    return dict(exp="identify", method=method, task=task_name, trial=trial, right={str(k): v for k, v in out.items()})


def _estimate_job(job):
    method, task_name, trial, budget = job
    task = leaderboard(task_name, n_candidates=10, seed=trial)
    r = estimate(task, method=method, budget=budget, seed=trial)
    return dict(exp="estimate", method=method, task=task_name, trial=trial, budget=budget, mae=r.mae)


_RB: dict = {}


def _routing_job(job):
    method, weight, trial = job
    if "task" not in _RB:
        from frugal_eval.tasks import make

        _RB["task"] = make("routerbench")
    task = _RB["task"]
    S, C = task.source.scores, task.costs
    ref = int(np.argmax(S.mean(1)))
    r = run_online(task, method=method, cost_weight=weight, seed=trial)
    st = r.examples
    return dict(exp="routing", method=method, weight=weight, trial=trial,
                quality=r.average_score / S[ref, st].mean(), cost=r.total_cost / C[ref, st].sum())


def _scaling_job(job):
    method, task_name, k, trial = job
    task = leaderboard(task_name, n_candidates=k, seed=trial)
    r = identify(task, method=method, seed=trial)
    return dict(exp="scaling", method=method, task=task_name, k=k, trial=trial, savings=r.savings,
                right=bool(r.best == int(np.argmax(task.truth))))


def _pool(fn, jobs, workers):
    if workers == 1:
        return [fn(j) for j in jobs]
    with Pool(workers) as pool:
        return pool.map(fn, jobs, chunksize=1)


def run(only, trials, workers):
    rows = []
    if "identify" in only:
        jobs = [(m, t, i) for m in ANYTIME + BUDGET_METHODS for t in TASKS[:4] for i in range(trials)]
        rows += _pool(_identify_job, jobs, workers)
    if "estimate" in only:
        jobs = [(m, t, i, b) for m in EST_METHODS for t in TASKS for i in range(trials) for b in EST_BUDGETS]
        rows += _pool(_estimate_job, jobs, workers)
    if "routing" in only:
        from frugal_eval.datasets import load_routerbench

        load_routerbench()
        jobs = [(m, w, i) for m in ROUTE_METHODS for w in COST_WEIGHTS for i in range(max(1, trials // 4))]
        rows += _pool(_routing_job, jobs, workers)
        from frugal_eval.tasks import make

        task = make("routerbench")
        S, C = task.source.scores, task.costs
        ref = int(np.argmax(S.mean(1)))
        for k, name in enumerate(task.names):  # each single model as a reference point
            rows.append(dict(exp="routing_single", method=name, quality=float(S[k].mean() / S[ref].mean()),
                             cost=float(C[k].sum() / C[ref].sum())))
    if "scaling" in only:
        jobs = [(m, t, k, i) for m in SCALE_METHODS for t in ("gsm8k", "arc_challenge", "truthfulqa")
                for k in SCALE_K for i in range(max(2, trials // 2))]
        rows += _pool(_scaling_job, jobs, workers)
    return rows


# ------------------------------------------------------------------ plotting
THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", grid="#e4e3df", base="#8f8e88",
                  series=["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]),
    "dark": dict(surface="#1a1a19", ink="#f0efec", ink2="#c3c2b7", grid="#34332f", base="#8a897f",
                 series=["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#9085e9"]),
}
BASELINES = {"uniform", "random"}


def _style(ax, c, title, xlabel, ylabel):
    from matplotlib.ticker import NullFormatter, NullLocator

    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_facecolor(c["surface"])
    ax.set_title(title, color=c["ink"], loc="left", fontsize=11.5, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, color=c["ink2"], fontsize=9)
    ax.set_ylabel(ylabel, color=c["ink2"], fontsize=9)
    ax.tick_params(colors=c["ink2"], labelsize=8.5, length=0)
    ax.grid(color=c["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(c["grid"])


def _colors(methods, c):
    out, i = {}, 0
    for m in methods:
        if m in BASELINES:
            out[m] = c["base"]
        else:
            out[m] = c["series"][i % len(c["series"])]
            i += 1
    return out


def _legend(ax, c):
    leg = ax.legend(frameon=False, fontsize=8.5, labelcolor=c["ink"], handlelength=1.6, loc="best")
    for t in leg.get_texts():
        t.set_fontfamily("DejaVu Sans Mono")


def plot(rows, out_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    have = {r["exp"] for r in rows}
    panels = [p for p in ("identify", "estimate", "routing", "scaling") if p in have]
    for theme, c in THEMES.items():
        fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 4.2), facecolor=c["surface"],
                                 squeeze=False)
        for ax, p in zip(axes[0], panels):
            if p == "identify":
                rs = [r for r in rows if r["exp"] == "identify"]
                methods = list(dict.fromkeys(r["method"] for r in rs))
                col = _colors(methods, c)
                for m in methods:
                    y = [100 * np.mean([r["right"][str(f)] for r in rs if r["method"] == m]) for f in FRACTIONS]
                    ax.plot([100 * f for f in FRACTIONS], y, label=m, color=col[m], linewidth=2,
                            marker="o", markersize=4, linestyle="--" if m in BASELINES else "-")
                _style(ax, c, "Find the best", "% of full evaluation cost", "picked the true best (%)")
                ax.set_ylim(0, 102)
            elif p == "estimate":
                rs = [r for r in rows if r["exp"] == "estimate"]
                methods = list(dict.fromkeys(r["method"] for r in rs))
                col = _colors(methods, c)
                for m in methods:
                    y = [100 * np.mean([r["mae"] for r in rs if r["method"] == m and r["budget"] == b])
                         for b in EST_BUDGETS]
                    ax.plot(EST_BUDGETS, y, label=m, color=col[m], linewidth=2, marker="o", markersize=4,
                            linestyle="--" if m in BASELINES else "-")
                ax.set_xscale("log")
                ax.set_xticks(EST_BUDGETS)
                ax.set_xticklabels([str(b) for b in EST_BUDGETS])
                _style(ax, c, "Estimate scores", "questions per model (log scale)", "mean abs error (points)")
                ax.set_ylim(bottom=0)
            elif p == "routing":
                rs = [r for r in rows if r["exp"] == "routing"]
                methods = list(dict.fromkeys(r["method"] for r in rs))
                col = _colors(methods, c)
                singles = [r for r in rows if r["exp"] == "routing_single"]
                ax.scatter([100 * r["cost"] for r in singles], [100 * r["quality"] for r in singles],
                           s=28, color=c["base"], zorder=3, label="single models")
                for r in singles:
                    if r["quality"] > 0.95:  # label the reference model only
                        ax.annotate(r["method"].split("/")[-1].replace("-1106-preview", ""),
                                    (100 * r["cost"], 100 * r["quality"]), textcoords="offset points",
                                    xytext=(-4, 6), ha="right", fontsize=7.5, color=c["ink2"])
                for m in methods:
                    pts = []
                    for w in COST_WEIGHTS:
                        sel = [r for r in rs if r["method"] == m and r["weight"] == w]
                        if sel:
                            pts.append((100 * np.mean([r["cost"] for r in sel]), 100 * np.mean([r["quality"] for r in sel])))
                    pts.sort()
                    ax.plot([p_[0] for p_ in pts], [p_[1] for p_ in pts], label=m, color=col[m], linewidth=2,
                            marker="o", markersize=4)
                _style(ax, c, "Routing on RouterBench", "cost (% of always using GPT-4)", "quality (% of GPT-4)")
            else:
                rs = [r for r in rows if r["exp"] == "scaling"]
                methods = list(dict.fromkeys(r["method"] for r in rs))
                col = _colors(methods, c)
                for m in methods:
                    y = [100 * np.mean([r["savings"] for r in rs if r["method"] == m and r["k"] == k]) for k in SCALE_K]
                    ax.plot(SCALE_K, y, label=m, color=col[m], linewidth=2, marker="o", markersize=4,
                            linestyle="--" if m in BASELINES else "-")
                ax.set_xscale("log")
                ax.set_xticks(SCALE_K)
                ax.set_xticklabels([str(k) for k in SCALE_K])
                _style(ax, c, "More candidates, bigger savings", "number of candidates (log scale)",
                       "calls saved, certified (%)")
                ax.set_ylim(bottom=0)
            _legend(ax, c)
        fig.text(0.01, 0.01, "Real per-question results of Open LLM Leaderboard models and RouterBench · "
                 "dashed gray = baseline", color=c["ink2"], fontsize=8.5)
        fig.tight_layout(rect=(0, 0.04, 1, 1), w_pad=3)
        suffix = "" if theme == "light" else "-dark"
        fig.savefig(os.path.join(out_dir, f"curves{suffix}.png"), dpi=170, facecolor=c["surface"])
        plt.close(fig)
    print(f"wrote {out_dir}/curves.png and curves-dark.png")


def headline(rows) -> dict:
    """A few numbers quoted in the README."""
    out = {}
    rs = [r for r in rows if r["exp"] == "identify"]
    for m in dict.fromkeys(r["method"] for r in rs):
        out[f"identify/{m}@20%"] = round(100 * np.mean([r["right"]["0.2"] for r in rs if r["method"] == m]), 1)
    rs = [r for r in rows if r["exp"] == "estimate"]
    for m in dict.fromkeys(r["method"] for r in rs):
        out[f"estimate/{m}@50"] = round(100 * np.mean([r["mae"] for r in rs if r["method"] == m and r["budget"] == 50]), 2)
    rs = [r for r in rows if r["exp"] == "scaling"]
    for m in dict.fromkeys(r["method"] for r in rs):
        for k in SCALE_K:
            out[f"scaling/{m}@{k}"] = round(100 * np.mean([r["savings"] for r in rs if r["method"] == m and r["k"] == k]), 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--only", nargs="+", default=["identify", "estimate", "routing", "scaling"])
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--raw-only", metavar="PATH")
    ap.add_argument("--merge", nargs="+", metavar="RAW")
    args = ap.parse_args()
    if args.merge:
        rows = [r for p in args.merge for r in json.load(open(p))]
    else:
        t0 = time.time()
        rows = run(args.only, args.trials, args.workers)
        print(f"{len(rows)} runs in {time.time() - t0:.0f}s")
        if args.raw_only:
            os.makedirs(os.path.dirname(os.path.abspath(args.raw_only)), exist_ok=True)
            json.dump(rows, open(args.raw_only, "w"))
            print(f"wrote {args.raw_only}")
            return
    os.makedirs(os.path.join(ROOT, "docs"), exist_ok=True)
    plot(rows, os.path.join(ROOT, "docs"))
    h = headline(rows)
    json.dump(h, open(os.path.join(ROOT, "benchmarks", "curves_summary.json"), "w"), indent=1)
    print(json.dumps(h, indent=1))


if __name__ == "__main__":
    main()
