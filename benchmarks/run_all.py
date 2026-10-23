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

from frugal_eval.bench import (DEFAULT_TASKS, drift_markdown, duel_markdown, routing_markdown, run,  # noqa: E402
                               run_drift, run_duel, run_routing, to_markdown)

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
    "routing": ("Cost-aware routing on RouterBench",
                "RouterBench (Hu et al., 2024): 36,497 real prompts, 11 LLMs, the dollar cost of every answer. "
                "Every prompt arrives once; the method picks one model per prompt and learns from the score it "
                "gets, penalised by cost times the cost weight. Cells: quality and total cost relative to always "
                "using the single best model (GPT-4). Contextual methods use a prompt embedding."),
    "duel": ("Pairwise comparisons on Chatbot Arena",
             "Real human votes from Chatbot Arena (lmarena-ai/arena-human-preference-55k, Apache-2.0): 8 models "
             "per trial, either random or the 8 strongest (nearly tied). Each comparison is one simulated vote drawn "
             "from the real win rate of that pair. Target: the model with the highest average win rate. Cells: "
             "median votes until a model within 2 points of the best is certified (95% confidence, capped at "
             "300,000), and how often the pick is exactly right after a fixed number of votes."),
    "drift": ("Online selection when a model silently degrades",
              "10 random leaderboard models; questions arrive in order and, halfway through, the best model's "
              "answers are replaced by those of a model about half as good (a silent model swap behind an "
              "endpoint). Cells: total score as a share of what the best model of each half would have scored."),
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
        if problem == "routing":
            raw[problem] = run_routing(trials=min(args.trials, 5))
        elif problem == "duel":
            raw[problem] = run_duel(trials=args.trials)
        elif problem == "drift":
            raw[problem] = run_drift(trials=max(3, args.trials // 2))
        else:
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
    special = {"routing": routing_markdown, "duel": duel_markdown, "drift": drift_markdown}
    order = [p for p in ("identify", "estimate", "online", "routing", "drift", "duel") if p in raw]
    tasks = list(dict.fromkeys(r["task"] for p in order if p not in special for r in raw[p]))
    sections = []
    for problem in order:
        title, text = INTRO[problem]
        table = special[problem](raw[problem]) if problem in special else to_markdown(problem, raw[problem])
        sections.append(f"## {title}\n\n{text}\n\n{table}\n")
    header = ("# Benchmarks\n\nEvery method on the same real data: per-question results of 395 models from the "
              "Open LLM Leaderboard (via tinyBenchmarks, MIT license). "
              f"Tasks: {', '.join(tasks)}. {trials} trials per task. "
              "Reproduce with `python benchmarks/run_all.py`, or run the Benchmark workflow on GitHub.\n\n")
    with open(os.path.join(ROOT, "BENCHMARKS.md"), "w", encoding="utf-8") as f:
        f.write(header + "\n".join(sections))
    write_summary(raw, tasks=tasks, trials=trials)
    print("wrote BENCHMARKS.md and benchmarks/summary.json")
    try:
        import plot
        plot.main()
    except ImportError:
        print("matplotlib not installed: skipped docs/benchmark.png")


def write_summary(raw: dict, tasks: list | None = None, trials: int | None = None) -> None:
    """Per-method headline numbers: used for the README figure and the method zoo table."""
    from frugal_eval.bench import summarize
    from frugal_eval.registry import list_methods

    metric = {"identify": "savings", "estimate": "mae"}
    path = os.path.join(ROOT, "benchmarks", "summary.json")
    try:  # keep sections that this run did not recompute
        with open(path) as f:
            out = json.load(f)
    except (OSError, ValueError):
        out = {}
    for problem, rows in raw.items():
        kinds = {m.name: m.kind for m in list_methods("online" if problem in ("routing", "drift") else problem)}
        methods = list(dict.fromkeys(r["method"] for r in rows))
        s = summarize(rows)
        settings = list(dict.fromkeys(r["task"] for r in rows))
        out[problem] = {}
        for m in methods:
            if problem == "online":
                v = np.mean([s[(m, t)]["avg"] / s[(m, t)]["best_fixed"] for t in settings]) * 100
                entry = {"value": round(float(v), 1)}
            elif problem in metric:
                v = np.mean([s[(m, t)][metric[problem]] for t in settings]) * 100
                entry = {"value": round(float(v), 1)}
            elif problem == "routing":
                key = "cost weight 0.5" if (m, "cost weight 0.5") in s else settings[-1]
                entry = {"quality": round(s[(m, key)]["score_vs_best"] * 100, 1),
                         "cost": round(s[(m, key)]["cost_vs_best"] * 100, 1), "setting": key}
            elif problem == "drift":
                entry = {"value": round(float(np.mean([s[(m, t)]["vs_piecewise"] for t in settings])) * 100, 1)}
            else:  # duel
                rs = [r for r in rows if r["method"] == m and r["task"] == "arena/random"]
                budgets = sorted(int(k.split("_")[-1]) for k in rs[0] if k.startswith("right_at_"))
                entry = {"votes": float(np.median([r["comparisons"] for r in rs])),
                         "right_at": {str(n): round(float(np.mean([r[f"right_at_{n}"] for r in rs])) * 100)
                                      for n in budgets}}
            entry["kind"] = kinds.get(m, "")
            out[problem][m] = entry
    if tasks:
        out["_meta"] = {"tasks": tasks, "trials": trials}
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    update_readme_zoo(out)


def zoo_markdown(summary: dict) -> str:
    """The README method zoo: every method, its key number on the shared benchmarks, and its paper."""
    from frugal_eval.registry import list_methods

    rows = ["| Problem | Method | Type | Headline result | Paper |", "|---|---|---|---|---|"]

    def paper(m):
        return f"[{m.paper}]({m.url})" if m.url else m.paper

    for m in list_methods("identify"):
        v = summary.get("identify", {}).get(m.name)
        res = "-" if v is None else (f"{v['value']:.0f}% of calls (preset budget)" if m.kind == "fixed budget"
                                     else f"**{v['value']:.0f}%** fewer calls, certified")
        rows.append(f"| Find the best | `{m.name}` | {m.kind} | {res} | {paper(m)} |")
    for m in list_methods("estimate"):
        v = summary.get("estimate", {}).get(m.name)
        res = "-" if v is None else f"**{v['value']:.1f}** pts error from 100 questions"
        rows.append(f"| Estimate scores | `{m.name}` | {m.kind} | {res} | {paper(m)} |")
    for m in list_methods("online"):
        parts = []
        if m.name in summary.get("online", {}):
            parts.append(f"**{summary['online'][m.name]['value']:.0f}%** of best model")
        if m.name in summary.get("routing", {}):
            r = summary["routing"][m.name]
            parts.append(f"RouterBench: {r['quality']:.0f}% quality at {r['cost']:.0f}% cost")
        if m.name in summary.get("drift", {}):
            parts.append(f"drift: {summary['drift'][m.name]['value']:.0f}%")
        rows.append(f"| Online selection | `{m.name}` | {m.kind} | {'; '.join(parts) or '-'} | {paper(m)} |")
    for m in list_methods("duel"):
        v = summary.get("duel", {}).get(m.name)
        if v is None:
            res = "-"
        else:
            right = ", ".join(f"{p}% right at {int(n):,}" for n, p in v["right_at"].items())
            res = f"**{v['votes']:,.0f}** votes to certify; {right}"
        rows.append(f"| Pairwise | `{m.name}` | {m.kind} | {res} | {paper(m)} |")
    return "\n".join(rows)


def update_readme_zoo(summary: dict) -> None:
    path = os.path.join(ROOT, "README.md")
    with open(path, encoding="utf-8") as f:
        text = f.read()
    start, end = "<!-- ZOO:START -->", "<!-- ZOO:END -->"
    if start in text and end in text:
        a, b = text.index(start) + len(start), text.index(end)
        text = text[:a] + "\n" + zoo_markdown(summary) + "\n" + text[b:]
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        print("updated the README method zoo")


if __name__ == "__main__":
    main()
