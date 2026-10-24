"""frugal-eval interactive demo (Hugging Face Space).

Every method runs live on real data: per-question results of 395 Open LLM
Leaderboard models, RouterBench (11 LLMs with real prices) and Chatbot Arena
human votes.
"""
from __future__ import annotations

import random
import time

import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import frugal_eval as fe  # noqa: E402

REPO = "https://github.com/raja-sunkara/frugal-eval"
BENCHMARKS = {
    "GSM8K (grade-school math, 1,319 questions)": "gsm8k",
    "ARC-Challenge (science, 1,172)": "arc_challenge",
    "Winogrande (commonsense, 1,267)": "winogrande",
    "TruthfulQA (truthfulness, 817)": "truthfulqa",
    "HellaSwag (sentence completion, 10,042)": "hellaswag",
    "MMLU (57 subjects, 14,042)": "mmlu",
}
FAST_IDENTIFY = ["uniform", "lucb", "successive_elimination", "eb_tc", "lil_ucb"]
ALL_IDENTIFY = [m.name for m in fe.list_methods("identify") if m.kind != "fixed budget"]
PAPERS = {m.name: m.paper for m in fe.list_methods()}

# palette (validated categorical order, light surface)
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]
BASE = "#a9a8a2"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})


def _ax(figsize=(7.5, 4.2)):
    fig, ax = plt.subplots(figsize=figsize, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    return fig, ax


def _title(ax, text):
    ax.set_title(text, color=INK, loc="left", fontsize=12, fontweight="bold", pad=10)


# ---------------------------------------------------------------- 1. find the best
def run_identify(bench_label, k, seed, confidence, methods):
    if not methods:
        raise gr.Error("Pick at least one method.")
    bench = BENCHMARKS[bench_label]
    task = fe.tasks.make(f"leaderboard/{bench}", n_candidates=int(k), seed=int(seed))
    truth = task.truth
    best = int(np.argmax(truth))
    delta = 1 - float(confidence)
    rows, results = [], {}
    for m in methods:
        t0 = time.time()
        r = fe.identify(task, method=m, delta=delta, seed=int(seed))
        results[m] = r
        rows.append({"method": m, "calls used": f"{r.total_calls:,} of {r.full_calls:,}",
                     "saved": f"{r.savings:.0%}", "picked": r.best_name,
                     "right?": "yes" if r.best == best else "NO", "time": f"{time.time() - t0:.1f}s",
                     "paper": PAPERS.get(m, "")})
    table = pd.DataFrame(rows)

    # chart 1: calls saved per method
    fig1, ax = _ax((7.5, 0.55 * len(methods) + 1.4))
    order = sorted(methods, key=lambda m: results[m].savings)
    vals = [100 * results[m].savings for m in order]
    colors = [BASE if m == "uniform" else SERIES[0] for m in order]
    ax.barh(range(len(order)), vals, color=colors, height=0.6, edgecolor=SURFACE, linewidth=2)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontfamily="DejaVu Sans Mono", color=INK)
    for i, v in enumerate(vals):
        ax.text(v + 1, i, f"{v:.0f}%", va="center", color=INK, fontsize=9)
    ax.set_xlim(0, max(100, max(vals) + 10))
    ax.set_xlabel("calls saved vs. running every model on every question (%)", color=INK2)
    _title(ax, "Calls saved, with the winner certified")
    fig1.tight_layout()

    # chart 2: where the main adaptive method spent its questions
    show = "lucb" if "lucb" in results else methods[0]
    r = results[show]
    fig2, ax = _ax((7.5, 0.42 * task.n_candidates + 1.4))
    order = np.argsort(truth)
    n = r.pulls[order]
    cols = [SERIES[1] if i == best else SERIES[0] for i in order]
    ax.barh(range(len(order)), n, color=cols, height=0.6, edgecolor=SURFACE, linewidth=2)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f"{task.names[i][:28]}  ({truth[i]:.0%})" for i in order], color=INK, fontsize=8.5)
    ax.axvline(task.n_examples, color=INK2, linestyle=(0, (3, 3)), linewidth=1)
    ax.text(task.n_examples, len(order) - 0.4, " all questions", color=INK2, fontsize=8, va="bottom")
    ax.set_xlabel("questions asked of each model (true accuracy in brackets)", color=INK2)
    _title(ax, f"Questions asked of each model by {show}")
    fig2.tight_layout()

    lucb_line = ""
    if "lucb" in results:
        lr = results["lucb"]
        lucb_line = (f"**LUCB** certified **{lr.best_name}** as the best of {task.n_candidates} models "
                     f"after **{lr.total_calls:,}** of {lr.full_calls:,} calls (**{lr.savings:.0%} saved**). ")
    head = (f"{lucb_line}A full evaluation would also pick **{task.names[best]}** "
            f"({truth[best]:.1%}). Data: real per-question results from the Open LLM Leaderboard.")
    return head, fig1, fig2, table


# ---------------------------------------------------------------- 2. estimate
def run_estimate(bench_label, k, seed, budget):
    bench = BENCHMARKS[bench_label]
    task = fe.tasks.make(f"leaderboard/{bench}", n_candidates=int(k), seed=int(seed))
    truth = task.truth
    fig, ax = _ax((7.0, 5.0))
    lo, hi = truth.min() - 0.05, truth.max() + 0.05
    ax.plot([lo, hi], [lo, hi], color=INK2, linestyle=(0, (3, 3)), linewidth=1)
    rows = []
    methods = [m.name for m in fe.list_methods("estimate")]
    for i, m in enumerate(methods):
        r = fe.estimate(task, method=m, budget=int(budget), seed=int(seed))
        color = BASE if m == "random" else SERIES[(i - 1) % len(SERIES)]
        ax.scatter(100 * truth, 100 * r.estimates, s=46, color=color, label=f"{m}  ({100 * r.mae:.1f} pts)",
                   edgecolor=SURFACE, linewidth=1.5, zorder=3)
        rows.append({"method": m, "mean abs error (points)": round(100 * r.mae, 2),
                     "questions run per model": int(budget), "calls saved": f"{r.savings:.1%}",
                     "paper": PAPERS.get(m, "")})
    ax.plot([100 * lo, 100 * hi], [100 * lo, 100 * hi], color=INK2, linestyle=(0, (3, 3)), linewidth=1)
    ax.set_xlim(100 * lo, 100 * hi)
    ax.set_ylim(100 * lo, 100 * hi)
    ax.set_xlabel("true score on the full benchmark (%)", color=INK2)
    ax.set_ylabel("estimated score (%)", color=INK2)
    leg = ax.legend(frameon=False, fontsize=9, loc="upper left")
    for t in leg.get_texts():
        t.set_fontfamily("DejaVu Sans Mono")
    _title(ax, f"Estimated vs true score, from {int(budget)} questions per model")
    fig.tight_layout()
    table = pd.DataFrame(rows).sort_values("mean abs error (points)")
    best = table.iloc[0]
    head = (f"With only **{int(budget)}** of {task.n_examples:,} questions per model, **{best['method']}** "
            f"estimates every score within **{best['mean abs error (points)']:.1f} points** on average. "
            "Points on the dashed line are perfect estimates. This is one random draw; the repo's benchmark "
            "averages many.")
    return head, fig, table


# ---------------------------------------------------------------- 3. routing
_RB: dict = {}


def _routerbench():
    if "task" not in _RB:
        _RB["task"] = fe.tasks.make("routerbench")
    return _RB["task"]


def run_routing(method, cost_weight, n_requests, seed):
    try:
        full = _routerbench()
    except Exception as exc:  # download problems
        raise gr.Error(f"Could not load RouterBench: {exc}")
    rng = np.random.default_rng(int(seed))
    cols = np.sort(rng.choice(full.n_examples, int(n_requests), replace=False))
    task = fe.Task("routerbench", fe.MatrixSource(full.source.scores[:, cols], full.names),
                   features=full.features[cols], truth=full.source.scores[:, cols].mean(1),
                   costs=full.costs[:, cols])
    S, C = task.source.scores, task.costs
    ref = int(np.argmax(S.mean(1)))
    r = fe.run_online(task, method=method, cost_weight=float(cost_weight), seed=int(seed))
    q = r.average_score / S[ref].mean()
    c = r.total_cost / C[ref].sum()

    fig1, ax = _ax((7.0, 4.6))
    singles_q = S.mean(1) / S[ref].mean()
    singles_c = C.sum(1) / C[ref].sum()
    ax.scatter(100 * singles_c, 100 * singles_q, s=34, color=BASE, zorder=3, label="each model used alone")
    ax.annotate(full.names[ref], (100 * singles_c[ref], 100 * singles_q[ref]), textcoords="offset points",
                xytext=(-6, -14), ha="right", fontsize=8, color=INK2)
    ax.scatter([100 * c], [100 * q], s=120, color=SERIES[1], zorder=4, edgecolor=SURFACE, linewidth=2,
               label=f"{method} (this run)")
    ax.set_xlabel(f"cost (% of always using {full.names[ref]})", color=INK2)
    ax.set_ylabel("quality (% of the same)", color=INK2)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    _title(ax, "Quality vs cost on RouterBench")
    fig1.tight_layout()

    fig2, ax = _ax((7.0, 4.6))
    share = r.counts / len(r.choices)
    order = np.argsort(share)
    ax.barh(range(len(order)), 100 * share[order], color=SERIES[0], height=0.6, edgecolor=SURFACE, linewidth=2)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([full.names[i].split("/")[-1] for i in order], fontsize=8.5, color=INK)
    ax.set_xlabel("share of requests sent to each model (%)", color=INK2)
    _title(ax, "Where the router sent requests")
    fig2.tight_layout()

    head = (f"**{method}** learned from scratch over **{int(n_requests):,}** real prompts and reached "
            f"**{q:.0%} of {full.names[ref]}'s quality at {c:.0%} of its cost** "
            f"(cost weight {float(cost_weight):g}). Raise the cost weight to save more; lower it for quality.")
    return head, fig1, fig2


# ---------------------------------------------------------------- 4. pairwise
def run_duel(pick, k, seed, method, epsilon, max_votes):
    task = fe.duel.arena(int(k), seed=int(seed), pick="top" if pick.startswith("Top") else "random")
    r = fe.find_best_pairwise(task, method=method, epsilon=float(epsilon), budget=int(max_votes), seed=int(seed))
    true = task.borda
    rows = [{"model": task.names[i], "true win rate": f"{true[i]:.1%}", "estimated": f"{r.borda[i]:.1%}",
             "votes involving this model": int(r.counts[i].sum())} for i in np.argsort(-true)]
    best = int(np.argmax(true))
    ok = true[r.best] >= true.max() - float(epsilon)
    status = "certified" if r.certified else "stopped at the vote limit, NOT certified"
    head = (f"**{method}** picked **{r.best_name}** after **{r.comparisons:,} votes** ({status}). "
            f"The true best by average win rate is **{task.names[best]}**"
            + (" - a match." if r.best == best else (" - within the tolerance." if ok else " - a miss."))
            + " Votes are drawn from real Chatbot Arena human preference rates.")
    fig, ax = _ax((7.0, 0.45 * len(rows) + 1.4))
    order = np.argsort(true)
    ax.barh(np.arange(len(order)) + 0.18, 100 * true[order], height=0.34, color=BASE, label="true")
    ax.barh(np.arange(len(order)) - 0.18, 100 * r.borda[order], height=0.34, color=SERIES[0], label="estimated")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([task.names[i] for i in order], fontsize=8.5, color=INK)
    ax.axvline(50, color=INK2, linestyle=(0, (3, 3)), linewidth=1)
    ax.set_xlabel("average win rate against the other models (%)", color=INK2)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    _title(ax, "Win rates: true vs estimated from the votes used")
    fig.tight_layout()
    return head, fig, pd.DataFrame(rows)


# ---------------------------------------------------------------- layout
INTRO = f"""
# frugal-eval: cheaper LLM evaluation, live

Methods from the research literature for spending fewer LLM calls on evaluation and routing,
running live on **real data**: per-question results of 395 Open LLM Leaderboard models, RouterBench
(11 LLMs with real prices) and 56k Chatbot Arena human votes. No API key needed.
[GitHub]({REPO}) · `pip install git+{REPO}`
"""

with gr.Blocks(title="frugal-eval demo") as demo:
    gr.Markdown(INTRO)
    with gr.Tab("Find the best model"):
        gr.Markdown("Which model is best? Adaptive methods stop asking about clearly weak models early, "
                    "and stop entirely once the winner is **certified** at the chosen confidence.")
        with gr.Row():
            b1 = gr.Dropdown(list(BENCHMARKS), value=list(BENCHMARKS)[0], label="Benchmark")
            k1 = gr.Slider(3, 30, value=10, step=1, label="Models to compare")
            s1 = gr.Number(value=0, precision=0, label="Random draw of models (seed)")
            c1 = gr.Radio(["0.9", "0.95", "0.99"], value="0.95", label="Confidence")
        m1 = gr.CheckboxGroup(ALL_IDENTIFY, value=FAST_IDENTIFY, label="Methods (top_two_ts and track_and_stop are slower)")
        with gr.Row():
            go1 = gr.Button("Run", variant="primary")
            new1 = gr.Button("New random models")
        h1 = gr.Markdown()
        with gr.Row():
            p1a = gr.Plot(label="Calls saved", show_label=False)
            p1b = gr.Plot(label="Questions per model", show_label=False)
        t1 = gr.Dataframe(label="Results", wrap=True)
        go1.click(run_identify, [b1, k1, s1, c1, m1], [h1, p1a, p1b, t1])
        new1.click(lambda: random.randint(1, 10_000), None, s1).then(run_identify, [b1, k1, s1, c1, m1],
                                                                   [h1, p1a, p1b, t1])

    with gr.Tab("Estimate scores"):
        gr.Markdown("Estimate every model's full-benchmark score from a few questions. `irt` and "
                    "`anchor_points` learn which questions are informative from 300 other models.")
        with gr.Row():
            b2 = gr.Dropdown(list(BENCHMARKS), value=list(BENCHMARKS)[0], label="Benchmark")
            k2 = gr.Slider(3, 30, value=10, step=1, label="Models")
            s2 = gr.Number(value=0, precision=0, label="Random draw of models (seed)")
            q2 = gr.Slider(10, 400, value=100, step=10, label="Questions per model")
        go2 = gr.Button("Run", variant="primary")
        h2 = gr.Markdown()
        p2 = gr.Plot(label="Estimated vs true", show_label=False)
        t2 = gr.Dataframe(label="Error by method", wrap=True)
        go2.click(run_estimate, [b2, k2, s2, q2], [h2, p2, t2])

    with gr.Tab("Route requests (cost-aware)"):
        gr.Markdown("Each prompt arrives once; the router picks one of 11 LLMs and learns only from the result, "
                    "scored as quality minus *cost weight* x cost. First run downloads RouterBench (about 30 s). "
                    "Here the router sees each prompt's task type; the repo's benchmark also uses prompt "
                    "embeddings, which give a better trade-off at moderate cost weights.")
        with gr.Row():
            m3 = gr.Dropdown(["linucb", "lin_ts", "thompson", "kl_ucb", "epsilon_greedy", "ucb1", "exp3", "uniform"],
                             value="linucb", label="Method")
            w3 = gr.Slider(0, 4, value=0.5, step=0.05, label="Cost weight")
            n3 = gr.Slider(2000, 36000, value=10000, step=1000, label="Number of requests")
            s3 = gr.Number(value=0, precision=0, label="Seed")
        go3 = gr.Button("Run", variant="primary")
        h3 = gr.Markdown()
        with gr.Row():
            p3a = gr.Plot(label="Quality vs cost", show_label=False)
            p3b = gr.Plot(label="Traffic share", show_label=False)
        go3.click(run_routing, [m3, w3, n3, s3], [h3, p3a, p3b])

    with gr.Tab("Pairwise votes (Chatbot Arena)"):
        gr.Markdown("Find the model with the highest average win rate from 'A or B?' votes, the format of "
                    "Chatbot Arena and LLM-as-judge evaluations. Every vote costs a judge call.")
        with gr.Row():
            pk4 = gr.Radio(["Random models", "Top models (close, hard)"], value="Random models", label="Models")
            k4 = gr.Slider(4, 12, value=8, step=1, label="How many")
            s4 = gr.Number(value=0, precision=0, label="Seed")
        with gr.Row():
            m4 = gr.Dropdown([m.name for m in fe.list_methods("duel")], value="borda_lucb", label="Method")
            e4 = gr.Slider(0, 0.05, value=0.02, step=0.005, label="Tolerance (win rate)")
            v4 = gr.Slider(10_000, 300_000, value=100_000, step=10_000, label="Vote limit")
        go4 = gr.Button("Run", variant="primary")
        h4 = gr.Markdown()
        p4 = gr.Plot(label="Win rates", show_label=False)
        t4 = gr.Dataframe(label="Ranking", wrap=True)
        go4.click(run_duel, [pk4, k4, s4, m4, e4, v4], [h4, p4, t4])

    gr.Markdown(f"Data: Open LLM Leaderboard per-question results via tinyBenchmarks (MIT); RouterBench "
                f"(Hu et al., 2024); Chatbot Arena human preferences (Apache-2.0). Methods and papers: "
                f"[method zoo]({REPO}#method-zoo).")
    demo.load(run_identify, [b1, k1, s1, c1, m1], [h1, p1a, p1b, t1])

if __name__ == "__main__":
    demo.launch()
