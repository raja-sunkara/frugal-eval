"""Draw the README benchmark figure from benchmarks/summary.json (written by run_all.py).

    python benchmarks/plot.py
"""
from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", grid="#e4e3df", accent="#2a78d6", base="#a9a8a2"),
    "dark": dict(surface="#1a1a19", ink="#f0efec", ink2="#c3c2b7", grid="#34332f", accent="#3987e5", base="#6d6c66"),
}

PANELS = [
    ("identify", "Find the best", "Calls saved (%), higher is better", lambda v: f"{v:.0f}%", False),
    ("estimate", "Estimate scores from 100 questions", "Error (points), lower is better", lambda v: f"{v:.1f}", True),
    ("online", "Online selection and routing", "% of best single model, higher is better", lambda v: f"{v:.0f}%", False),
]


def draw(summary: dict, theme: str, path: str) -> None:
    c = THEMES[theme]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    panels = [p for p in PANELS if p[0] in summary]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.4 * len(panels), 4.1), facecolor=c["surface"], squeeze=False)
    for ax, (key, title, xlabel, fmt, lower_better) in zip(axes[0], panels):
        items = [(n, d["value"]) for n, d in summary[key].items() if d["kind"] != "fixed budget"]
        items.sort(key=lambda t: t[1], reverse=lower_better)  # best ends up on top
        names = [n for n, _ in items]
        vals = [v for _, v in items]
        best = min(vals) if lower_better else max(vals)
        colors = [c["base"] if summary[key][n]["kind"] == "baseline" else c["accent"] for n in names]
        ax.set_facecolor(c["surface"])
        bars = ax.barh(range(len(names)), vals, height=0.62, color=colors, edgecolor=c["surface"], linewidth=2)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, color=c["ink"], fontfamily="DejaVu Sans Mono", fontsize=9)
        span = max(vals)
        inside = key == "online"  # keep labels clear of the 100% reference line
        for bar, v in zip(bars, vals):
            x = bar.get_width() - span * 0.02 if inside else bar.get_width() + span * 0.02
            color = "#ffffff" if inside else (c["ink"] if v == best else c["ink2"])
            ax.text(x, bar.get_y() + bar.get_height() / 2, fmt(v), va="center", ha="right" if inside else "left",
                    color=color, fontweight="bold" if v == best else "normal", fontsize=9)
        ax.set_xlim(0, span * 1.22)
        if key == "online":
            ax.axvline(100, color=c["ink2"], linewidth=1, linestyle=(0, (3, 3)))
            ax.text(99, len(names) - 0.4, "best single model ", color=c["ink2"], fontsize=8, va="bottom", ha="right")
        ax.set_title(title, color=c["ink"], loc="left", fontsize=11.5, fontweight="bold", pad=10)
        ax.set_xlabel(xlabel, color=c["ink2"], fontsize=9)
        ax.tick_params(axis="x", colors=c["ink2"], labelsize=8.5, length=0)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", color=c["grid"], linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(c["grid"])
    fig.text(0.01, 0.01, "Real per-question results of 395 Open LLM Leaderboard models · 10 models per trial · "
             "20 trials × 4 benchmarks · gray = baseline", color=c["ink2"], fontsize=8.5)
    fig.tight_layout(rect=(0, 0.04, 1, 1), w_pad=3)
    fig.savefig(path, dpi=180, facecolor=c["surface"])
    plt.close(fig)


def main() -> None:
    with open(os.path.join(ROOT, "benchmarks", "summary.json")) as f:
        summary = json.load(f)
    os.makedirs(os.path.join(ROOT, "docs"), exist_ok=True)
    for theme in THEMES:
        suffix = "" if theme == "light" else "-dark"
        draw(summary, theme, os.path.join(ROOT, "docs", f"benchmark{suffix}.png"))
    print("wrote docs/benchmark.png and docs/benchmark-dark.png")


if __name__ == "__main__":
    main()
