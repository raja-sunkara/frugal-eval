"""Real per-example results from the Open LLM Leaderboard (395 models).

The data comes from the tinyBenchmarks project (Polo et al., 2024, MIT license,
https://github.com/felipemaiapolo/tinyBenchmarks), stored compactly here.
Each task is a matrix of shape (395 models, n_examples): 1 = correct, 0 = wrong
(TruthfulQA holds the MC2 probability score in [0, 1]).
"""
from __future__ import annotations

import os

import numpy as np

DATA = os.path.join(os.path.dirname(__file__), "data", "open_llm_leaderboard.npz")
TASKS = ("gsm8k", "arc_challenge", "winogrande", "hellaswag", "truthfulqa", "mmlu")

_cache: dict = {}


def load(task: str) -> tuple[np.ndarray, list[str]]:
    if task not in TASKS:
        raise ValueError(f"task must be one of {TASKS}")
    if "file" not in _cache:
        with np.load(DATA) as npz:  # read everything now; an open zip handle breaks under fork
            _cache["file"] = {k: npz[k] for k in npz.files}
    f = _cache["file"]
    if task not in _cache:
        if f"{task}__bits" in f:
            n = int(f[f"{task}__n"])
            scores = np.unpackbits(f[f"{task}__bits"], axis=1)[:, :n].astype(float)
        else:
            scores = np.clip(f[f"{task}__f16"].astype(float), 0.0, 1.0)
        _cache[task] = scores
    return _cache[task], [str(m) for m in f["models"]]
