"""Tasks: what is being evaluated.

A Task wraps a score source (candidates x examples) plus optional side
information that some methods use:

* ``prior``    - scores of *other* models on the same examples, shape (M, N).
                 Methods such as IRT or anchor points learn which examples are
                 informative from these.
* ``features`` - a feature vector per example, shape (N, d), for contextual
                 (routing) methods.
* ``truth``    - the full-test-set score of each candidate, when known, so a
                 benchmark can grade the answer.

Built-in task families are registered with ``register_task`` and listed with
``list_tasks``; add your own the same way.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

from .sources import FunctionSource, MatrixSource, Source

DATA = os.path.join(os.path.dirname(__file__), "data", "open_llm_leaderboard.npz")
LEADERBOARD_TASKS = ("gsm8k", "arc_challenge", "winogrande", "hellaswag", "truthfulqa", "mmlu")


@dataclass
class Task:
    name: str
    source: Source
    prior: np.ndarray | None = None
    features: np.ndarray | None = None
    truth: np.ndarray | None = None
    info: dict = field(default_factory=dict)

    @property
    def names(self) -> list[str]:
        return list(self.source.names)

    @property
    def n_candidates(self) -> int:
        return self.source.n_candidates

    @property
    def n_examples(self) -> int:
        return self.source.n_examples


def from_scores(scores: Any, names: Sequence[str] | None = None, *, prior: Any = None,
                features: Any = None, name: str = "custom") -> Task:
    """A task from scores you already have (n_candidates x n_examples), e.g. for replay."""
    source = MatrixSource(scores, names)
    return Task(name, source, _opt(prior), _opt(features), truth=source.scores.mean(1))


def from_function(candidates: Sequence[Any], examples: Sequence[Any], score_fn: Callable[[Any, Any], float], *,
                  names: Sequence[str] | None = None, prior: Any = None, features: Any = None,
                  max_workers: int = 8, cache_path: str | None = None, name: str = "live") -> Task:
    """A live task: ``score_fn(candidate, example)`` is called only when a method asks for it."""
    source = FunctionSource(candidates, examples, score_fn, names=names, max_workers=max_workers,
                            cache_path=cache_path)
    return Task(name, source, _opt(prior), _opt(features))


def _opt(x: Any) -> np.ndarray | None:
    return None if x is None else np.asarray(x, dtype=float)


# ---------------------------------------------------------------- registry
_TASKS: dict[str, Callable[..., Task]] = {}


def register_task(name: str):
    def decorator(fn: Callable[..., Task]) -> Callable[..., Task]:
        _TASKS[name] = fn
        return fn
    return decorator


def list_tasks() -> list[str]:
    return sorted(_TASKS)


def make(name: str, **kwargs) -> Task:
    """Build a registered task by name, e.g. ``make("leaderboard/gsm8k", n_candidates=10, seed=0)``."""
    if name not in _TASKS:
        raise ValueError(f"unknown task {name!r}; available: {', '.join(list_tasks())}")
    return _TASKS[name](**kwargs)


# ---------------------------------------------------------- leaderboard data
_cache: dict[str, Any] = {}


def leaderboard_matrix(task: str) -> tuple[np.ndarray, list[str]]:
    """Per-example results of 395 Open LLM Leaderboard models, shape (395, n_examples).

    Data: tinyBenchmarks (Polo et al., 2024), MIT license; see data/NOTICE.
    """
    if task not in LEADERBOARD_TASKS:
        raise ValueError(f"task must be one of {LEADERBOARD_TASKS}")
    if "file" not in _cache:
        with np.load(DATA) as npz:  # read fully: an open zip handle breaks under multiprocessing
            _cache["file"] = {k: npz[k] for k in npz.files}
    f = _cache["file"]
    if task not in _cache:
        if f"{task}__bits" in f:
            n = int(f[f"{task}__n"])
            _cache[task] = np.unpackbits(f[f"{task}__bits"], axis=1)[:, :n].astype(float)
        else:
            _cache[task] = np.clip(f[f"{task}__f16"].astype(float), 0.0, 1.0)
    return _cache[task], [str(m) for m in f["models"]]


def example_features(prior: np.ndarray, dim: int = 8) -> np.ndarray:
    """Embed each example by how the prior models did on it (SVD), plus a bias term."""
    centered = prior - prior.mean(0, keepdims=True)
    _, s, vt = np.linalg.svd(centered, full_matrices=False)
    d = min(dim, len(s))
    emb = (vt[:d].T * s[:d]) / np.sqrt(max(prior.shape[0], 1))
    emb /= np.maximum(np.linalg.norm(emb, axis=1, keepdims=True).max(), 1e-12)
    return np.hstack([np.ones((prior.shape[1], 1)), emb])


def leaderboard(task: str = "gsm8k", n_candidates: int = 10, seed: int = 0, pick: str = "random",
                n_prior: int = 300, n_examples: int | None = None, feature_dim: int = 8) -> Task:
    """Candidates are real leaderboard models; ``prior`` holds a disjoint set of other models.

    pick="random": random candidates. pick="top": the strongest models (scores nearly tied, the hard case).
    """
    scores, names = leaderboard_matrix(task)
    rng = np.random.default_rng(seed)
    M = scores.shape[0]
    if pick == "top":
        order = np.argsort(-scores.mean(1))
        cand = order[:n_candidates]
        rest = rng.permutation(order[n_candidates:])
    elif pick == "random":
        perm = rng.permutation(M)
        cand, rest = perm[:n_candidates], perm[n_candidates:]
    else:
        raise ValueError("pick must be 'random' or 'top'")
    prior_idx = rest[:n_prior]
    cols = np.arange(scores.shape[1])
    if n_examples is not None and n_examples < len(cols):
        cols = np.sort(rng.choice(cols, n_examples, replace=False))
    mat = scores[np.ix_(cand, cols)]
    prior = scores[np.ix_(prior_idx, cols)]
    short = [names[i].split("/")[-1][:40] for i in cand]
    return Task(f"leaderboard/{task}", MatrixSource(mat, short), prior=prior,
                features=example_features(prior, feature_dim), truth=mat.mean(1),
                info={"benchmark": task, "pick": pick, "seed": seed})


for _t in LEADERBOARD_TASKS:
    register_task(f"leaderboard/{_t}")(lambda _t=_t, **kw: leaderboard(_t, **kw))


@register_task("synthetic")
def synthetic(n_candidates: int = 10, n_examples: int = 1000, seed: int = 0, n_prior: int = 50,
              dim: int = 2) -> Task:
    """Models with a latent ability answering items with latent difficulty (a 2PL IRT model)."""
    rng = np.random.default_rng(seed)
    theta = rng.normal(0, 1, (n_candidates + n_prior, dim))
    a = np.abs(rng.normal(1, 0.3, (n_examples, dim)))
    b = rng.normal(0, 1, n_examples)
    p = 1 / (1 + np.exp(-(theta @ a.T - b)))
    x = (rng.random(p.shape) < p).astype(float)
    mat, prior = x[:n_candidates], x[n_candidates:]
    return Task("synthetic", MatrixSource(mat), prior=prior, features=example_features(prior),
                truth=mat.mean(1), info={"seed": seed})
