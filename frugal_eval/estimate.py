"""Estimate every candidate's full-test-set score from a small subset of examples.

Each method picks ``budget`` examples (the same ones for every candidate), runs
the candidates on them, and turns those results into a score estimate. Methods
that use ``task.prior`` (results of other models on the same examples) learn
which examples are representative or informative.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from .registry import get, register


@dataclass
class EstimateResult:
    method: str
    names: list[str]
    estimates: np.ndarray
    examples: np.ndarray          # indices of the examples that were run
    calls: int
    n_examples: int
    truth: np.ndarray | None = None
    extra: dict = field(default_factory=dict, repr=False)

    @property
    def errors(self) -> np.ndarray | None:
        return None if self.truth is None else self.estimates - self.truth

    @property
    def mae(self) -> float | None:
        return None if self.truth is None else float(np.mean(np.abs(self.errors)))

    @property
    def full_calls(self) -> int:
        return len(self.names) * self.n_examples

    @property
    def savings(self) -> float:
        return 1.0 - self.calls / self.full_calls

    def summary(self) -> str:
        width = max(10, max(len(n) for n in self.names))
        head = f"{'candidate':<{width}}  {'estimate':>8}" + ("  {:>7}  {:>7}".format("true", "error") if self.truth is not None else "")
        lines = [head]
        for k in np.argsort(-self.estimates):
            row = f"{self.names[k]:<{width}}  {self.estimates[k]:8.3f}"
            if self.truth is not None:
                row += f"  {self.truth[k]:7.3f}  {self.errors[k]:+7.3f}"
            lines.append(row)
        lines += ["", f"Method: {self.method}. Examples run: {len(self.examples)} of {self.n_examples} per candidate "
                      f"({self.savings:.0%} of calls saved)."]
        if self.truth is not None:
            lines.append(f"Mean absolute error: {self.mae:.4f}")
        return "\n".join(lines)


class _Query:
    """Runs all candidates on chosen examples and counts the calls."""

    def __init__(self, task: Any):
        self.source = task.source
        self.calls = 0

    def __call__(self, idx: np.ndarray) -> np.ndarray:
        K = self.source.n_candidates
        pairs = [(k, int(i)) for k in range(K) for i in idx]
        vals = np.array(self.source.score_many(pairs), dtype=float).reshape(K, len(idx))
        self.calls += len(pairs)
        return vals


def _need_prior(task: Any, name: str) -> np.ndarray:
    if task.prior is None:
        raise ValueError(f"{name} needs task.prior: scores of other models on the same examples")
    return np.asarray(task.prior, dtype=float)


# ----------------------------------------------------------------- methods
@register("estimate", "random", kind="baseline", paper="Baseline",
          summary="Run a uniformly random subset and report the plain average.")
def est_random(task: Any, budget: int, rng: np.random.Generator, query: _Query, **_: Any):
    idx = np.sort(rng.choice(task.n_examples, budget, replace=False))
    return query(idx).mean(1), idx, {}


@register("estimate", "stratified", kind="classical", needs_prior=True,
          paper="Stratified sampling (Neyman, 1934), as used in efficient-benchmarking work",
          url="https://doi.org/10.2307/2342192",
          summary="Group examples by difficulty (from prior models), sample each group in proportion.")
def est_stratified(task: Any, budget: int, rng: np.random.Generator, query: _Query, n_strata: int = 10, **_: Any):
    prior = _need_prior(task, "stratified")
    difficulty = prior.mean(0) + rng.random(task.n_examples) * 1e-9
    strata = np.array_split(np.argsort(difficulty), min(n_strata, budget))
    idx, weights = [], []
    alloc = np.maximum(1, np.round(budget * np.array([len(s) for s in strata]) / task.n_examples)).astype(int)
    while alloc.sum() > budget:
        alloc[np.argmax(alloc)] -= 1
    for s, a in zip(strata, alloc):
        pick = rng.choice(s, min(a, len(s)), replace=False)
        idx.extend(pick)
        weights.extend([len(s) / (task.n_examples * len(pick))] * len(pick))
    idx, weights = np.array(idx), np.array(weights)
    order = np.argsort(idx)
    idx, weights = idx[order], weights[order]
    return query(idx) @ weights, idx, {}


def _kmedoids(points: np.ndarray, k: int, rng: np.random.Generator, iters: int = 15) -> tuple[np.ndarray, np.ndarray]:
    """k-means++ then Lloyd iterations; returns medoid indices (real examples) and cluster sizes."""
    n = len(points)
    k = min(k, n)
    centers = [points[rng.integers(n)]]
    d2 = ((points - centers[0]) ** 2).sum(1)
    for _ in range(1, k):
        probs = d2 / d2.sum() if d2.sum() > 0 else np.full(n, 1 / n)
        centers.append(points[rng.choice(n, p=probs)])
        d2 = np.minimum(d2, ((points - centers[-1]) ** 2).sum(1))
    C = np.array(centers)
    sq = (points ** 2).sum(1)
    for _ in range(iters):
        dist = sq[:, None] - 2 * points @ C.T + (C ** 2).sum(1)[None, :]
        lab = dist.argmin(1)
        for c in range(k):
            members = points[lab == c]
            if len(members):
                C[c] = members.mean(0)
    dist = sq[:, None] - 2 * points @ C.T + (C ** 2).sum(1)[None, :]
    lab = dist.argmin(1)
    medoids, sizes = [], []
    for c in range(k):
        members = np.flatnonzero(lab == c)
        if len(members) == 0:
            continue
        medoids.append(int(members[np.argmin(dist[members, c])]))
        sizes.append(len(members))
    return np.array(medoids), np.array(sizes, dtype=float)


@register("estimate", "anchor_points", kind="representative subset", needs_prior=True,
          paper="Vivek, Ethayarajh, Yang & Kiela (EACL 2024), Anchor Points",
          url="https://arxiv.org/abs/2309.08638",
          summary="Cluster examples by how prior models answered them; run one representative per cluster, weight by cluster size.")
def est_anchor_points(task: Any, budget: int, rng: np.random.Generator, query: _Query, **_: Any):
    prior = _need_prior(task, "anchor_points")
    medoids, sizes = _kmedoids(prior.T, budget, rng)
    order = np.argsort(medoids)
    medoids, sizes = medoids[order], sizes[order]
    return query(medoids) @ (sizes / sizes.sum()), medoids, {}


# ------------------------------------------------------------------ IRT
def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def fit_irt(Y: np.ndarray, dim: int = 5, iters: int = 400, lr: float = 0.05, l2: float = 1e-3,
            seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit a multidimensional 2PL IRT model P(correct) = sigmoid(theta_m . alpha_i - beta_i) with Adam.

    Y has shape (models, examples) with values in [0, 1] (soft labels allowed).
    Returns (theta, alpha, beta).
    """
    rng = np.random.default_rng(seed)
    M, N = Y.shape
    params = [rng.normal(0, 0.1, (M, dim)), rng.normal(0, 0.1, (N, dim)) + 0.5, np.zeros(N)]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    b1, b2 = 0.9, 0.999
    for t in range(1, iters + 1):
        theta, alpha, beta = params
        P = _sigmoid(theta @ alpha.T - beta[None, :])
        G = (P - Y) / Y.size  # d(mean BCE)/d(logit)
        reg = l2 / Y.size  # weight decay on the same scale as the averaged loss
        grads = [G @ alpha + reg * theta, G.T @ theta + reg * alpha, -G.sum(0) + reg * beta]
        for i, g in enumerate(grads):
            m[i] = b1 * m[i] + (1 - b1) * g
            v[i] = b2 * v[i] + (1 - b2) * g * g
            params[i] = params[i] - lr * (m[i] / (1 - b1 ** t)) / (np.sqrt(v[i] / (1 - b2 ** t)) + 1e-8)
    return params[0], params[1], params[2]


def fit_ability(y: np.ndarray, alpha: np.ndarray, beta: np.ndarray, prior_sd: float = 1.0,
                iters: int = 50) -> np.ndarray:
    """MAP estimate of one model's ability vector from its results on a few examples (Newton steps)."""
    d = alpha.shape[1]
    theta = np.zeros(d)
    for _ in range(iters):
        p = _sigmoid(alpha @ theta - beta)
        grad = alpha.T @ (y - p) - theta / prior_sd ** 2
        H = -(alpha.T * (p * (1 - p))) @ alpha - np.eye(d) / prior_sd ** 2
        step = np.linalg.solve(H, grad)
        theta = theta - step
        if np.abs(step).max() < 1e-6:
            break
    return theta


@register("estimate", "irt", kind="model-based", needs_prior=True,
          paper="Polo, Weber, Choshen, Sun, Xu & Yurochkin (ICML 2024), tinyBenchmarks (p-IRT)",
          url="https://arxiv.org/abs/2402.14992",
          summary="Fit an item response model on prior models; infer each candidate's ability from anchor examples and predict the rest.")
def est_irt(task: Any, budget: int, rng: np.random.Generator, query: _Query, dim: int = 5, **_: Any):
    prior = _need_prior(task, "irt")
    theta_p, alpha, beta = fit_irt(prior, dim=dim, seed=int(rng.integers(1 << 31)))
    # anchors: cluster examples in the IRT item space (discrimination and difficulty)
    emb = np.hstack([alpha, beta[:, None]])
    anchors, _ = _kmedoids(emb, budget, rng)
    anchors = np.sort(anchors)
    obs = query(anchors)
    rest = np.setdiff1d(np.arange(task.n_examples), anchors)
    n, N = len(anchors), task.n_examples
    est = np.empty(obs.shape[0])
    for k in range(obs.shape[0]):
        th = fit_ability(obs[k], alpha[anchors], beta[anchors])
        pred_rest = _sigmoid(alpha[rest] @ th - beta[rest]).mean() if len(rest) else 0.0
        est[k] = (n / N) * obs[k].mean() + (1 - n / N) * pred_rest
    return est, anchors, {"dim": dim}


# --------------------------------------------------------------- entry point
def estimate(task: Any, method: str = "random", *, budget: int = 100, seed: int | None = 0,
             **options: Any) -> EstimateResult:
    """Estimate each candidate's full score after running only ``budget`` examples per candidate."""
    if not hasattr(task, "source"):
        from .tasks import from_scores
        task = from_scores(task)
    info = get("estimate", method)
    budget = int(min(max(1, budget), task.n_examples))
    rng = np.random.default_rng(seed)
    query = _Query(task)
    est, idx, extra = info.fn(task, budget, rng, query, **options)
    return EstimateResult(method=method, names=task.names, estimates=np.clip(np.asarray(est, float), 0, 1),
                          examples=np.asarray(idx), calls=query.calls, n_examples=task.n_examples,
                          truth=None if task.truth is None else np.asarray(task.truth), extra=extra)
