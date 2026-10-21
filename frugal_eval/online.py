"""Online selection: pick a candidate for each incoming example while learning.

This is the setting of serving live traffic: requests arrive one at a time, you
choose which prompt or model answers each one, and you only see the score of
the one you chose. Good methods quickly concentrate on the best candidate
(low regret). Contextual methods use per-example features to route each request
to the candidate that is best *for that kind of request*.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .registry import get, register


@dataclass
class OnlineResult:
    method: str
    names: list[str]
    choices: np.ndarray
    rewards: np.ndarray                      # what was optimised: the score, or score minus cost penalty
    examples: np.ndarray
    best_fixed_reward: float | None = None   # total reward of the best single candidate on this stream
    oracle_reward: float | None = None       # total reward of a perfect per-example router
    scores: np.ndarray | None = None         # quality of each answer
    costs: np.ndarray | None = None          # cost of each call, when the task has costs
    cost_weight: float = 0.0
    extra: dict = field(default_factory=dict, repr=False)

    @property
    def average_score(self) -> float:
        return float((self.scores if self.scores is not None else self.rewards).mean())

    @property
    def total_cost(self) -> float | None:
        return None if self.costs is None else float(self.costs.sum())

    @property
    def total_reward(self) -> float:
        return float(self.rewards.sum())

    @property
    def average_reward(self) -> float:
        return float(self.rewards.mean())

    @property
    def regret(self) -> float | None:
        """Reward lost compared with always using the best single candidate (can be negative for routers)."""
        return None if self.best_fixed_reward is None else self.best_fixed_reward - self.total_reward

    @property
    def counts(self) -> np.ndarray:
        return np.bincount(self.choices, minlength=len(self.names))

    def summary(self) -> str:
        lines = [f"Method: {self.method}. Requests: {len(self.choices)}. Average score: {self.average_score:.3f}"]
        if self.costs is not None:
            lines.append(f"Total cost: {self.total_cost:.4f} ({self.total_cost / len(self.choices):.6f} per request); "
                         f"cost weight {self.cost_weight:g}")
        if self.best_fixed_reward is not None:
            T = len(self.choices)
            lines.append(f"Best single candidate would average {self.best_fixed_reward / T:.3f} (objective); "
                         f"regret {self.regret:.1f} ({self.regret / T:+.3f} per request)")
        if self.oracle_reward is not None:
            lines.append(f"Perfect per-request routing would average {self.oracle_reward / len(self.choices):.3f}")
        share = self.counts / max(1, len(self.choices))
        for k in np.argsort(-share)[:5]:
            lines.append(f"  {self.names[k]}: chosen {share[k]:.0%}")
        return "\n".join(lines)


class Policy:
    """Override ``select`` and ``update``. ``x`` is the example's feature vector (or None)."""

    def __init__(self, K: int, rng: np.random.Generator, dim: int | None = None, **_: Any):
        self.K, self.rng, self.dim = K, rng, dim
        self.n = np.zeros(K)
        self.s = np.zeros(K)
        self.t = 0

    def select(self, x: np.ndarray | None) -> int:
        raise NotImplementedError

    def update(self, arm: int, reward: float, x: np.ndarray | None) -> None:
        self.n[arm] += 1
        self.s[arm] += reward
        self.t += 1

    def _untried(self) -> int | None:
        zeros = np.flatnonzero(self.n == 0)
        return int(zeros[0]) if zeros.size else None

    @property
    def means(self) -> np.ndarray:
        return np.divide(self.s, self.n, out=np.zeros(self.K), where=self.n > 0)


@register("online", "uniform", kind="baseline", paper="Baseline", summary="Pick a candidate uniformly at random.")
class Uniform(Policy):
    def select(self, x):
        return int(self.rng.integers(self.K))


@register("online", "epsilon_greedy", kind="non-contextual",
          paper="Auer, Cesa-Bianchi & Fischer (Machine Learning 2002), epsilon_n-greedy",
          url="https://doi.org/10.1023/A:1013689704352",
          summary="Mostly use the best so far; explore at random with a decaying probability.")
class EpsilonGreedy(Policy):
    def __init__(self, K, rng, c: float = 5.0, **kw):
        super().__init__(K, rng, **kw)
        self.c = c

    def select(self, x):
        u = self._untried()
        if u is not None:
            return u
        if self.rng.random() < min(1.0, self.c * self.K / max(self.t, 1)):
            return int(self.rng.integers(self.K))
        return int(np.argmax(self.means))


@register("online", "ucb1", kind="non-contextual", paper="Auer, Cesa-Bianchi & Fischer (Machine Learning 2002)",
          url="https://doi.org/10.1023/A:1013689704352", summary="Optimism: pick the highest upper confidence bound.")
class UCB1(Policy):
    def select(self, x):
        u = self._untried()
        if u is not None:
            return u
        return int(np.argmax(self.means + np.sqrt(2 * math.log(self.t + 1) / self.n)))


def _kl_bern(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-12, 1 - 1e-12)
    q = np.clip(q, 1e-12, 1 - 1e-12)
    return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))


@register("online", "kl_ucb", kind="non-contextual", paper="Garivier & Cappé (COLT 2011)",
          url="https://proceedings.mlr.press/v19/garivier11a/garivier11a.pdf",
          summary="Upper confidence bounds from the KL divergence; tighter than UCB1 for scores near 0 or 1.")
class KLUCB(Policy):
    def select(self, x):
        u = self._untried()
        if u is not None:
            return u
        target = math.log(self.t + 1) / self.n
        lo, hi = self.means.copy(), np.ones(self.K)
        for _ in range(25):  # bisection for the largest q with kl(mean, q) <= target
            mid = (lo + hi) / 2
            ok = _kl_bern(self.means, mid) <= target
            lo = np.where(ok, mid, lo)
            hi = np.where(ok, hi, mid)
        return int(np.argmax(lo))


@register("online", "thompson", kind="non-contextual", paper="Agrawal & Goyal (COLT 2012); Thompson (1933)",
          url="https://proceedings.mlr.press/v23/agrawal12/agrawal12.pdf",
          summary="Sample each candidate's score from its posterior and pick the highest sample.")
class Thompson(Policy):
    def __init__(self, K, rng, **kw):
        super().__init__(K, rng, **kw)
        self.a = np.ones(K)
        self.b = np.ones(K)

    def select(self, x):
        return int(np.argmax(self.rng.beta(self.a, self.b)))

    def update(self, arm, reward, x):
        super().update(arm, reward, x)
        r = float(self.rng.random() < reward)  # Bernoulli trick for scores in [0, 1]
        self.a[arm] += r
        self.b[arm] += 1 - r


@register("online", "exp3", kind="adversarial", paper="Auer, Cesa-Bianchi, Freund & Schapire (SIAM J. Computing 2002)",
          url="https://doi.org/10.1137/S0097539701398375",
          summary="Exponential weights with importance-weighted rewards; robust when quality drifts over time.")
class EXP3(Policy):
    def __init__(self, K, rng, horizon: int = 10000, **kw):
        super().__init__(K, rng, **kw)
        self.gamma = min(1.0, math.sqrt(K * math.log(K) / ((math.e - 1) * max(horizon, 1))))
        self.logw = np.zeros(K)
        self.p = np.full(K, 1.0 / K)

    def select(self, x):
        w = np.exp(self.logw - self.logw.max())
        self.p = (1 - self.gamma) * w / w.sum() + self.gamma / self.K
        return int(self.rng.choice(self.K, p=self.p))

    def update(self, arm, reward, x):
        super().update(arm, reward, x)
        self.logw[arm] += self.gamma * (reward / self.p[arm]) / self.K


@register("online", "linucb", kind="contextual", needs_features=True,
          paper="Li, Chu, Langford & Schapire (WWW 2010)", url="https://arxiv.org/abs/1003.0146",
          summary="Per-candidate ridge regression on example features with an upper confidence bonus: routes each request.")
class LinUCB(Policy):
    def __init__(self, K, rng, dim=None, alpha: float = 0.5, ridge: float = 1.0, **kw):
        super().__init__(K, rng, dim=dim, **kw)
        if dim is None:
            raise ValueError("linucb needs example features (task.features)")
        self.alpha = alpha
        self.Ainv = np.stack([np.eye(dim) / ridge for _ in range(K)])
        self.bvec = np.zeros((K, dim))

    def select(self, x):
        theta = np.einsum("kij,kj->ki", self.Ainv, self.bvec)
        mean = theta @ x
        width = np.sqrt(np.einsum("i,kij,j->k", x, self.Ainv, x))
        return int(np.argmax(mean + self.alpha * width))

    def update(self, arm, reward, x):
        super().update(arm, reward, x)
        A = self.Ainv[arm]
        Ax = A @ x
        self.Ainv[arm] = A - np.outer(Ax, Ax) / (1 + x @ Ax)  # Sherman-Morrison
        self.bvec[arm] += reward * x


@register("online", "lin_ts", kind="contextual", needs_features=True,
          paper="Agrawal & Goyal (ICML 2013), Thompson sampling for contextual bandits",
          url="https://proceedings.mlr.press/v28/agrawal13.pdf",
          summary="Linear Thompson sampling: sample each candidate's regression weights, route to the best sampled score.")
class LinTS(LinUCB):
    def __init__(self, K, rng, dim=None, v: float = 0.25, ridge: float = 1.0, **kw):
        super().__init__(K, rng, dim=dim, ridge=ridge, **kw)
        self.v = v

    def select(self, x):
        scores = np.empty(self.K)
        for k in range(self.K):
            mu = self.Ainv[k] @ self.bvec[k]
            cov = self.v ** 2 * (self.Ainv[k] + self.Ainv[k].T) / 2
            scores[k] = self.rng.multivariate_normal(mu, cov, method="cholesky") @ x
        return int(np.argmax(scores))


def run_online(task: Any, method: str = "thompson", *, horizon: int | None = None, seed: int | None = 0,
               cost_weight: float = 0.0, cost_scale: float | None = None, **options: Any) -> OnlineResult:
    """Simulate (or run live) online selection over a stream of the task's examples.

    With ``cost_weight`` > 0 (and ``task.costs``), each call is rewarded by
    ``score - cost_weight * cost / cost_scale``, rescaled to [0, 1], so methods
    trade quality against spend. ``cost_scale`` defaults to the 99th percentile cost.
    """
    if not hasattr(task, "source"):
        from .tasks import from_scores
        task = from_scores(task)
    info = get("online", method)
    rng = np.random.default_rng(seed)
    N, K = task.n_examples, task.n_candidates
    T = N if horizon is None else min(int(horizon), N)
    stream = rng.permutation(N)[:T]
    feats = None if task.features is None else np.asarray(task.features, dtype=float)
    if info.needs_features and feats is None:
        raise ValueError(f"{method} needs task.features (one vector per example)")
    costs = None if task.costs is None else np.asarray(task.costs, dtype=float)
    if cost_weight > 0 and costs is None:
        raise ValueError("cost_weight > 0 needs task.costs (cost of each candidate on each example)")
    lam = float(cost_weight)
    scale = float(cost_scale) if cost_scale else (float(np.quantile(costs, 0.99)) if costs is not None else 1.0)
    scale = max(scale, 1e-12)

    def utility(score: np.ndarray | float, cost: np.ndarray | float) -> np.ndarray | float:
        if lam == 0:
            return score
        return (score - lam * np.minimum(np.asarray(cost) / scale, 1.0) + lam) / (1 + lam)

    policy = info.fn(K, rng, dim=None if feats is None else feats.shape[1], horizon=T, **options)
    choices, scores_seen, costs_seen = np.empty(T, dtype=int), np.empty(T), np.zeros(T)
    rewards = np.empty(T)
    for t, i in enumerate(stream):
        x = None if feats is None else feats[i]
        arm = policy.select(x)
        r = task.source.score_many([(arm, int(i))])[0]
        c = costs[arm, i] if costs is not None else 0.0
        u = float(utility(r, c))
        policy.update(arm, u, x)
        choices[t], scores_seen[t], costs_seen[t], rewards[t] = arm, r, c, u
    best_fixed = oracle = None
    full = getattr(task.source, "scores", None)
    if full is not None:  # replay: the full matrix is known, so regret can be computed
        sub = full[:, stream]
        if costs is not None and lam > 0:
            sub = utility(sub, costs[:, stream])
        best_fixed = float(sub.sum(1).max())
        oracle = float(sub.max(0).sum())
    return OnlineResult(method=method, names=task.names, choices=choices, rewards=rewards, examples=stream,
                        best_fixed_reward=best_fixed, oracle_reward=oracle, scores=scores_seen,
                        costs=None if costs is None else costs_seen, cost_weight=lam)
