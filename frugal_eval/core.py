"""Adaptive search for the best candidate (best-arm identification on a finite test set)."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

from .bounds import radius
from .sources import FunctionSource, MatrixSource, Source

ALGORITHMS = ("lucb", "uniform", "successive_halving")


@dataclass
class Result:
    """Outcome of an adaptive evaluation."""

    best: int
    names: list[str]
    means: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    pulls: np.ndarray
    n_examples: int
    stopped_by: str  # "confidence" | "budget" | "exhausted"
    algorithm: str
    delta: float
    epsilon: float
    history: list[tuple[int, int]] = field(default_factory=list, repr=False)

    @property
    def best_name(self) -> str:
        return self.names[self.best]

    @property
    def total_calls(self) -> int:
        return int(self.pulls.sum())

    @property
    def full_calls(self) -> int:
        return len(self.names) * self.n_examples

    @property
    def savings(self) -> float:
        """Fraction of calls saved compared with running every candidate on every example."""
        return 1.0 - self.total_calls / self.full_calls

    @property
    def certified(self) -> bool:
        """True when the winner is guaranteed (with prob >= 1 - delta) to be best within epsilon."""
        return self.stopped_by in {"confidence", "exhausted"}

    def ranking(self) -> list[int]:
        return [int(k) for k in np.argsort(-self.means, kind="stable")]

    def summary(self) -> str:
        width = max(10, max(len(n) for n in self.names))
        lines = [f"{'candidate':<{width}}  {'score':>7}  {'interval':>17}  {'examples':>9}"]
        for k in self.ranking():
            mark = "  <- best" if k == self.best else ""
            lines.append(
                f"{self.names[k]:<{width}}  {self.means[k]:7.3f}  "
                f"[{max(self.lower[k], 0):6.3f}, {min(self.upper[k], 1):6.3f}]  "
                f"{int(self.pulls[k]):>4}/{self.n_examples:<4}{mark}"
            )
        verdict = {
            "confidence": f"Certified: best within {self.epsilon:g} with probability >= {1 - self.delta:g}.",
            "exhausted": "Every candidate was run on every example; the ranking is exact.",
            "budget": "Stopped at the call budget; the winner is the best guess, NOT certified.",
        }[self.stopped_by]
        lines += ["", f"Best: {self.best_name}", verdict,
                  f"Calls: {self.total_calls:,} of {self.full_calls:,} ({self.savings:.0%} saved)"]
        return "\n".join(lines)


class Evaluation:
    """Running state: which examples each candidate has seen and the confidence intervals.

    All candidates walk through the test set in the same random order, so
    candidates are compared on the same examples (a paired design), which keeps
    comparisons fair and reduces noise.
    """

    def __init__(self, source: Source, delta: float = 0.05, epsilon: float = 0.0,
                 seed: int | None = 0, record_history: bool = False):
        if not 0 < delta < 1:
            raise ValueError("delta must be between 0 and 1")
        if epsilon < 0:
            raise ValueError("epsilon must be >= 0")
        self.source = source
        self.K, self.N = source.n_candidates, source.n_examples
        if self.K < 2:
            raise ValueError("need at least two candidates to compare")
        self.delta, self.epsilon = delta, epsilon
        self.order = np.random.default_rng(seed).permutation(self.N)
        self.n = np.zeros(self.K, dtype=int)
        self.s = np.zeros(self.K)
        self.s2 = np.zeros(self.K)
        # X[k, t] = score of candidate k on the t-th example of the shared order
        self.X = np.zeros((self.K, self.N))
        # running paired statistics on the examples both k and j have seen:
        # M[k, j] = count, D[k, j] = sum of (x_k - x_j), D2[k, j] = sum of (x_k - x_j)^2
        self.M = np.zeros((self.K, self.K))
        self.D = np.zeros((self.K, self.K))
        self.D2 = np.zeros((self.K, self.K))
        self.record_history = record_history
        self.history: list[tuple[int, int]] = []

    # -- sampling ----------------------------------------------------------
    def available(self) -> np.ndarray:
        return self.n < self.N

    def pull(self, arms: Sequence[int]) -> None:
        """Score each listed candidate on its next unseen example (one batch of calls)."""
        pairs, seen = [], set()
        for k in arms:
            k = int(k)
            if k in seen or self.n[k] >= self.N:
                continue
            seen.add(k)
            pairs.append((k, int(self.order[self.n[k]])))
        if not pairs:
            return
        for (k, _), x in zip(pairs, self.source.score_many(pairs)):
            t = self.n[k]
            self.X[k, t] = x
            ahead = self.n > t  # candidates that have already seen this example
            ahead[k] = False
            if ahead.any():
                d = x - self.X[ahead, t]
                self.M[k, ahead] += 1
                self.M[ahead, k] += 1
                self.D[k, ahead] += d
                self.D[ahead, k] -= d
                self.D2[k, ahead] += d * d
                self.D2[ahead, k] += d * d
            self.n[k] += 1
            self.s[k] += x
            self.s2[k] += x * x
        if self.record_history:
            self.history.append((int(self.n.sum()), int(np.argmax(self.means))))

    # -- statistics --------------------------------------------------------
    @property
    def means(self) -> np.ndarray:
        return np.divide(self.s, self.n, out=np.full(self.K, 0.5), where=self.n > 0)

    def radii(self) -> np.ndarray:
        return radius(self.n, self.s, self.s2, self.N, self.K, self.delta)

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        m, r = self.means, self.radii()
        return m - r, m + r

    def gaps(self, h: int) -> tuple[np.ndarray, np.ndarray]:
        """Estimated score gap (h minus j) and its confidence radius, for every j.

        Gaps are estimated from paired differences on the examples both
        candidates have seen. Because strong and weak models tend to get the same
        easy questions right, the paired difference is far less noisy than the
        two scores separately, which is where most of the savings come from.
        """
        n = self.M[h].copy()
        seen = n > 0
        seen[h] = False
        diff = np.zeros(self.K)
        rad = np.full(self.K, np.inf)
        rad[h] = 0.0
        if seen.any():
            ns, dsum, d2 = n[seen], self.D[h, seen], self.D2[h, seen]
            # rescale differences from [-1, 1] to [0, 1]: d' = (d + 1) / 2
            s1 = (dsum + ns) / 2.0
            s2 = (d2 + 2.0 * dsum + ns) / 4.0
            diff[seen] = dsum / ns
            rad[seen] = 2.0 * radius(ns, s1, s2, self.N, self.K * self.K, self.delta)
        return diff, rad

    def leader_and_challenger(self) -> tuple[int, int]:
        h = int(np.argmax(self.means))
        diff, rad = self.gaps(h)
        threat = rad - diff  # how much better than the leader j could still be
        threat[h] = -np.inf
        return h, int(np.argmax(threat))

    def done(self) -> str | None:
        if not self.available().any():
            return "exhausted"
        h = int(np.argmax(self.means))
        diff, rad = self.gaps(h)
        if np.all(np.delete(diff - rad, h) >= -self.epsilon):
            return "confidence"
        return None

    def result(self, stopped_by: str, algorithm: str) -> Result:
        lo, up = self.bounds()
        m = self.means
        best = int(np.argmax(m))
        return Result(best=best, names=list(self.source.names), means=m.copy(), lower=lo, upper=up,
                      pulls=self.n.copy(), n_examples=self.N, stopped_by=stopped_by, algorithm=algorithm,
                      delta=self.delta, epsilon=self.epsilon, history=list(self.history))


# ======================================================================
# sampling rules
# ======================================================================
def _over_budget(ev: Evaluation, budget: int | None) -> bool:
    return budget is not None and ev.n.sum() >= budget


def _warm_start(ev: Evaluation, per_arm: int) -> None:
    for _ in range(min(per_arm, ev.N)):
        ev.pull(range(ev.K))


def run_uniform(ev: Evaluation, budget: int | None = None, warmup: int = 10) -> Result:
    """Baseline: every candidate on the same examples in lockstep, with the same stopping rule."""
    _warm_start(ev, 1)
    while True:
        status = ev.done()
        if status:
            return ev.result(status, "uniform")
        if _over_budget(ev, budget):
            return ev.result("budget", "uniform")
        ev.pull(np.flatnonzero(ev.available()))


def run_lucb(ev: Evaluation, budget: int | None = None, warmup: int = 10) -> Result:
    """LUCB (Kalyanakrishnan et al., 2012): sample the leader and its strongest challenger."""
    _warm_start(ev, warmup)
    while True:
        status = ev.done()
        if status:
            return ev.result(status, "lucb")
        if _over_budget(ev, budget):
            return ev.result("budget", "lucb")
        h, c = ev.leader_and_challenger()
        # grow the set of examples both have seen: advance whichever is behind (or both)
        common = min(ev.n[h], ev.n[c])
        arms = [k for k in (h, c) if ev.n[k] == common and ev.n[k] < ev.N]
        if not arms:
            arms = [k for k in (h, c) if ev.n[k] < ev.N]
        if not arms:  # both exact: fall back to the widest open interval
            r = ev.radii()
            r[~ev.available()] = -1
            arms = [int(np.argmax(r))]
        ev.pull(arms)


def run_successive_halving(ev: Evaluation, budget: int | None = None, warmup: int = 10) -> Result:
    """Successive halving (Karnin et al., 2013): fixed budget, no confidence guarantee."""
    if budget is None:
        raise ValueError("successive_halving needs a budget (number of calls)")
    alive = list(range(ev.K))
    rounds = max(1, math.ceil(math.log2(ev.K)))
    spent = 0
    for r in range(rounds):
        per_arm = min(ev.N, ev.n[alive].max() + max(1, (budget - spent) // (len(alive) * (rounds - r))))
        while True:
            need = [k for k in alive if ev.n[k] < per_arm]
            if not need or spent >= budget:
                break
            ev.pull(need)
            spent = int(ev.n.sum())
        if len(alive) == 1:
            break
        alive = sorted(alive, key=lambda k: -ev.means[k])[: max(1, math.ceil(len(alive) / 2))]
    result = ev.result("budget", "successive_halving")
    result.best = alive[0]
    return result


RUNNERS: dict[str, Callable[..., Result]] = {
    "lucb": run_lucb,
    "uniform": run_uniform,
    "successive_halving": run_successive_halving,
}


def find_best(
    scores: Any = None,
    *,
    candidates: Sequence[Any] | None = None,
    examples: Sequence[Any] | None = None,
    score_fn: Callable[[Any, Any], float] | None = None,
    names: Sequence[str] | None = None,
    algorithm: str = "lucb",
    delta: float = 0.05,
    epsilon: float = 0.0,
    budget: int | None = None,
    seed: int | None = 0,
    warmup: int = 10,
    max_workers: int = 8,
    cache_path: str | None = None,
    record_history: bool = False,
) -> Result:
    """Find the best candidate while running as few evaluations as possible.

    Either pass ``scores`` (an array, n_candidates x n_examples, of scores you
    already have, for simulation), or ``candidates``, ``examples`` and
    ``score_fn(candidate, example) -> float in [0, 1]`` to evaluate live.

    With the default stopping rule, the returned candidate is, with probability
    at least ``1 - delta``, within ``epsilon`` of the candidate a full
    evaluation on all examples would pick.
    """
    if algorithm not in RUNNERS:
        raise ValueError(f"algorithm must be one of {', '.join(ALGORITHMS)}")
    if scores is not None:
        source: Source = scores if hasattr(scores, "score_many") else MatrixSource(scores, names)
    elif candidates is not None and examples is not None and score_fn is not None:
        source = FunctionSource(candidates, examples, score_fn, names=names,
                                max_workers=max_workers, cache_path=cache_path)
    else:
        raise ValueError("pass either `scores`, or `candidates`, `examples` and `score_fn`")
    ev = Evaluation(source, delta=delta, epsilon=epsilon, seed=seed, record_history=record_history)
    return RUNNERS[algorithm](ev, budget=budget, warmup=warmup)
