"""Find the best candidate from pairwise comparisons (dueling bandits).

Many LLM evaluations are pairwise: a human or an LLM judge sees two answers and
says which is better (Chatbot Arena, AlpacaEval, MT-Bench pairwise). Every
comparison costs a judge call. These methods choose which pairs to compare so the
winner is certified with as few comparisons as possible.

Target: the **Borda winner**, the candidate with the highest average win rate
against a uniformly random opponent (ties count as half a win). It always exists,
and it is what "average win rate" leaderboards report.

All fixed-confidence methods share one stopping rule: stop when the leader's
Borda score is confidently above every other candidate's (pairwise anytime
Hoeffding intervals with a union bound over pairs and time), so they are
compared fairly.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

from .registry import get, register


@dataclass
class DuelTask:
    """A pairwise task. ``prefs[i, j]`` = probability that i beats j (ties count half), ``prefs + prefs.T = 1``.

    ``duel(i, j, rng)`` returns 1.0 if i wins, 0.0 if j wins, 0.5 for a tie. Replay tasks
    sample from ``prefs``; live tasks call a judge.
    """
    name: str
    names: list[str]
    prefs: np.ndarray | None = None
    judge: Callable[[int, int], float] | None = None
    info: dict = field(default_factory=dict)

    @property
    def n_candidates(self) -> int:
        return len(self.names)

    @property
    def borda(self) -> np.ndarray | None:
        if self.prefs is None:
            return None
        K = len(self.names)
        return (self.prefs.sum(1) - np.diag(self.prefs)) / (K - 1)

    @property
    def condorcet(self) -> int | None:
        """Index of the candidate that beats every other with probability > 1/2, if one exists."""
        if self.prefs is None:
            return None
        for i in range(len(self.names)):
            if all(self.prefs[i, j] > 0.5 for j in range(len(self.names)) if j != i):
                return i
        return None

    def duel(self, i: int, j: int, rng: np.random.Generator) -> float:
        if self.judge is not None:
            return float(self.judge(i, j))
        p = self.prefs[i, j]
        # ties are folded into the win probability; replay draws a win or a loss
        return float(rng.random() < p)


@dataclass
class DuelResult:
    method: str
    names: list[str]
    best: int
    borda: np.ndarray          # estimated Borda scores
    comparisons: int
    stopped_by: str            # "confidence" | "budget"
    counts: np.ndarray = field(repr=False, default=None)

    @property
    def best_name(self) -> str:
        return self.names[self.best]

    @property
    def certified(self) -> bool:
        return self.stopped_by == "confidence"

    def summary(self) -> str:
        width = max(10, max(len(n) for n in self.names))
        lines = [f"{'candidate':<{width}}  {'win rate':>8}"]
        for k in np.argsort(-self.borda):
            lines.append(f"{self.names[k]:<{width}}  {self.borda[k]:8.3f}" + ("  <- best" if k == self.best else ""))
        verdict = "Certified." if self.certified else "Stopped at the budget; NOT certified."
        lines += ["", f"Best: {self.best_name}. {verdict} Comparisons: {self.comparisons:,}"]
        return "\n".join(lines)


class DuelState:
    """Pairwise counts and the shared Borda confidence intervals."""

    def __init__(self, task: DuelTask, delta: float, seed: int | None, epsilon: float = 0.0):
        self.task, self.delta, self.epsilon = task, delta, epsilon
        self.K = task.n_candidates
        if self.K < 2:
            raise ValueError("need at least two candidates")
        self.rng = np.random.default_rng(seed)
        self.n = np.zeros((self.K, self.K))   # comparisons of each pair (symmetric)
        self.w = np.zeros((self.K, self.K))   # wins of i over j (ties add 0.5 to both)
        # Borda-reduction counts: candidate i against a uniformly random opponent
        self.bn = np.zeros(self.K)
        self.bw = np.zeros(self.K)
        self.total = 0

    def compare(self, i: int, j: int) -> float:
        r = self.task.duel(i, j, self.rng)
        self.n[i, j] += 1
        self.n[j, i] += 1
        self.w[i, j] += r
        self.w[j, i] += 1 - r
        self.total += 1
        return r

    def compare_random(self, i: int) -> None:
        """Borda reduction (Jamieson et al., 2015): i against a uniformly random other candidate."""
        j = int(self.rng.integers(self.K - 1))
        j += j >= i
        r = self.compare(i, j)
        self.bn[i] += 1
        self.bw[i] += r

    # -- confidence ------------------------------------------------------
    def _log(self, n: np.ndarray, m: int) -> np.ndarray:
        n = np.maximum(n, 1)
        return np.log(math.pi ** 2 * m * n * n / (3 * self.delta))

    def pair_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        p = np.divide(self.w, self.n, out=np.full_like(self.w, 0.5), where=self.n > 0)
        rad = np.where(self.n > 0, np.sqrt(self._log(self.n, self.K * self.K) / (2 * np.maximum(self.n, 1))), 1.0)
        lo, up = np.clip(p - rad, 0, 1), np.clip(p + rad, 0, 1)
        np.fill_diagonal(lo, 0)
        np.fill_diagonal(up, 0)
        return lo, up

    def borda_bounds(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Borda estimate and interval for every candidate.

        From pairwise counts: b_i = mean over opponents j of the observed win rate p_ij. Each comparison
        of pair (i, j) moves b_i by at most 1 / ((K-1) n_ij), so Hoeffding's inequality for this weighted
        sum gives the radius sqrt(L * sum_j 1/n_ij / (2 (K-1)^2)), with L a union bound over candidates
        and the total number of comparisons. Unsampled pairs add their full range 1/(K-1).
        Reduction samples (i against a random opponent) give a second, independent interval; the two
        are intersected.
        """
        K = self.K
        off = ~np.eye(K, dtype=bool)
        p = np.divide(self.w, self.n, out=np.full_like(self.w, 0.5), where=self.n > 0)
        est = np.where(off, p, 0).sum(1) / (K - 1)
        L = math.log(math.pi ** 2 * 2 * K * max(self.total, 1) ** 2 / (3 * self.delta))
        seen = (self.n > 0) & off
        inv = np.where(seen, 1.0 / np.maximum(self.n, 1), 0.0).sum(1)
        missing = (~seen & off).sum(1)
        rad = np.sqrt(L * inv / (2 * (K - 1) ** 2)) + missing / (K - 1)
        blo, bup = est - rad, est + rad
        if self.bn.any():  # reduction samples are unbiased for the Borda score itself
            m = np.divide(self.bw, self.bn, out=np.full(K, 0.5), where=self.bn > 0)
            r2 = np.where(self.bn > 0, np.sqrt(L / (2 * np.maximum(self.bn, 1))), 1.0)
            use = (self.bn > 0) & (r2 < rad)
            est = np.where(use, m, est)
            blo, bup = np.maximum(blo, m - r2), np.minimum(bup, m + r2)
        return est, blo, bup

    def done(self) -> bool:
        est, lo, up = self.borda_bounds()
        h = int(np.argmax(est))
        return bool(lo[h] > np.delete(up, h).max() - self.epsilon)

    def result(self, method: str, stopped_by: str) -> DuelResult:
        est, _, _ = self.borda_bounds()
        return DuelResult(method, list(self.task.names), int(np.argmax(est)), est, self.total, stopped_by,
                          counts=self.n.copy())


def _stop(st: DuelState, budget: int | None, name: str) -> DuelResult | None:
    """Shared stopping rule. To stay fast it is checked after every 0.5% growth in comparisons
    (at most 0.5% overshoot), identically for every method."""
    if budget is not None and st.total >= budget:
        return st.result(name, "confidence" if st.done() else "budget")
    if st.total >= getattr(st, "_next_check", 0):
        st._next_check = st.total + max(1, int(0.005 * st.total))
        if st.done():
            return st.result(name, "confidence")
    return None


# ------------------------------------------------------------------- methods
@register("duel", "round_robin", kind="baseline", paper="Baseline",
          summary="Compare every pair in turn until the Borda winner is certified.")
def round_robin(st: DuelState, budget: int | None = None, **_: Any) -> DuelResult:
    pairs = [(i, j) for i in range(st.K) for j in range(i + 1, st.K)]
    while True:
        for i, j in pairs:
            out = _stop(st, budget, "round_robin")
            if out:
                return out
            st.compare(i, j)


@register("duel", "borda_lucb", kind="fixed confidence",
          paper="Jamieson, Katariya, Deshpande & Nowak (AISTATS 2015), Borda reduction; with LUCB",
          url="https://proceedings.mlr.press/v38/jamieson15.html",
          summary="Treat 'i vs a random opponent' as one arm whose mean is i's Borda score, then run LUCB.")
def borda_lucb(st: DuelState, budget: int | None = None, warmup: int = 5, **_: Any) -> DuelResult:
    for _ in range(warmup):
        for i in range(st.K):
            st.compare_random(i)
    while True:
        out = _stop(st, budget, "borda_lucb")
        if out:
            return out
        est, lo, up = st.borda_bounds()
        h = int(np.argmax(est))
        up_others = up.copy()
        up_others[h] = -np.inf
        c = int(np.argmax(up_others))
        st.compare_random(h)
        st.compare_random(c)


@register("duel", "borda_elimination", kind="fixed confidence",
          paper="Jamieson, Katariya, Deshpande & Nowak (AISTATS 2015), Borda reduction; with successive elimination",
          url="https://proceedings.mlr.press/v38/jamieson15.html",
          summary="Borda reduction in rounds; drop candidates whose Borda score is confidently below the leader's.")
def borda_elimination(st: DuelState, budget: int | None = None, **_: Any) -> DuelResult:
    alive = np.ones(st.K, dtype=bool)
    while True:
        out = _stop(st, budget, "borda_elimination")
        if out:
            return out
        for i in np.flatnonzero(alive):
            st.compare_random(int(i))
        est, lo, up = st.borda_bounds()
        best_lo = lo[alive].max()
        alive &= up >= best_lo
        if alive.sum() <= 1:
            alive[int(np.argmax(est))] = True


@register("duel", "savage", kind="fixed confidence",
          paper="Urvoy, Clerot, Féraud & Naamane (ICML 2013), SAVAGE",
          url="https://proceedings.mlr.press/v28/urvoy13.html",
          summary="Sample the least-explored pair whose outcome could still change the winner; skip settled pairs.")
def savage(st: DuelState, budget: int | None = None, **_: Any) -> DuelResult:
    """SAVAGE for the Borda winner. A pair stops mattering when its interval excludes 1/2 or when
    neither candidate can still be the winner; among the rest, the least-sampled pair is compared."""
    while True:
        out = _stop(st, budget, "savage")
        if out:
            return out
        est, blo, bup = st.borda_bounds()
        contender = bup >= blo.max()
        lo, up = st.pair_bounds()
        open_pair = (lo < 0.5) & (up > 0.5)
        relevant = open_pair & (contender[:, None] | contender[None, :])
        np.fill_diagonal(relevant, False)
        if not relevant.any():
            relevant = contender[:, None] & ~np.eye(st.K, dtype=bool)
        n = np.where(relevant, st.n, np.inf)
        i, j = np.unravel_index(int(np.argmin(n)), n.shape)
        st.compare(int(i), int(j))


# ---------------------------------------------------------------- entry point
def find_best_pairwise(task: DuelTask, method: str = "borda_lucb", *, delta: float = 0.05,
                       epsilon: float = 0.0, budget: int | None = None, seed: int | None = 0,
                       **options: Any) -> DuelResult:
    """Find the Borda winner with few pairwise comparisons.

    Certified results are, with probability at least ``1 - delta``, within ``epsilon`` (in win rate)
    of the true Borda winner. With ``budget`` set, the method stops there and returns its best guess.
    """
    info = get("duel", method)
    st = DuelState(task, delta=delta, seed=seed, epsilon=epsilon)
    return info.fn(st, budget=budget, **options)


# ------------------------------------------------------------------ tasks
def bradley_terry(wins: np.ndarray, ties: np.ndarray | None = None, iters: int = 500, prior: float = 1.0) -> np.ndarray:
    """Bradley-Terry strengths from pairwise wins (MM algorithm, Hunter 2004), ties counted as half wins.

    A small symmetric prior (``prior`` virtual draws per pair) keeps strengths finite.
    """
    W = wins.astype(float) + (0.5 * ties if ties is not None else 0)
    K = W.shape[0]
    W = W + prior * 0.5 * (1 - np.eye(K))
    N = W + W.T
    s = np.ones(K)
    for _ in range(iters):
        denom = (N / (s[:, None] + s[None, :])).sum(1)
        s_new = W.sum(1) / np.maximum(denom, 1e-12)
        s_new /= np.exp(np.log(s_new).mean())
        if np.max(np.abs(s_new - s)) < 1e-10:
            s = s_new
            break
        s = s_new
    return s


def from_votes(wins: np.ndarray, ties: np.ndarray, names: Sequence[str], min_votes: int = 50,
               name: str = "votes") -> DuelTask:
    """A replay task from vote counts: empirical win rates where a pair has at least ``min_votes``
    votes, Bradley-Terry predictions elsewhere."""
    wins, ties = np.asarray(wins, float), np.asarray(ties, float)
    n = wins + wins.T + ties
    emp = np.divide(wins + 0.5 * ties, n, out=np.full_like(wins, 0.5), where=n > 0)
    s = bradley_terry(wins, ties)
    bt = s[:, None] / (s[:, None] + s[None, :])
    prefs = np.where(n >= min_votes, emp, bt)
    np.fill_diagonal(prefs, 0.5)
    prefs = (prefs + 1 - prefs.T) / 2  # exact symmetry
    return DuelTask(name, list(names), prefs=prefs,
                    info={"empirical_pairs": int((np.triu(n, 1) >= min_votes).sum()),
                          "pairs": len(names) * (len(names) - 1) // 2})


def synthetic_duel(n_candidates: int = 8, seed: int = 0, spread: float = 1.0) -> DuelTask:
    """Bradley-Terry candidates with random strengths."""
    rng = np.random.default_rng(seed)
    theta = rng.normal(0, spread, n_candidates)
    prefs = 1 / (1 + np.exp(-(theta[:, None] - theta[None, :])))
    return DuelTask("synthetic_duel", [f"model_{k}" for k in range(n_candidates)], prefs=prefs)


def from_judge(candidates: Sequence[Any], judge: Callable[[Any, Any], float],
               names: Sequence[str] | None = None) -> DuelTask:
    """A live task. ``judge(a, b)`` returns 1.0 if a is better, 0.0 if b is better, 0.5 for a tie,
    e.g. an LLM judge comparing the two candidates' answers on a random prompt."""
    cands = list(candidates)
    names = list(names) if names else [str(c)[:40] for c in cands]
    return DuelTask("live", names, judge=lambda i, j: judge(cands[i], cands[j]))


ARENA = __import__("os").path.join(__import__("os").path.dirname(__file__), "data", "arena_votes.npz")


def arena_votes() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Human vote counts from Chatbot Arena (lmarena-ai/arena-human-preference-55k, Apache-2.0): 59 models."""
    with np.load(ARENA) as f:
        return f["wins"].astype(float), f["ties"].astype(float), [str(m) for m in f["models"]]


def arena(n_candidates: int = 8, seed: int = 0, pick: str = "random", min_votes: int = 50) -> DuelTask:
    """Chatbot Arena replay: ``n_candidates`` models, real human preference rates between them.

    pick="random": random models. pick="top": the strongest models by Bradley-Terry strength (close, hard).
    Pairs with fewer than ``min_votes`` votes use Bradley-Terry predictions fitted on all votes.
    """
    wins, ties, names = arena_votes()
    s = bradley_terry(wins, ties)
    rng = np.random.default_rng(seed)
    if pick == "top":
        idx = np.argsort(-s)[:n_candidates]
    else:
        idx = rng.choice(len(names), n_candidates, replace=False)
    sub_w, sub_t = wins[np.ix_(idx, idx)], ties[np.ix_(idx, idx)]
    n = sub_w + sub_w.T + sub_t
    emp = np.divide(sub_w + 0.5 * sub_t, n, out=np.full_like(sub_w, 0.5), where=n > 0)
    bt = s[idx][:, None] / (s[idx][:, None] + s[idx][None, :])
    prefs = np.where(n >= min_votes, emp, bt)
    np.fill_diagonal(prefs, 0.5)
    prefs = (prefs + 1 - prefs.T) / 2
    return DuelTask(f"arena/{pick}", [names[i] for i in idx], prefs=prefs,
                    info={"seed": seed, "empirical_pairs": int((np.triu(n, 1) >= min_votes).sum()),
                          "pairs": n_candidates * (n_candidates - 1) // 2})
