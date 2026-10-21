"""Where scores come from.

A *source* answers "what is candidate k's score on example i?" with a number in
[0, 1]. MatrixSource replays scores you already have (for simulation and
benchmarks); FunctionSource calls your own scoring function, e.g. an LLM call
plus a grader, in parallel and with an on-disk cache.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Protocol, Sequence

import numpy as np


class Source(Protocol):
    names: list[str]

    @property
    def n_candidates(self) -> int: ...

    @property
    def n_examples(self) -> int: ...

    def score_many(self, pairs: Sequence[tuple[int, int]]) -> list[float]: ...


def _check(value: float, where: str) -> float:
    value = float(value)
    if not (0.0 <= value <= 1.0) or value != value:
        raise ValueError(f"score for {where} must be a number in [0, 1], got {value!r}")
    return value


class MatrixSource:
    """Scores you already have: an array of shape (n_candidates, n_examples)."""

    def __init__(self, scores: Any, names: Sequence[str] | None = None):
        arr = np.asarray(scores, dtype=float)
        if arr.ndim != 2:
            raise ValueError("scores must be 2-D: one row per candidate, one column per example")
        if np.isnan(arr).any() or arr.min() < 0 or arr.max() > 1:
            raise ValueError("scores must be in [0, 1] with no missing values")
        self.scores = arr
        self.names = list(names) if names is not None else [f"candidate_{k}" for k in range(arr.shape[0])]
        if len(self.names) != arr.shape[0]:
            raise ValueError("need one name per candidate")
        self.calls = 0

    @property
    def n_candidates(self) -> int:
        return self.scores.shape[0]

    @property
    def n_examples(self) -> int:
        return self.scores.shape[1]

    def score_many(self, pairs: Sequence[tuple[int, int]]) -> list[float]:
        self.calls += len(pairs)
        return [float(self.scores[k, i]) for k, i in pairs]


class _Cache:
    """Tiny SQLite cache so re-running an evaluation never pays for the same call twice."""

    def __init__(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS scores (key TEXT PRIMARY KEY, score REAL NOT NULL)")
        self._db.commit()

    def get(self, key: str) -> float | None:
        with self._lock:
            row = self._db.execute("SELECT score FROM scores WHERE key = ?", (key,)).fetchone()
        return None if row is None else row[0]

    def put(self, key: str, score: float) -> None:
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO scores(key, score) VALUES (?, ?)", (key, score))
            self._db.commit()


def _stable_key(obj: Any) -> str:
    try:
        text = json.dumps(obj, sort_keys=True, default=str)
    except TypeError:
        text = repr(obj)
    return hashlib.sha256(text.encode()).hexdigest()[:24]


class FunctionSource:
    """Scores computed on demand by ``score_fn(candidate, example) -> float in [0, 1]``.

    Calls run in a thread pool (LLM calls are I/O bound). With ``cache_path`` set,
    results are stored in SQLite keyed by the candidate and the example contents,
    so interrupted or repeated runs are free.
    """

    def __init__(self, candidates: Sequence[Any], examples: Sequence[Any],
                 score_fn: Callable[[Any, Any], float], names: Sequence[str] | None = None,
                 max_workers: int = 8, cache_path: str | None = None):
        if not candidates or not examples:
            raise ValueError("need at least one candidate and one example")
        self.candidates = list(candidates)
        self.examples = list(examples)
        self.score_fn = score_fn
        self.names = list(names) if names is not None else [_short_name(c, k) for k, c in enumerate(candidates)]
        self.max_workers = max(1, int(max_workers))
        self.cache = _Cache(cache_path) if cache_path else None
        self._cand_keys = [_stable_key(c) for c in self.candidates]
        self._ex_keys = [_stable_key(e) for e in self.examples]
        self.calls = 0  # calls that actually ran score_fn (cache misses)

    @property
    def n_candidates(self) -> int:
        return len(self.candidates)

    @property
    def n_examples(self) -> int:
        return len(self.examples)

    def _one(self, pair: tuple[int, int]) -> float:
        k, i = pair
        key = f"{self._cand_keys[k]}:{self._ex_keys[i]}"
        if self.cache is not None:
            hit = self.cache.get(key)
            if hit is not None:
                return hit
        value = _check(self.score_fn(self.candidates[k], self.examples[i]), f"{self.names[k]} on example {i}")
        self.calls += 1
        if self.cache is not None:
            self.cache.put(key, value)
        return value

    def score_many(self, pairs: Sequence[tuple[int, int]]) -> list[float]:
        if len(pairs) == 1 or self.max_workers == 1:
            return [self._one(p) for p in pairs]
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(pairs))) as pool:
            return list(pool.map(self._one, pairs))


def _short_name(candidate: Any, k: int) -> str:
    if isinstance(candidate, str):
        return candidate if len(candidate) <= 40 else candidate[:37] + "..."
    if isinstance(candidate, dict):
        label = candidate.get("name") or "/".join(str(candidate[x]) for x in ("model", "prompt") if x in candidate)
        if label:
            return str(label)[:40]
    return f"candidate_{k}"
