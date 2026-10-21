"""Plugin registry for methods.

Every method belongs to a *problem*:

* ``identify``  - find the best candidate with as few calls as possible
* ``estimate``  - estimate every candidate's full-test-set score from a few examples
* ``online``    - pick a candidate for each incoming example while learning (regret)

Add your own method with the ``register`` decorator; it then works everywhere
(the API, the CLI and the benchmark):

    from frugal_eval.registry import register

    @register("identify", "my_method", paper="Me et al. (2026)", url="https://...")
    def my_method(ev, budget=None, **opts):
        ...
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

PROBLEMS = ("identify", "estimate", "online")


@dataclass(frozen=True)
class MethodInfo:
    problem: str
    name: str
    fn: Callable
    paper: str = ""
    url: str = ""
    summary: str = ""
    kind: str = ""          # e.g. "fixed confidence", "fixed budget", "baseline", "contextual"
    needs_prior: bool = False     # needs results of other models on the same examples
    needs_features: bool = False  # needs a feature vector per example


_REGISTRY: dict[str, dict[str, MethodInfo]] = {p: {} for p in PROBLEMS}


def register(problem: str, name: str, *, paper: str = "", url: str = "", summary: str = "",
             kind: str = "", needs_prior: bool = False, needs_features: bool = False):
    if problem not in PROBLEMS:
        raise ValueError(f"problem must be one of {PROBLEMS}")

    def decorator(fn: Callable) -> Callable:
        _REGISTRY[problem][name] = MethodInfo(problem, name, fn, paper, url, summary or (fn.__doc__ or "").strip()
                                              .split("\n")[0], kind, needs_prior, needs_features)
        return fn

    return decorator


def get(problem: str, name: str) -> MethodInfo:
    _load_builtin()
    try:
        return _REGISTRY[problem][name]
    except KeyError:
        known = ", ".join(sorted(_REGISTRY.get(problem, {})))
        raise ValueError(f"unknown {problem} method {name!r}; available: {known}") from None


def list_methods(problem: str | None = None) -> list[MethodInfo]:
    """All registered methods, optionally for one problem."""
    _load_builtin()
    problems = [problem] if problem else list(PROBLEMS)
    return [info for p in problems for info in _REGISTRY[p].values()]


def _load_builtin() -> None:
    # importing the modules registers their methods
    from . import estimate, identify, online  # noqa: F401
