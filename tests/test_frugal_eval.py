import csv

import numpy as np
import pytest

from frugal_eval import find_best
from frugal_eval.bounds import radius
from frugal_eval.cli import main
from frugal_eval.llm import contains, exact_match, last_number


def synthetic(gaps=(0.0, 0.1, 0.2, 0.3), n=2000, seed=0):
    rng = np.random.default_rng(seed)
    p = 0.8 - np.array(gaps)
    return (rng.random((len(p), n)) < p[:, None]).astype(float)


@pytest.mark.parametrize("algorithm", ["lucb", "uniform"])
def test_finds_best_and_saves_calls(algorithm):
    scores = synthetic()
    result = find_best(scores, algorithm=algorithm, delta=0.05)
    assert result.best == int(np.argmax(scores.mean(1)))
    assert result.certified and result.savings > (0.5 if algorithm == "lucb" else 0.2)


def test_lucb_cheaper_than_uniform():
    scores = synthetic(gaps=(0.0, 0.05, 0.3, 0.3, 0.3, 0.3))
    lucb = find_best(scores, algorithm="lucb")
    uniform = find_best(scores, algorithm="uniform")
    assert lucb.total_calls < uniform.total_calls


def test_exact_when_everything_is_evaluated():
    scores = np.array([[1, 0, 1, 0], [1, 0, 1, 0]], dtype=float)  # a perfect tie
    result = find_best(scores, delta=0.01)
    assert result.stopped_by in {"exhausted", "confidence"}
    assert np.allclose(result.means, 0.5)


def test_radius_collapses_on_full_population():
    r = radius(np.array([100, 50]), np.array([60.0, 30.0]), np.array([60.0, 30.0]), 100, 2, 0.05)
    assert r[0] == 0.0 and 0 < r[1] < 1


def test_budget_and_successive_halving():
    scores = synthetic()
    capped = find_best(scores, budget=200)
    assert capped.total_calls <= 210 and not capped.certified
    sh = find_best(scores, algorithm="successive_halving", budget=2000)
    assert sh.best == 0 and sh.total_calls <= 2000
    with pytest.raises(ValueError):
        find_best(scores, algorithm="successive_halving")


def test_live_function_source_with_cache(tmp_path):
    calls = []

    def score(candidate, example):
        calls.append((candidate, example))
        return float(example % candidate == 0)

    kwargs = dict(candidates=[1, 2, 3], examples=list(range(300)), score_fn=score,
                  cache_path=str(tmp_path / "cache.db"), max_workers=4)
    first = find_best(**kwargs)
    assert first.best_name == "candidate_0"  # candidate 1 divides everything
    n_calls = len(calls)
    second = find_best(**kwargs)
    assert len(calls) == n_calls  # everything came from the cache
    assert second.best == first.best


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        find_best(np.array([[0.5, 1.5], [0.2, 0.1]]))
    with pytest.raises(ValueError):
        find_best(np.ones((1, 5)))
    with pytest.raises(ValueError):
        find_best(candidates=[1, 2], examples=[1], score_fn=lambda c, e: 2.0)


def test_scorers():
    assert exact_match("The Paris.", "paris") == 1.0
    assert contains("I think it is Paris", "paris") == 1.0
    assert last_number("so the answer is 1,234.", "1234") == 1.0
    assert last_number("no number", "3") == 0.0


def test_cli(tmp_path, capsys):
    scores = synthetic(n=500)
    path = tmp_path / "scores.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["prompt_a", "prompt_b", "prompt_c", "prompt_d"])
        w.writerows(scores.T.astype(int).tolist())
    assert main(["replay", str(path)]) == 0
    out = capsys.readouterr().out
    assert "Best: prompt_a" in out and "saved" in out
