import csv

import numpy as np
import pytest

import frugal_eval as fe
from frugal_eval import find_best
from frugal_eval.bounds import radius
from frugal_eval.cli import main
from frugal_eval.llm import contains, exact_match, last_number
from frugal_eval.registry import list_methods

FIXED_BUDGET = {m.name for m in list_methods("identify") if m.kind == "fixed budget"}
FIXED_CONF = [m.name for m in list_methods("identify") if m.name not in FIXED_BUDGET]


def synthetic(gaps=(0.0, 0.1, 0.2, 0.3), n=2000, seed=0):
    rng = np.random.default_rng(seed)
    p = 0.8 - np.array(gaps)
    return (rng.random((len(p), n)) < p[:, None]).astype(float)


@pytest.fixture(scope="module")
def lb_task():
    return fe.tasks.make("leaderboard/gsm8k", n_candidates=6, seed=3, n_examples=600)


# ------------------------------------------------------------ registry
def test_every_method_has_a_paper_and_runs_through_the_registry():
    methods = list_methods()
    assert {m.problem for m in methods} == {"identify", "estimate", "online"}
    for m in methods:
        assert m.paper, m.name
    assert len(list_methods("identify")) >= 7 and len(list_methods("estimate")) >= 4
    assert len(list_methods("online")) >= 8


def test_custom_method_plugs_in():
    @fe.register("identify", "always_first", paper="test")
    def always_first(ev, **_):
        ev.pull([0])
        return ev.result("budget", "always_first")

    r = fe.identify(synthetic(), method="always_first")
    assert r.algorithm == "always_first" and r.total_calls == 1


# ------------------------------------------------------------ identify
@pytest.mark.parametrize("method", FIXED_CONF)
def test_fixed_confidence_methods_find_best(method):
    scores = synthetic()
    r = fe.identify(scores, method=method, delta=0.05)
    assert r.best == 0 and r.certified


@pytest.mark.parametrize("method", sorted(FIXED_BUDGET))
def test_fixed_budget_methods_respect_budget(method):
    r = fe.identify(synthetic(), method=method, budget=1500)
    assert r.best == 0 and r.total_calls <= 1500 and not r.certified
    with pytest.raises(ValueError):
        fe.identify(synthetic(), method=method)


def test_lucb_saves_more_than_uniform():
    scores = synthetic(gaps=(0.0, 0.05, 0.3, 0.3, 0.3, 0.3))
    assert fe.identify(scores, method="lucb").total_calls < fe.identify(scores, method="uniform").total_calls


def test_exact_tie_terminates():
    scores = np.array([[1, 0, 1, 0], [1, 0, 1, 0]], dtype=float)
    r = fe.identify(scores, delta=0.01)
    assert r.stopped_by in {"exhausted", "confidence"} and np.allclose(r.means, 0.5)


def test_radius_collapses_on_full_population():
    r = radius(np.array([100, 50]), np.array([60.0, 30.0]), np.array([60.0, 30.0]), 100, 2, 0.05)
    assert r[0] == 0.0 and 0 < r[1] < 1


def test_live_function_source_with_cache(tmp_path):
    calls = []

    def score(candidate, example):
        calls.append((candidate, example))
        return float(example % candidate == 0)

    kwargs = dict(candidates=[1, 2, 3], examples=list(range(300)), score_fn=score,
                  cache_path=str(tmp_path / "cache.db"), max_workers=4)
    first = find_best(**kwargs)
    assert first.best_name == "candidate_0"
    n_calls = len(calls)
    assert find_best(**kwargs).best == first.best and len(calls) == n_calls


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        find_best(np.array([[0.5, 1.5], [0.2, 0.1]]))
    with pytest.raises(ValueError):
        find_best(np.ones((1, 5)))
    with pytest.raises(ValueError):
        find_best(candidates=[1, 2], examples=[1], score_fn=lambda c, e: 2.0)
    with pytest.raises(ValueError):
        fe.identify(synthetic(), method="nope")


def test_identify_on_real_data(lb_task):
    r = fe.identify(lb_task, method="lucb")
    assert r.best == int(np.argmax(lb_task.truth)) and r.savings > 0


# ------------------------------------------------------------ estimate
@pytest.mark.parametrize("method", [m.name for m in list_methods("estimate")])
def test_estimate_methods(lb_task, method):
    r = fe.estimate(lb_task, method=method, budget=60)
    assert r.estimates.shape == (lb_task.n_candidates,)
    assert r.calls == 60 * lb_task.n_candidates
    assert r.mae < 0.1


def test_prior_methods_need_prior():
    with pytest.raises(ValueError):
        fe.estimate(synthetic(), method="irt", budget=50)


def test_irt_fit_recovers_synthetic_scores():
    # averaged over seeds so small numerical differences between platforms do not flip the result
    maes = [fe.estimate(fe.tasks.make("synthetic", n_candidates=5, n_examples=600, n_prior=200, seed=s),
                        method="irt", budget=60, seed=s).mae for s in range(3)]
    assert np.mean(maes) < 0.08


# ------------------------------------------------------------ online
@pytest.mark.parametrize("method", [m.name for m in list_methods("online")])
def test_online_methods(lb_task, method):
    r = fe.run_online(lb_task, method=method, horizon=400)
    assert len(r.choices) == 400 and r.oracle_reward >= r.best_fixed_reward
    assert 0 <= r.average_reward <= 1


def test_learning_beats_random(lb_task):
    rand = fe.run_online(lb_task, method="uniform").average_reward
    assert fe.run_online(lb_task, method="thompson").average_reward > rand


def test_contextual_needs_features():
    with pytest.raises(ValueError):
        fe.run_online(synthetic(), method="linucb")


# ------------------------------------------------------------ misc
def test_scorers():
    assert exact_match("The Paris.", "paris") == 1.0
    assert contains("I think it is Paris", "paris") == 1.0
    assert last_number("so the answer is 1,234.", "1234") == 1.0
    assert last_number("no number", "3") == 0.0


def test_cli_replay_and_methods(tmp_path, capsys):
    scores = synthetic(n=500)
    path = tmp_path / "scores.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["prompt_a", "prompt_b", "prompt_c", "prompt_d"])
        w.writerows(scores.T.astype(int).tolist())
    assert main(["replay", str(path)]) == 0
    assert "Best: prompt_a" in capsys.readouterr().out
    assert main(["methods"]) == 0
    assert "Kalyanakrishnan" in capsys.readouterr().out


def test_track_and_stop_weights_match_paper_example():
    # Garivier & Kaufmann (2016): mu = (0.5, 0.45, 0.43, 0.4) gives w* = (0.417, 0.390, 0.136, 0.057)
    from frugal_eval.identify import optimal_weights

    w = optimal_weights(np.array([0.5, 0.45, 0.43, 0.4]))
    assert np.allclose(w, [0.417, 0.390, 0.136, 0.057], atol=2e-3)
