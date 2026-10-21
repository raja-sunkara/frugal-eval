"""frugal-eval: methods from the research literature for cheaper LLM evaluation and online model selection.

Three problems, one interface:

    import frugal_eval as fe

    task = fe.tasks.make("leaderboard/gsm8k", n_candidates=10, seed=0)
    fe.identify(task, method="lucb")          # find the best candidate with few calls
    fe.estimate(task, method="irt", budget=100)  # estimate every score from 100 examples
    fe.run_online(task, method="linucb")      # route each request while learning

    fe.list_methods()                          # every method, with its paper
"""
from . import tasks
from .estimate import EstimateResult, estimate
from .identify import ALGORITHMS, Evaluation, Result, find_best, identify
from .online import OnlineResult, Policy, run_online
from .registry import MethodInfo, list_methods, register
from .sources import FunctionSource, MatrixSource
from .tasks import Task

__all__ = [
    "identify", "estimate", "run_online", "find_best", "list_methods", "register", "tasks", "Task",
    "Result", "EstimateResult", "OnlineResult", "Evaluation", "Policy", "MethodInfo",
    "MatrixSource", "FunctionSource", "ALGORITHMS",
]
__version__ = "0.2.0"
