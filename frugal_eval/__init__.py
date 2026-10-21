"""frugal-eval: find your best prompt or model with a fraction of the evaluation calls."""
from .core import ALGORITHMS, Evaluation, Result, find_best
from .sources import FunctionSource, MatrixSource

__all__ = ["find_best", "Result", "Evaluation", "MatrixSource", "FunctionSource", "ALGORITHMS"]
__version__ = "0.1.0"
