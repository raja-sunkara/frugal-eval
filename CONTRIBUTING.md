# Contributing

Thanks for helping. The goal of frugal-eval is to make methods from the literature easy to use and fair to compare, so every addition should come with a paper, tests and benchmark numbers.

## Adding a method

1. Pick a published method (conference, journal or a well-cited preprint) and open an issue naming the paper.
2. Implement it in the matching module and register it:
   - `frugal_eval/identify.py`: a function `fn(ev, budget=None, **options) -> Result`. Use the shared stopping rule (`ev.done()`) for fixed-confidence methods so results are comparable.
   - `frugal_eval/estimate.py`: a function `fn(task, budget, rng, query, **options) -> (estimates, examples, info)`.
   - `frugal_eval/online.py`: a `Policy` subclass with `select(x)` and `update(arm, reward, x)`.
3. Give `register(...)` the paper, a link, a one-line summary and the method type.
4. Add a test in `tests/` and run `pytest`.
5. Run `python benchmarks/run_all.py` and include the new BENCHMARKS.md rows in the pull request.

Keep implementations faithful to the paper. If you simplify something, say so in the docstring.

## Development

```bash
pip install -e ".[dev]"
pytest -q
frugal-eval demo
```
