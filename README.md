# frugal-eval

**Find your best prompt or model without running every candidate on every example.**

Comparing 10 prompts on a 1,000-question test set normally costs 10,000 LLM calls. `frugal-eval` decides which candidate to test on which example next, drops the obvious losers early, and stops as soon as it can **certify** the winner, usually well before the full run finishes.

```python
from frugal_eval import find_best

result = find_best(candidates=prompts, examples=test_set, score_fn=run_and_grade)
print(result.summary())
```

Real output, 6 Open LLM Leaderboard models on GSM8K (1,319 questions):

```
candidate                    score           interval   examples
Pallas-0.5-LASER-0.1         0.612  [ 0.572,  0.651]  1100/1319  <- best
Pallas-0.5-LASER-exp2-0.1    0.529  [ 0.490,  0.569]  1100/1319
Mistral-7B-Instruct-v0.2     0.359  [ 0.242,  0.477]   451/1319
TowerInstruct-7B-v0.1        0.153  [ 0.000,  0.358]   176/1319
orca_mini_v3_7b              0.057  [ 0.000,  0.287]   140/1319
Tulpar-7b-v0                 0.044  [ 0.000,  0.279]   135/1319

Best: Pallas-0.5-LASER-0.1
Certified: best within 0 with probability >= 0.95.
Calls: 3,102 of 7,914 (61% saved)
```

The weak models were dropped after about 150 questions each; the full evaluation agrees on the winner.

## Results on real data

Replayed on real per-example results from the **Open LLM Leaderboard** (395 models; data from [tinyBenchmarks](https://github.com/felipemaiapolo/tinyBenchmarks)). Each trial picks 10 random models, hides their scores, and lets `frugal-eval` choose which questions to "run". 30 trials per task, confidence 95%.

| Task | Questions | Picked the true best | Calls saved |
|---|---:|---:|---:|
| GSM8K | 1,319 | 100% | **71%** |
| TruthfulQA | 817 | 100% | **30%** |
| ARC-Challenge | 1,172 | 100% | **22%** |
| Winogrande | 1,267 | 100% | **14%** |

Savings depend on how far apart the candidates are. When one candidate is clearly better, most of the test set never needs to run. When the top candidates are nearly tied (for example the 10 strongest leaderboard models, often within a fraction of a point), proving which is best takes almost the full test set, and `frugal-eval` will tell you so rather than guess. Setting `epsilon=0.01` ("any candidate within one point of the best is fine") helps in that case.

Reproduce with `python benchmarks/run_benchmark.py`.

## Install

```bash
pip install git+https://github.com/raja-sunkara/frugal-eval
```

Python 3.9+. Dependencies: `numpy`, `httpx`.

## Usage

### Live: compare prompts or models

Write one function that runs a candidate on one example and returns a score between 0 and 1.

```python
from frugal_eval import find_best
from frugal_eval.llm import chat, last_number

candidates = [
    {"model": "gpt-4o-mini", "prompt": "Answer the question.\n\n{q}"},
    {"model": "gpt-4o-mini", "prompt": "Think step by step, then give the final number.\n\n{q}"},
    {"model": "gpt-4.1-nano", "prompt": "Think step by step, then give the final number.\n\n{q}"},
]
test_set = [{"q": "...", "answer": "42"}, ...]

def run_and_grade(candidate, example):
    reply = chat(candidate["model"], candidate["prompt"].format(q=example["q"]))
    return last_number(reply, example["answer"])

result = find_best(
    candidates=candidates,
    examples=test_set,
    score_fn=run_and_grade,
    delta=0.05,          # allowed chance of picking the wrong winner
    epsilon=0.0,         # accept anything within this much of the best
    cache_path="eval_cache.db",  # re-runs are free
    max_workers=8,       # parallel calls
)
print(result.summary())
```

`chat` works with any OpenAI-compatible API (OpenAI, Groq, Together, OpenRouter, vLLM, Ollama, a LiteLLM proxy). Set `OPENAI_BASE_URL` and `OPENAI_API_KEY`. Built-in scorers: `exact_match`, `contains`, `last_number`, `choice_letter`. Any grader works, including an LLM judge, as long as it returns a number in [0, 1].

### Offline: you already have scores

```python
result = find_best(scores)  # array of shape (n_candidates, n_examples)
```

Or from the command line, with a CSV that has one column per candidate and one row per example:

```bash
frugal-eval replay scores.csv --epsilon 0.01
```

## How it works

1. All candidates go through the test set in the **same random order**, so they are always compared on the same questions.
2. After each step, every candidate gets a **confidence interval** for the score it would get on the *whole* test set. The intervals use finite-population bounds (Hoeffding–Serfling and empirical Bernstein–Serfling, Bardenet & Maillard 2015), which shrink to zero once a candidate has seen every example. They hold simultaneously for all candidates and all steps, which is what makes stopping early safe.
3. **LUCB** (Kalyanakrishnan et al., 2012) chooses what to run next: the current leader and the candidate most likely to beat it.
4. It **stops** when the leader's lower bound clears every other candidate's upper bound (minus `epsilon`).

**The guarantee:** with probability at least `1 - delta`, the returned candidate scores within `epsilon` of the one a full evaluation would pick. With `budget=` set, it stops at the budget instead and says the answer is not certified.

Algorithms: `lucb` (default), `uniform` (same stopping rule, no adaptive sampling; a baseline), `successive_halving` (fixed budget, no guarantee).

## Roadmap

- Paired comparisons between candidates, for bigger savings when candidates are close
- Integrations with promptfoo, Inspect and lm-evaluation-harness
- Async live evaluation
- Hosted dashboard for teams

Issues and pull requests are welcome.

## Citation

If you use frugal-eval in research, please cite this repository. The benchmark data comes from tinyBenchmarks (Polo et al., 2024, MIT license).

## License

MIT
