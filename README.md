# frugal-eval

**Methods from the research literature for cheaper LLM evaluation and online model selection: one interface, benchmarked on real data.**

[![tests](https://github.com/raja-sunkara/frugal-eval/actions/workflows/tests.yml/badge.svg)](https://github.com/raja-sunkara/frugal-eval/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/raja-sunkara/frugal-eval/blob/main/examples/quickstart.ipynb)

Evaluating LLMs is expensive: 10 prompts on a 1,000-question test set is 10,000 calls. A lot of that work is wasted, and the bandits, active testing and psychometrics literature has good methods for not wasting it. They are scattered across papers, each with its own code. frugal-eval puts them behind one interface and ranks them on the same real data.

It covers three problems:

| Problem | Question it answers | Example |
|---|---|---|
| **Find the best** (`identify`) | Which prompt or model is best, using as few calls as possible? | Certified winner with up to 75% fewer calls |
| **Estimate scores** (`estimate`) | What would each model score on the full benchmark, from a few questions? | Scores within ~2 points from 100 of 1,319 questions |
| **Online selection** (`run_online`) | Which model should answer each incoming request while you are still learning? | Routing that beats every single model |

## Try it in 30 seconds (no API key)

```bash
pip install git+https://github.com/raja-sunkara/frugal-eval
frugal-eval demo --task gsm8k
```

This runs every method on real per-question results of 395 Open LLM Leaderboard models, which ship with the package. Or open the [Colab notebook](https://colab.research.google.com/github/raja-sunkara/frugal-eval/blob/main/examples/quickstart.ipynb).

## Usage

```python
import frugal_eval as fe

task = fe.tasks.make("leaderboard/gsm8k", n_candidates=10, seed=0)   # real data, 10 models

fe.identify(task, method="lucb").summary()                # find the best with few calls
fe.estimate(task, method="irt", budget=100).summary()     # estimate every score from 100 questions
fe.run_online(task, method="linucb").summary()            # route each question while learning

fe.list_methods()                                         # every method, with its paper
```

Real output of `fe.identify(task, method="lucb")`:

```
candidate                              score           interval   examples
NeuralPipe-7B-slerp                    0.675  [ 0.629,  0.721]  1044/1319  <- best
Tess-M-v1.3                            0.579  [ 0.533,  0.624]  1044/1319
alpaca-lora-65b-en-pt-es-ca            0.254  [ 0.088,  0.419]   268/1319
...
Llama-2-7b-ft-instruct-es              0.025  [ 0.000,  0.331]    80/1319

Best: NeuralPipe-7B-slerp
Certified: best within 0 with probability >= 0.95.
Calls: 3,247 of 13,190 (75% saved)
```

### Your own prompts and models

Write one function that runs a candidate on one example and returns a score in [0, 1]:

```python
import frugal_eval as fe
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

task = fe.tasks.from_function(candidates, test_set, run_and_grade,
                              cache_path="eval_cache.db",  # re-runs are free
                              max_workers=8)               # parallel calls
print(fe.identify(task, method="lucb", delta=0.05).summary())
```

`chat` works with any OpenAI-compatible API (OpenAI, Groq, Together, OpenRouter, vLLM, Ollama, a LiteLLM proxy) via `OPENAI_BASE_URL` and `OPENAI_API_KEY`. Built-in graders: `exact_match`, `contains`, `last_number`, `choice_letter`; any function returning a number in [0, 1] works, including an LLM judge.

Methods that learn from history (`irt`, `anchor_points`, `stratified`) take `prior=`: results of other models on the same examples. Contextual routing methods take `features=`: one vector per example (for example a prompt embedding).

Already have scores? `fe.identify(scores_array)` or `frugal-eval replay scores.csv`.

## Methods

| Problem | Method | Type | Paper | What it does |
|---|---|---|---|---|
| Find the best | `uniform` | baseline | Baseline | Every candidate on the same examples in lockstep, same stopping rule as the adaptive methods. |
| Find the best | `lucb` | fixed confidence | [Kalyanakrishnan, Tewari, Auer & Stone (ICML 2012)](https://proceedings.mlr.press/v23/kalyanakrishnan12/kalyanakrishnan12.pdf) | Evaluate the current leader and its most dangerous challenger. |
| Find the best | `successive_elimination` | fixed confidence | [Even-Dar, Mannor & Mansour (JMLR 2006)](https://www.jmlr.org/papers/volume7/evendar06a/evendar06a.pdf) | Evaluate all surviving candidates in rounds; drop any that is confidently worse than the leader. |
| Find the best | `lil_ucb` | fixed confidence | [Jamieson, Malloy, Nowak & Bubeck (COLT 2014)](https://proceedings.mlr.press/v35/jamieson14.pdf) | Evaluate the candidate with the highest law-of-the-iterated-logarithm UCB index. |
| Find the best | `top_two_ts` | fixed confidence | [Russo (COLT 2016; Operations Research 2020), Top-Two Thompson Sampling](https://arxiv.org/abs/1602.08448) | Sample a leader from the posterior; with probability 1 - beta evaluate a re-sampled challenger instead. |
| Find the best | `successive_halving` | fixed budget | [Karnin, Koren & Somekh (ICML 2013)](https://proceedings.mlr.press/v28/karnin13.pdf) | Split the budget into rounds; after each round keep the better half. |
| Find the best | `successive_rejects` | fixed budget | [Audibert, Bubeck & Munos (COLT 2010)](https://www.learningtheory.org/colt2010/papers/59Audibert.pdf) | Run K-1 phases of growing length; reject the worst candidate after each phase. |
| Estimate scores | `random` | baseline | Baseline | Run a uniformly random subset and report the plain average. |
| Estimate scores | `stratified` | classical | [Stratified sampling (Neyman, 1934), as used in efficient-benchmarking work](https://doi.org/10.2307/2342192) | Group examples by difficulty (from prior models), sample each group in proportion. |
| Estimate scores | `anchor_points` | representative subset | [Vivek, Ethayarajh, Yang & Kiela (EACL 2024), Anchor Points](https://arxiv.org/abs/2309.08638) | Cluster examples by how prior models answered them; run one representative per cluster, weight by cluster size. |
| Estimate scores | `irt` | model-based | [Polo, Weber, Choshen, Sun, Xu & Yurochkin (ICML 2024), tinyBenchmarks (p-IRT)](https://arxiv.org/abs/2402.14992) | Fit an item response model on prior models; infer each candidate's ability from anchor examples and predict the rest. |
| Online selection | `uniform` | baseline | Baseline | Pick a candidate uniformly at random. |
| Online selection | `epsilon_greedy` | non-contextual | [Auer, Cesa-Bianchi & Fischer (Machine Learning 2002), epsilon_n-greedy](https://doi.org/10.1023/A:1013689704352) | Mostly use the best so far; explore at random with a decaying probability. |
| Online selection | `ucb1` | non-contextual | [Auer, Cesa-Bianchi & Fischer (Machine Learning 2002)](https://doi.org/10.1023/A:1013689704352) | Optimism: pick the highest upper confidence bound. |
| Online selection | `kl_ucb` | non-contextual | [Garivier & Cappé (COLT 2011)](https://proceedings.mlr.press/v19/garivier11a/garivier11a.pdf) | Upper confidence bounds from the KL divergence; tighter than UCB1 for scores near 0 or 1. |
| Online selection | `thompson` | non-contextual | [Agrawal & Goyal (COLT 2012); Thompson (1933)](https://proceedings.mlr.press/v23/agrawal12/agrawal12.pdf) | Sample each candidate's score from its posterior and pick the highest sample. |
| Online selection | `exp3` | adversarial | [Auer, Cesa-Bianchi, Freund & Schapire (SIAM J. Computing 2002)](https://doi.org/10.1137/S0097539701398375) | Exponential weights with importance-weighted rewards; robust when quality drifts over time. |
| Online selection | `linucb` | contextual | [Li, Chu, Langford & Schapire (WWW 2010)](https://arxiv.org/abs/1003.0146) | Per-candidate ridge regression on example features with an upper confidence bonus: routes each request. |
| Online selection | `lin_ts` | contextual | [Agrawal & Goyal (ICML 2013), Thompson sampling for contextual bandits](https://proceedings.mlr.press/v28/agrawal13.pdf) | Linear Thompson sampling: sample each candidate's regression weights, route to the best sampled score. |

## Benchmarks

Real per-question results of 395 Open LLM Leaderboard models; 10 random models per trial, 20 trials on each of GSM8K, ARC-Challenge, Winogrande and TruthfulQA. Every method sees exactly the same problems.

| Problem | Best method today | Result | Baseline |
|---|---|---|---|
| Find the best | `lucb` / `successive_elimination` | **32% fewer calls** on average (65% on GSM8K), true best returned in 100% of trials | `uniform`: 6% |
| Estimate scores | `irt` | **1.7 points** average error from 100 questions per model (about 92% fewer calls) | `random`: 3.3 points |
| Online selection | `lin_ts` | **102%** of the best single model's score: routing beats every individual model | `uniform`: 75% |

Savings in "find the best" depend on how close the candidates are: when the top models are within a fraction of a point, certifying the winner needs most of the test set, and the methods say so rather than guess. The routing features in this benchmark are learned from other models' results on each question; with your own data, use a prompt embedding.

Full tables, per task: [BENCHMARKS.md](BENCHMARKS.md). Reproduce: `python benchmarks/run_all.py`.

## Add a method or a task

Every method is a plugin. Register it once and it works in the API, the CLI and the benchmark:

```python
import frugal_eval as fe

@fe.register("identify", "my_method", paper="Me et al. (2026)", url="https://arxiv.org/abs/...")
def my_method(ev, budget=None, **options):
    while not ev.done():
        ev.pull([...])          # evaluate candidates on their next example
    return ev.result("confidence", "my_method")
```

Online methods subclass `fe.Policy` (`select`, `update`); estimation methods return `(estimates, examples_used, info)`. Tasks are registered the same way with `fe.tasks.register_task`. New methods from recent papers are added regularly; contributions are welcome, see [CONTRIBUTING.md](CONTRIBUTING.md).

## How the guarantee works

All candidates go through the test set in the same random order, and comparisons use **paired differences** on the questions both have seen, which removes most of the noise. Confidence intervals use finite-population bounds (Hoeffding–Serfling and empirical Bernstein–Serfling, [Bardenet & Maillard 2015](https://arxiv.org/abs/1309.4029)) that hold for all candidates and all steps at once and shrink to zero once the whole test set is covered. Fixed-confidence methods stop when the leader beats every other candidate by more than the uncertainty: with probability at least `1 - delta`, the winner is within `epsilon` of what a full evaluation would pick. Fixed-budget methods spend the budget you give them and say the answer is not certified.

## Roadmap

Planned additions, all from published work:

- Track-and-Stop (Garivier & Kaufmann, COLT 2016) and Top-Two UCB (Jourdan & Degenne, 2022)
- Confidence sequences for sampling without replacement (Waudby-Smith & Ramdas, NeurIPS 2020)
- Low-rank evaluation acceleration (Zhou et al., ICLR 2025) and model-similarity methods (SySRs, ICML 2026)
- gp-IRT and more estimators from tinyBenchmarks (Polo et al., ICML 2024)
- Cost-aware routing between strong and cheap models
- Integrations with promptfoo, Inspect and lm-evaluation-harness; PyPI release

## Citation and data

If frugal-eval helps your research, please cite this repository and the papers of the methods you use (listed above). The bundled benchmark data comes from [tinyBenchmarks](https://github.com/felipemaiapolo/tinyBenchmarks) (Polo et al., 2024, MIT license); see `frugal_eval/data/NOTICE`.

## License

MIT
