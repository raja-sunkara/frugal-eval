# Benchmarks

Every method on the same real data: per-question results of 395 models from the Open LLM Leaderboard (via tinyBenchmarks, MIT license). Tasks: arc_challenge, gsm8k, hellaswag, mmlu, truthfulqa, winogrande. 20 trials per task. Reproduce with `python benchmarks/run_all.py`, or run the Benchmark workflow on GitHub.

## Find the best candidate

10 random leaderboard models per trial; confidence 95%. Fixed-confidence methods stop when the winner is certified; fixed-budget methods get 30% of the full cost. Cells: calls saved versus running everything, and how often the true best model was returned.

| Method | Type | arc_challenge | gsm8k | hellaswag | mmlu | truthfulqa | winogrande | Avg saved |
|---|---|---|---|---|---|---|---|---|
| `lucb` | fixed confidence | 21% saved, 100% right | 65% saved, 100% right | 32% saved, 100% right | 70% saved, 100% right | 30% saved, 100% right | 11% saved, 100% right | **38%** |
| `successive_elimination` | fixed confidence | 21% saved, 100% right | 65% saved, 100% right | 32% saved, 100% right | 70% saved, 100% right | 30% saved, 100% right | 11% saved, 100% right | **38%** |
| `track_and_stop` | fixed confidence | 12% saved, 100% right | 47% saved, 100% right | 20% saved, 100% right | 53% saved, 100% right | 23% saved, 100% right | 6% saved, 100% right | **27%** |
| `eb_tc` | fixed confidence | 8% saved, 100% right | 55% saved, 100% right | 15% saved, 100% right | 60% saved, 100% right | 14% saved, 100% right | 4% saved, 100% right | **26%** |
| `lil_ucb` | fixed confidence | 8% saved, 100% right | 47% saved, 100% right | 11% saved, 100% right | 29% saved, 100% right | 13% saved, 100% right | 5% saved, 100% right | **19%** |
| `top_two_ts` | fixed confidence | 7% saved, 100% right | 24% saved, 100% right | 7% saved, 100% right | 36% saved, 100% right | 14% saved, 100% right | 3% saved, 100% right | **15%** |
| `uniform` | baseline | 2% saved, 100% right | 14% saved, 100% right | 4% saved, 100% right | 30% saved, 100% right | 8% saved, 100% right | 1% saved, 100% right | **10%** |
| `successive_halving` | fixed budget | 70% saved, 100% right | 70% saved, 100% right | 70% saved, 100% right | 70% saved, 95% right | 70% saved, 100% right | 70% saved, 95% right | 70% (preset) |
| `successive_rejects` | fixed budget | 70% saved, 85% right | 70% saved, 100% right | 70% saved, 95% right | 70% saved, 100% right | 70% saved, 95% right | 70% saved, 90% right | 70% (preset) |

## Estimate every score from 100 examples

10 random leaderboard models per trial, 100 examples each; methods that learn from history use 300 other leaderboard models. Cells: mean absolute error in percentage points and rank correlation with the true scores.

| Method | Type | arc_challenge | gsm8k | hellaswag | mmlu | truthfulqa | winogrande | Avg error |
|---|---|---|---|---|---|---|---|---|
| `irt` | model-based | 1.8 pts (rank ρ 0.88) | 2.3 pts (rank ρ 0.97) | 1.1 pts (rank ρ 0.90) | 2.1 pts (rank ρ 0.91) | 1.3 pts (rank ρ 0.97) | 1.5 pts (rank ρ 0.80) | **1.7 pts** |
| `anchor_points` | representative subset | 1.8 pts (rank ρ 0.94) | 3.2 pts (rank ρ 0.96) | 1.2 pts (rank ρ 0.92) | 2.1 pts (rank ρ 0.93) | 1.4 pts (rank ρ 0.96) | 2.6 pts (rank ρ 0.84) | **2.1 pts** |
| `stratified` | classical | 2.4 pts (rank ρ 0.82) | 2.2 pts (rank ρ 0.96) | 1.8 pts (rank ρ 0.84) | 2.7 pts (rank ρ 0.89) | 1.9 pts (rank ρ 0.95) | 2.3 pts (rank ρ 0.78) | **2.2 pts** |
| `random` | baseline | 3.8 pts (rank ρ 0.90) | 2.9 pts (rank ρ 0.95) | 2.6 pts (rank ρ 0.82) | 3.7 pts (rank ρ 0.88) | 2.8 pts (rank ρ 0.92) | 3.6 pts (rank ρ 0.79) | **3.2 pts** |

## Online selection and routing

10 random leaderboard models; every question in the benchmark arrives once in random order and the method picks one model per question. Cells: average score and its share of what the best single model would score. Above 100% means routing beat every single model.

| Method | Type | arc_challenge | gsm8k | hellaswag | mmlu | truthfulqa | winogrande | Avg vs best single |
|---|---|---|---|---|---|---|---|---|
| `lin_ts` | contextual | 0.712 (102%) | 0.680 (103%) | 0.882 (101%) | 0.795 (111%) | 0.658 (101%) | 0.844 (102%) | **103%** |
| `linucb` | contextual | 0.701 (101%) | 0.678 (103%) | 0.881 (101%) | 0.795 (111%) | 0.649 (100%) | 0.844 (102%) | **103%** |
| `thompson` | non-contextual | 0.649 (93%) | 0.619 (94%) | 0.858 (98%) | 0.706 (99%) | 0.598 (92%) | 0.798 (97%) | **95%** |
| `epsilon_greedy` | non-contextual | 0.660 (95%) | 0.590 (90%) | 0.866 (99%) | 0.709 (99%) | 0.599 (92%) | 0.806 (98%) | **95%** |
| `kl_ucb` | non-contextual | 0.645 (93%) | 0.609 (92%) | 0.855 (98%) | 0.703 (98%) | 0.580 (89%) | 0.791 (96%) | **94%** |
| `ucb1` | non-contextual | 0.622 (89%) | 0.540 (82%) | 0.838 (96%) | 0.677 (94%) | 0.540 (83%) | 0.779 (94%) | **90%** |
| `exp3` | adversarial | 0.623 (89%) | 0.499 (76%) | 0.836 (96%) | 0.666 (93%) | 0.531 (81%) | 0.778 (94%) | **88%** |
| `uniform` | baseline | 0.605 (87%) | 0.299 (45%) | 0.815 (93%) | 0.574 (80%) | 0.503 (77%) | 0.767 (93%) | **79%** |
