# Benchmarks

Every method on the same real data: per-question results of 395 models from the Open LLM Leaderboard (via tinyBenchmarks, MIT license). 20 trials per task. Reproduce with `python benchmarks/run_all.py`.

## Find the best candidate

10 random leaderboard models per trial; confidence 95%. Fixed-confidence methods stop when the winner is certified; fixed-budget methods get 30% of the full cost. Cells: calls saved versus running everything, and how often the true best model was returned.

| Method | Type | gsm8k | arc_challenge | winogrande | truthfulqa | Avg saved |
|---|---|---|---|---|---|---|
| `lucb` | fixed confidence | 65% saved, 100% right | 21% saved, 100% right | 11% saved, 100% right | 30% saved, 100% right | **32%** |
| `successive_elimination` | fixed confidence | 65% saved, 100% right | 21% saved, 100% right | 11% saved, 100% right | 30% saved, 100% right | **32%** |
| `lil_ucb` | fixed confidence | 47% saved, 100% right | 8% saved, 100% right | 5% saved, 100% right | 13% saved, 100% right | **18%** |
| `top_two_ts` | fixed confidence | 24% saved, 100% right | 7% saved, 100% right | 3% saved, 100% right | 14% saved, 100% right | **12%** |
| `uniform` | baseline | 14% saved, 100% right | 2% saved, 100% right | 1% saved, 100% right | 8% saved, 100% right | **6%** |
| `successive_halving` | fixed budget | 70% saved, 100% right | 70% saved, 100% right | 70% saved, 95% right | 70% saved, 100% right | 70% (preset) |
| `successive_rejects` | fixed budget | 70% saved, 100% right | 70% saved, 85% right | 70% saved, 90% right | 70% saved, 95% right | 70% (preset) |

## Estimate every score from 100 examples

10 random leaderboard models per trial, 100 examples each; methods that learn from history use 300 other leaderboard models. Cells: mean absolute error in percentage points and rank correlation with the true scores.

| Method | Type | gsm8k | arc_challenge | winogrande | truthfulqa | Avg error |
|---|---|---|---|---|---|---|
| `irt` | model-based | 2.2 pts (rank ρ 0.97) | 1.8 pts (rank ρ 0.89) | 1.5 pts (rank ρ 0.80) | 1.3 pts (rank ρ 0.97) | **1.7 pts** |
| `stratified` | classical | 2.2 pts (rank ρ 0.96) | 2.4 pts (rank ρ 0.82) | 2.3 pts (rank ρ 0.78) | 1.9 pts (rank ρ 0.95) | **2.2 pts** |
| `anchor_points` | representative subset | 3.2 pts (rank ρ 0.96) | 1.8 pts (rank ρ 0.94) | 2.6 pts (rank ρ 0.83) | 1.4 pts (rank ρ 0.96) | **2.3 pts** |
| `random` | baseline | 2.9 pts (rank ρ 0.95) | 3.8 pts (rank ρ 0.90) | 3.6 pts (rank ρ 0.79) | 2.8 pts (rank ρ 0.92) | **3.3 pts** |

## Online selection and routing

10 random leaderboard models; every question in the benchmark arrives once in random order and the method picks one model per question. Cells: average score and its share of what the best single model would score. Above 100% means routing beat every single model.

| Method | Type | gsm8k | arc_challenge | winogrande | truthfulqa | Avg vs best single |
|---|---|---|---|---|---|---|
| `lin_ts` | contextual | 0.680 (103%) | 0.712 (102%) | 0.844 (102%) | 0.658 (101%) | **102%** |
| `linucb` | contextual | 0.678 (103%) | 0.701 (101%) | 0.844 (102%) | 0.649 (100%) | **101%** |
| `thompson` | non-contextual | 0.619 (94%) | 0.649 (93%) | 0.798 (97%) | 0.598 (92%) | **94%** |
| `epsilon_greedy` | non-contextual | 0.590 (90%) | 0.660 (95%) | 0.806 (98%) | 0.599 (92%) | **93%** |
| `kl_ucb` | non-contextual | 0.609 (92%) | 0.645 (93%) | 0.791 (96%) | 0.580 (89%) | **92%** |
| `ucb1` | non-contextual | 0.540 (82%) | 0.622 (89%) | 0.779 (94%) | 0.540 (83%) | **87%** |
| `exp3` | adversarial | 0.499 (76%) | 0.623 (89%) | 0.778 (94%) | 0.531 (81%) | **85%** |
| `uniform` | baseline | 0.299 (45%) | 0.605 (87%) | 0.767 (93%) | 0.503 (77%) | **75%** |
