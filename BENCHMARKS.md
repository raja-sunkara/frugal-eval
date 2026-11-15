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
| `irt` | model-based | 1.8 pts (rank ρ 0.88) | 2.3 pts (rank ρ 0.97) | 1.1 pts (rank ρ 0.90) | 2.1 pts (rank ρ 0.91) | 1.3 pts (rank ρ 0.96) | 1.5 pts (rank ρ 0.81) | **1.7 pts** |
| `anchor_points` | representative subset | 1.8 pts (rank ρ 0.94) | 3.2 pts (rank ρ 0.96) | 1.2 pts (rank ρ 0.92) | 2.1 pts (rank ρ 0.93) | 1.4 pts (rank ρ 0.96) | 2.6 pts (rank ρ 0.83) | **2.1 pts** |
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
| `discounted_thompson` | non-stationary | 0.649 (93%) | 0.602 (91%) | 0.847 (97%) | 0.677 (94%) | 0.586 (90%) | 0.789 (95%) | **94%** |
| `sliding_window_ucb` | non-stationary | 0.640 (92%) | 0.601 (91%) | 0.837 (96%) | 0.663 (92%) | 0.570 (87%) | 0.789 (95%) | **92%** |
| `ucb1` | non-contextual | 0.622 (89%) | 0.540 (82%) | 0.838 (96%) | 0.677 (94%) | 0.540 (83%) | 0.779 (94%) | **90%** |
| `exp3` | adversarial | 0.623 (89%) | 0.499 (76%) | 0.836 (96%) | 0.666 (93%) | 0.531 (81%) | 0.778 (94%) | **88%** |
| `discounted_ucb` | non-stationary | 0.615 (88%) | 0.490 (74%) | 0.826 (95%) | 0.613 (86%) | 0.530 (81%) | 0.775 (94%) | **86%** |
| `uniform` | baseline | 0.605 (87%) | 0.299 (45%) | 0.815 (93%) | 0.574 (80%) | 0.503 (77%) | 0.767 (93%) | **79%** |

## Cost-aware routing on RouterBench

RouterBench (Hu et al., 2024): 36,497 real prompts, 11 LLMs, the dollar cost of every answer. Every prompt arrives once; the method picks one model per prompt and learns from the score it gets, penalised by cost times the cost weight. Cells: quality and total cost relative to always using the single best model (GPT-4). Contextual methods use a prompt embedding.

| Method | Type | cost weight 0 | cost weight 0.5 | cost weight 1 |
|---|---|---|---|---|
| `linucb` | contextual | 99% quality, 82% cost | 94% quality, 23% cost | 88% quality, 13% cost |
| `lin_ts` | contextual | 99% quality, 77% cost | 93% quality, 23% cost | 89% quality, 15% cost |
| `thompson` | non-contextual | 100% quality, 99% cost | 83% quality, 13% cost | 82% quality, 7% cost |
| `kl_ucb` | non-contextual | 99% quality, 98% cost | 84% quality, 18% cost | 81% quality, 7% cost |
| `ucb1` | non-contextual | 97% quality, 92% cost | 82% quality, 25% cost | 78% quality, 11% cost |
| `exp3` | adversarial | 95% quality, 85% cost | 82% quality, 28% cost | 77% quality, 13% cost |
| `discounted_thompson` | non-stationary | 95% quality, 86% cost | 81% quality, 28% cost | 77% quality, 14% cost |
| `sliding_window_ucb` | non-stationary | 92% quality, 77% cost | 79% quality, 29% cost | 75% quality, 17% cost |
| `epsilon_greedy` | non-contextual | 100% quality, 99% cost | 89% quality, 43% cost | 81% quality, 6% cost |
| `discounted_ucb` | non-stationary | 80% quality, 45% cost | 73% quality, 29% cost | 71% quality, 23% cost |
| `uniform` | baseline | 66% quality, 25% cost | 66% quality, 25% cost | 66% quality, 25% cost |

## Online selection when a model silently degrades

10 random leaderboard models; questions arrive in order and, halfway through, the best model's answers are replaced by those of a model about half as good (a silent model swap behind an endpoint). Cells: total score as a share of what the best model of each half would have scored.

| Method | Type | drift/mmlu | drift/hellaswag | drift/gsm8k | Avg |
|---|---|---|---|---|---|
| `kl_ucb` | non-contextual | 95.6% | 97.6% | 88.0% | **93.7%** |
| `thompson` | non-contextual | 94.7% | 97.4% | 86.7% | **93.0%** |
| `discounted_thompson` | non-stationary | 93.9% | 96.8% | 86.8% | **92.5%** |
| `sliding_window_ucb` | non-stationary | 92.3% | 95.9% | 87.9% | **92.0%** |
| `epsilon_greedy` | non-contextual | 92.8% | 97.9% | 80.7% | **90.5%** |
| `ucb1` | non-contextual | 92.7% | 95.9% | 77.6% | **88.7%** |
| `exp3` | adversarial | 91.1% | 95.3% | 70.1% | **85.5%** |
| `discounted_ucb` | non-stationary | 85.8% | 94.3% | 69.7% | **83.3%** |
| `uniform` | baseline | 79.8% | 91.5% | 44.0% | **71.8%** |

## Pairwise comparisons on Chatbot Arena

Real human votes from Chatbot Arena (lmarena-ai/arena-human-preference-55k, Apache-2.0): 8 models per trial, either random or the 8 strongest (nearly tied). Each comparison is one simulated vote drawn from the real win rate of that pair. Target: the model with the highest average win rate. Cells: median votes until a model within 2 points of the best is certified (95% confidence, capped at 300,000), and how often the pick is exactly right after a fixed number of votes.

| Method | Type | arena/random: votes to certify (median) | arena/random: right after 1,000 votes | arena/random: right after 5,000 votes | arena/top: votes to certify (median) | arena/top: right after 1,000 votes | arena/top: right after 5,000 votes |
|---|---|---|---|---|---|---|---|
| `borda_lucb` | fixed confidence | 43,706 | 75% | 80% | 173,430 | 40% | 40% |
| `borda_elimination` | fixed confidence | 44,968 (95% certified) | 75% | 80% | 177,380 | 60% | 90% |
| `round_robin` | baseline | 70,530 (80% certified) | 85% | 80% | 300,000 (0% certified) | 70% | 60% |
| `savage` | fixed confidence | 300,000 (30% certified) | 85% | 90% | 208,868 (90% certified) | 70% | 60% |
