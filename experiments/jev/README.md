# Jev pilot (experimental, not part of the library)

Does Jev (TypeSafe AI) make a useful, cheap feature source for online routing?
For a stratified sample of RouterBench prompts, Jev answers four typed questions
(task type, difficulty, whether a small model would suffice, whether multi-step
reasoning is needed). Its probabilities then serve as context features for the
routing methods in frugal-eval, compared against prompt embeddings.

Run: add a repository secret `TYPESAFE_API_KEY` (direct API) or `AI_GATEWAY_API_KEY`
(Vercel AI Gateway), then run the `jev-pilot` workflow on this branch. The output is
encrypted before it is stored.
