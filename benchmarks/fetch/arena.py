"""Download Chatbot Arena human votes and keep only per-pair vote counts.

Writes out/arena.npz with:
  models  (K,)        model names with at least MIN_BATTLES battles
  wins    (K, K)      wins[i, j]  = votes where model i beat model j
  ties    (K, K)      ties[i, j]  = tied votes between i and j (symmetric)
and out/report.json (source, license from the data card, sizes). No prompts or
answers are kept: only aggregate counts.
"""
from __future__ import annotations

import json
import os
import re
import sys

import numpy as np

OUT = sys.argv[1] if len(sys.argv) > 1 else "out"
SOURCES = [  # tried in order
    ("lmarena-ai/arena-human-preference-55k", "train.csv"),
    ("lmsys/chatbot_arena_conversations", None),
]
MIN_BATTLES = 300


def outcomes(df) -> tuple[list, list, list]:
    a, b = df["model_a"].astype(str).tolist(), df["model_b"].astype(str).tolist()
    if "winner" in df:  # chatbot_arena_conversations: model_a / model_b / tie / tie (bothbad)
        w = df["winner"].astype(str).tolist()
        res = [1.0 if x == "model_a" else 0.0 if x == "model_b" else 0.5 for x in w]
    else:  # arena-human-preference-55k: one-hot winner columns
        res = [1.0 if wa else 0.0 if wb else 0.5
               for wa, wb in zip(df["winner_model_a"].astype(int), df["winner_model_b"].astype(int))]
    return a, b, res


def main() -> None:
    import pandas as pd
    from huggingface_hub import hf_hub_download, list_repo_files

    os.makedirs(OUT, exist_ok=True)
    report: dict = {"attempts": []}
    df = None
    for repo, fname in SOURCES:
        try:
            card = open(hf_hub_download(repo, "README.md", repo_type="dataset"), encoding="utf-8").read()
            lic = re.search(r"^license:\s*(.+)$", card, re.M)
            files = list_repo_files(repo, repo_type="dataset")
            if fname is None:
                fname = next(f for f in files if f.endswith(".parquet"))
            path = hf_hub_download(repo, fname, repo_type="dataset")
            df = pd.read_csv(path) if fname.endswith(".csv") else pd.read_parquet(path)
            report.update(source=f"https://huggingface.co/datasets/{repo}", file=fname,
                          license=lic.group(1).strip() if lic else None, card_head=card[:1200],
                          rows=len(df), columns=[str(c) for c in df.columns])
            break
        except Exception as exc:  # gated or missing: try the next source
            report["attempts"].append({"repo": repo, "error": repr(exc)[:300]})
    if df is None:
        raise SystemExit("no arena source could be downloaded; see report.json")

    a, b, res = outcomes(df)
    counts: dict[str, int] = {}
    for x, y in zip(a, b):
        counts[x] = counts.get(x, 0) + 1
        counts[y] = counts.get(y, 0) + 1
    models = sorted(m for m, c in counts.items() if c >= MIN_BATTLES)
    idx = {m: i for i, m in enumerate(models)}
    K = len(models)
    wins, ties = np.zeros((K, K), np.int32), np.zeros((K, K), np.int32)
    for x, y, r in zip(a, b, res):
        if x in idx and y in idx and x != y:
            i, j = idx[x], idx[y]
            if r == 1.0:
                wins[i, j] += 1
            elif r == 0.0:
                wins[j, i] += 1
            else:
                ties[i, j] += 1
                ties[j, i] += 1
    np.savez_compressed(os.path.join(OUT, "arena.npz"), models=np.array(models), wins=wins, ties=ties)
    n = wins + wins.T + ties
    report.update(n_models=K, votes_kept=int(n.sum() // 2), pairs_with_votes=int((np.triu(n, 1) > 0).sum()),
                  pairs_total=K * (K - 1) // 2, models=models)
    with open(os.path.join(OUT, "report.json"), "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps({k: report[k] for k in ("source", "license", "rows", "n_models", "votes_kept",
                                             "pairs_with_votes", "pairs_total")}, indent=1))


if __name__ == "__main__":
    main()
