"""Download RouterBench (Hu et al., 2024) from Hugging Face and store a compact copy.

RouterBench: real prompts from several benchmarks, answered by 11 LLMs, with each
answer's quality score and dollar cost. Used for cost-aware routing.

Runs in GitHub Actions (see .github/workflows/fetch-data.yml), which has open
internet access. Writes:
  out/routerbench.npz   scores (models x prompts), costs, benchmark of each prompt,
                        a small prompt embedding, model names
  out/report.json       what was found (columns, shapes, license text), for review
"""
from __future__ import annotations

import json
import os
import re
import sys

import numpy as np

OUT = sys.argv[1] if len(sys.argv) > 1 else "out"
REPO = "withmartian/routerbench"
FILE = "routerbench_0shot.pkl"


def main() -> None:
    from huggingface_hub import hf_hub_download

    os.makedirs(OUT, exist_ok=True)
    report: dict = {"source": f"https://huggingface.co/datasets/{REPO}", "file": FILE}
    try:
        card = hf_hub_download(REPO, "README.md", repo_type="dataset")
        text = open(card, encoding="utf-8").read()
        report["card_head"] = text[:1500]
        m = re.search(r"^license:\s*(.+)$", text, re.M)
        report["license"] = m.group(1).strip() if m else None
    except Exception as exc:  # the card is informational only
        report["card_error"] = repr(exc)

    import pandas as pd

    path = hf_hub_download(REPO, FILE, repo_type="dataset")
    df = pd.read_pickle(path)
    report["shape"] = list(df.shape)
    report["columns"] = [str(c) for c in df.columns]
    report["dtypes"] = {str(c): str(t) for c, t in df.dtypes.items()}
    report["head"] = df.head(3).astype(str).map(lambda s: s[:120]).to_dict(orient="list")

    cols = [str(c) for c in df.columns]
    models = [c[: -len("|total_cost")] for c in cols if c.endswith("|total_cost") and c[: -len("|total_cost")] in cols]
    report["models"] = models
    if not models:
        raise SystemExit("could not find '<model>' and '<model>|total_cost' columns; see report.json")

    scores = df[models].astype(float).to_numpy().T
    costs = df[[m + "|total_cost" for m in models]].astype(float).to_numpy().T
    keep = np.isfinite(scores).all(0) & np.isfinite(costs).all(0)
    report["dropped_rows"] = int((~keep).sum())
    scores, costs = np.clip(scores[:, keep], 0, 1), costs[:, keep]
    evals = df["eval_name"].astype(str).to_numpy()[keep] if "eval_name" in df else np.array(["unknown"] * keep.sum())
    names, codes = np.unique(evals, return_inverse=True)
    report["benchmarks"] = {str(n): int((codes == i).sum()) for i, n in enumerate(names)}
    report["mean_score"] = dict(zip(models, np.round(scores.mean(1), 4).tolist()))
    report["mean_cost_usd"] = dict(zip(models, np.round(costs.mean(1), 6).tolist()))

    emb = None
    if "prompt" in df:
        try:
            from sentence_transformers import SentenceTransformer

            prompts = [str(p)[:2000] for p in df["prompt"].to_numpy()[keep]]
            enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")  # Apache-2.0
            full = enc.encode(prompts, batch_size=256, show_progress_bar=False, normalize_embeddings=True)
            centered = full - full.mean(0)
            _, _, vt = np.linalg.svd(centered[: min(len(centered), 20000)], full_matrices=False)
            emb = (centered @ vt[:16].T).astype(np.float16)
            report["embedding"] = "all-MiniLM-L6-v2, reduced to 16 dims with PCA"
        except Exception as exc:
            report["embedding_error"] = repr(exc)

    np.savez_compressed(
        os.path.join(OUT, "routerbench.npz"),
        scores=scores.astype(np.float16), costs=costs.astype(np.float32), benchmark=codes.astype(np.int16),
        benchmark_names=names, models=np.array(models),
        **({"embedding": emb} if emb is not None else {}),
    )
    report["npz_bytes"] = os.path.getsize(os.path.join(OUT, "routerbench.npz"))
    with open(os.path.join(OUT, "report.json"), "w") as f:
        json.dump(report, f, indent=1, default=str)
    print(json.dumps({k: report[k] for k in ("shape", "models", "license", "npz_bytes") if k in report}, indent=1))


if __name__ == "__main__":
    main()
