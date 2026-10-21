"""External datasets that are downloaded on first use (not shipped with the package).

RouterBench (Hu et al., 2024, https://arxiv.org/abs/2403.12031): 36,497 prompts from
standard benchmarks answered by 11 LLMs, with a quality score and the dollar cost of
every answer. Its data card states no license, so frugal-eval does not redistribute
it: ``load_routerbench`` downloads it from Hugging Face and keeps a compact copy in
``~/.cache/frugal-eval`` (override with FRUGAL_EVAL_CACHE).

Downloading needs ``pip install "frugal-eval[routerbench]"`` (huggingface_hub, pandas);
prompt embeddings additionally use sentence-transformers if it is installed.
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np

ROUTERBENCH_REPO = "withmartian/routerbench"
ROUTERBENCH_FILE = "routerbench_0shot.pkl"


def cache_dir() -> str:
    path = os.environ.get("FRUGAL_EVAL_CACHE") or os.path.join(os.path.expanduser("~"), ".cache", "frugal-eval")
    os.makedirs(path, exist_ok=True)
    return path


def prepare_routerbench(out_path: str, embed: bool = True, report: dict | None = None) -> str:
    """Download RouterBench from Hugging Face and write a compact .npz to ``out_path``.

    Only scores, costs, the source benchmark of each prompt and (optionally) a 16-dimensional
    prompt embedding are kept; prompts and model answers are not stored.
    """
    try:
        import pandas as pd
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError('RouterBench needs: pip install "frugal-eval[routerbench]"') from exc
    report = report if report is not None else {}
    # The upstream file is a pandas pickle; loading it runs pickle code from that repository.
    df = pd.read_pickle(hf_hub_download(ROUTERBENCH_REPO, ROUTERBENCH_FILE, repo_type="dataset"))
    cols = [str(c) for c in df.columns]
    models = [c[: -len("|total_cost")] for c in cols if c.endswith("|total_cost") and c[: -len("|total_cost")] in cols]
    if not models:
        raise ValueError("unexpected RouterBench format: no '<model>|total_cost' columns")
    scores = df[models].astype(float).to_numpy().T
    costs = df[[m + "|total_cost" for m in models]].astype(float).to_numpy().T
    keep = np.isfinite(scores).all(0) & np.isfinite(costs).all(0)
    scores, costs = np.clip(scores[:, keep], 0, 1), costs[:, keep]
    evals = df["eval_name"].astype(str).to_numpy()[keep]
    names, codes = np.unique(evals, return_inverse=True)
    arrays: dict[str, Any] = dict(scores=scores.astype(np.float16), costs=costs.astype(np.float32),
                                  benchmark=codes.astype(np.int16), benchmark_names=np.array([str(n) for n in names]),
                                  models=np.array([str(m) for m in models]))
    report.update(shape=list(scores.shape), models=models, dropped_rows=int((~keep).sum()))
    if embed and "prompt" in df:
        try:
            from sentence_transformers import SentenceTransformer

            prompts = [str(p)[:2000] for p in df["prompt"].to_numpy()[keep]]
            enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            full = enc.encode(prompts, batch_size=256, show_progress_bar=False, normalize_embeddings=True)
            centered = full - full.mean(0)
            _, _, vt = np.linalg.svd(centered[:20000], full_matrices=False)
            arrays["embedding"] = (centered @ vt[:16].T).astype(np.float16)
            report["embedding"] = "all-MiniLM-L6-v2 reduced to 16 dims with PCA"
        except ImportError:
            report["embedding"] = "skipped: sentence-transformers not installed"
    np.savez_compressed(out_path, **arrays)
    return out_path


def load_routerbench(path: str | None = None) -> dict[str, np.ndarray]:
    """RouterBench arrays, downloading and caching them on first use."""
    path = path or os.path.join(cache_dir(), "routerbench.npz")
    if not os.path.exists(path):
        prepare_routerbench(path)
    with np.load(path) as f:
        return {k: f[k] for k in f.files}
