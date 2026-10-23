"""Pilot: Jev (TypeSafe AI) as a feature source for online routing on RouterBench.

For a stratified sample of RouterBench prompts, Jev answers a few typed questions
about each prompt (task type, difficulty, whether a small model would suffice,
whether it needs multi-step reasoning). Its probabilities are saved as features,
together with latency and token usage, so the routing methods in frugal-eval can
be compared with Jev features, prompt-embedding features, and both.

Runs in GitHub Actions (.github/workflows/jev-pilot.yml). Uses TYPESAFE_API_KEY
(direct API) or AI_GATEWAY_API_KEY (Vercel AI Gateway), whichever is set.
Only progress counts are printed; the output file is encrypted before upload.
"""
from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests

N = int(os.environ.get("JEV_N", "2000"))
OUT = sys.argv[1] if len(sys.argv) > 1 else "out"
MAX_CHARS = 2000

if os.environ.get("TYPESAFE_API_KEY"):
    URL, KEY, MODEL = "https://api.typesafe.ai/v1/systemone", os.environ["TYPESAFE_API_KEY"], "jev-latest"
elif os.environ.get("AI_GATEWAY_API_KEY"):
    URL, KEY, MODEL = ("https://ai-gateway.vercel.sh/typesafe/v1/systemone", os.environ["AI_GATEWAY_API_KEY"],
                       "typesafe-ai/jev")
else:
    raise SystemExit("set TYPESAFE_API_KEY or AI_GATEWAY_API_KEY as a repository secret")

TASK_TYPES = {
    "math": "Arithmetic, algebra or word problems that need a numeric answer",
    "code": "Writing, completing or explaining program code",
    "knowledge": "Multiple-choice or factual questions about a subject (science, law, history, medicine...)",
    "commonsense": "Everyday reasoning: completing a scenario or resolving an ambiguous sentence",
    "language": "Translation, riddles, wordplay or questions about a specific language or culture",
    "open_ended": "Open-ended writing, summarising, chat or advice",
}
DIFFICULTY = ["very easy", "easy", "medium", "hard", "very hard"]
QUESTIONS = {
    "task_type": {"type": "choice", "instructions": "What kind of task is this prompt?", "criteria": TASK_TYPES},
    "difficulty": {"type": "score", "instructions": "How hard is it for a language model to answer this prompt "
                   "correctly?", "criteria": DIFFICULTY},
    "small_model_ok": {"type": "noul", "instructions": "Would a small, cheap 7-billion-parameter language model "
                       "most likely answer this prompt correctly?"},
    "multi_step": {"type": "noul", "instructions": "Does answering correctly require multi-step reasoning or "
                   "calculation?"},
}
FEATURE_NAMES = ([f"task_type={k}" for k in TASK_TYPES] + [f"difficulty={d}" for d in DIFFICULTY]
                 + ["difficulty_score", "small_model_ok", "multi_step", "task_type_confidence",
                    "difficulty_confidence"])


def ask(prompt: str) -> tuple[list[float], float, int, str]:
    body = {"model": MODEL, "state": prompt[:MAX_CHARS], "questions": QUESTIONS}
    headers = {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}
    for attempt in range(6):
        t0 = time.time()
        try:
            r = requests.post(URL, json=body, headers=headers, timeout=60)
        except requests.RequestException as exc:
            err = repr(exc)
            time.sleep(2 ** attempt)
            continue
        latency = time.time() - t0
        if r.status_code == 429 or r.status_code >= 500:
            err = f"http {r.status_code}"
            time.sleep(float(r.headers.get("retry-after", 2 ** attempt)))
            continue
        if r.status_code != 200:
            return [], latency, 0, f"http {r.status_code}: {r.text[:300]}"
        d = r.json()
        a = d["answers"]
        tt, df = a["task_type"], a["difficulty"]
        feats = ([float(tt["probabilities"].get(k, 0.0)) for k in TASK_TYPES]
                 + [float(df["probabilities"].get(str(i), 0.0)) for i in range(len(DIFFICULTY))]
                 + [float(df["score"]) / (len(DIFFICULTY) - 1), float(a["small_model_ok"]["noul"]),
                    float(a["multi_step"]["noul"]), float(tt.get("confidence", 0.0)),
                    float(df.get("confidence", 0.0))])
        tokens = int(d.get("usage", {}).get("input_tokens", 0)) + int(d.get("usage", {}).get("output_tokens", 0))
        return feats, latency, tokens, ""
    return [], 0.0, 0, err


def main() -> None:
    import pandas as pd
    from huggingface_hub import hf_hub_download

    df = pd.read_pickle(hf_hub_download("withmartian/routerbench", "routerbench_0shot.pkl", repo_type="dataset"))
    rng = np.random.default_rng(0)
    # stratified sample: the same share of every benchmark as in the full set
    groups = df.groupby("eval_name").indices
    idx = []
    for name, rows in groups.items():
        k = max(1, round(N * len(rows) / len(df)))
        idx.extend(rng.choice(rows, min(k, len(rows)), replace=False).tolist())
    idx = np.sort(np.array(idx[: N] if len(idx) > N else idx))
    prompts = [str(p) for p in df["prompt"].to_numpy()[idx]]

    # one probe call first, so a bad key or schema fails fast without spending
    feats, latency, tokens, err = ask(prompts[0])
    if err:
        raise SystemExit(f"probe call failed: {err}")
    print(f"probe ok ({latency * 1000:.0f} ms); asking Jev about {len(prompts)} prompts", flush=True)

    results = [None] * len(prompts)
    done = 0
    with ThreadPoolExecutor(max_workers=16) as pool:
        for i, res in zip(range(len(prompts)), pool.map(ask, prompts)):
            results[i] = res
            done += 1
            if done % 250 == 0:
                print(f"  {done}/{len(prompts)}", flush=True)

    ok = np.array([bool(r[0]) for r in results])
    F = np.array([r[0] if r[0] else [np.nan] * len(FEATURE_NAMES) for r in results], dtype=np.float32)
    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(os.path.join(OUT, "jev_features.npz"), rows=idx, features=F, ok=ok,
                        latency=np.array([r[1] for r in results], np.float32),
                        tokens=np.array([r[2] for r in results], np.int32),
                        feature_names=np.array(FEATURE_NAMES))
    errors = [r[3] for r in results if r[3]]
    meta = {"n": len(prompts), "ok": int(ok.sum()), "model_requested": MODEL, "endpoint": URL.split("/v1")[0],
            "errors_sample": errors[:5], "questions": QUESTIONS}
    with open(os.path.join(OUT, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"done: {int(ok.sum())}/{len(prompts)} prompts answered, {len(errors)} errors")


if __name__ == "__main__":
    main()
