"""Small helpers for evaluating LLM prompts and models.

``chat`` talks to any OpenAI-compatible endpoint (OpenAI, Groq, Together,
OpenRouter, vLLM, Ollama, LiteLLM proxy, ...). Scorers turn a model answer into
a score in [0, 1].
"""
from __future__ import annotations

import os
import re
import string
import time
from typing import Any

import httpx

_client: httpx.Client | None = None


def chat(model: str, prompt: str | list[dict], *, system: str | None = None, temperature: float = 0.0,
         max_tokens: int = 512, base_url: str | None = None, api_key: str | None = None,
         retries: int = 4, timeout: float = 60.0, **extra: Any) -> str:
    """Return the text of one chat completion.

    Uses OPENAI_BASE_URL / OPENAI_API_KEY from the environment unless given.
    Retries on rate limits and server errors with exponential backoff.
    """
    global _client
    if _client is None:
        _client = httpx.Client(timeout=timeout)
    base = (base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
    key = api_key or os.environ.get("OPENAI_API_KEY", "")
    messages = prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}]
    if system:
        messages = [{"role": "system", "content": system}, *messages]
    body = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens, **extra}
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    for attempt in range(retries + 1):
        try:
            resp = _client.post(f"{base}/chat/completions", json=body, headers=headers)
        except httpx.TransportError:
            if attempt == retries:
                raise
            time.sleep(2 ** attempt)
            continue
        if resp.status_code in (429, 500, 502, 503, 504) and attempt < retries:
            time.sleep(float(resp.headers.get("retry-after", 2 ** attempt)))
            continue
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"] or ""
    raise RuntimeError("unreachable")


# ---------------------------------------------------------------- scorers
def normalize(text: str) -> str:
    text = str(text).lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction: str, answer: str) -> float:
    """1.0 if the normalized answer equals the normalized prediction."""
    return float(normalize(prediction) == normalize(answer))


def contains(prediction: str, answer: str) -> float:
    """1.0 if the normalized answer appears in the prediction."""
    return float(normalize(answer) in normalize(prediction))


_NUMBER = re.compile(r"-?\d[\d,]*\.?\d*")


def last_number(prediction: str, answer: str | float, tol: float = 1e-6) -> float:
    """1.0 if the last number in the prediction equals the answer (GSM8K-style grading)."""
    found = _NUMBER.findall(str(prediction))
    if not found:
        return 0.0
    try:
        pred = float(found[-1].replace(",", "").rstrip("."))
        gold = float(str(answer).replace(",", ""))
    except ValueError:
        return 0.0
    return float(abs(pred - gold) <= tol * max(1.0, abs(gold)))


def choice_letter(prediction: str, answer: str) -> float:
    """1.0 if the first standalone option letter (A-E) in the prediction matches the answer."""
    m = re.search(r"\b([A-E])\b", str(prediction).upper())
    return float(bool(m) and m.group(1) == str(answer).strip().upper()[:1])
