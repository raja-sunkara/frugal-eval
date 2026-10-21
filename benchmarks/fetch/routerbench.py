"""Download RouterBench from Hugging Face and write a compact copy plus a report.

Used by .github/workflows/fetch-data.yml to check the upstream format; the library
itself downloads the data on first use (frugal_eval.datasets.load_routerbench).
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from frugal_eval.datasets import prepare_routerbench  # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 else "out"
os.makedirs(OUT, exist_ok=True)
report: dict = {"source": "https://huggingface.co/datasets/withmartian/routerbench"}
path = prepare_routerbench(os.path.join(OUT, "routerbench.npz"), embed=True, report=report)
report["npz_bytes"] = os.path.getsize(path)
with open(os.path.join(OUT, "report.json"), "w") as f:
    json.dump(report, f, indent=1)
print(json.dumps(report, indent=1))
