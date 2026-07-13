#!/usr/bin/env python3
"""Thin launcher for the Matryoshka (MRL) retrieval eval.

Sets EVAL_DIMS / EVAL_DB *before* importing run_eval (which captures
EMBEDDING_DIMENSIONS at import time and passes it down to vector_store), then
delegates to run_eval.main(). Query embeddings are generated once at 4096 and
truncated + renormalized to --dims in-process, so a truncated MRL index can be
searched with the same OpenRouter query vector. No new embedding API calls
beyond what run_eval already makes (the provider cache is width-stable at 4096).

    .venv/bin/python evals/run_eval_mrl.py --db consulting_eval_1024 --dims 1024
    .venv/bin/python evals/run_eval_mrl.py --db consulting_eval_2048 --dims 2048
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--db", required=True, help="target database name (on port 5434)")
ap.add_argument("--dims", type=int, required=True, help="index/column width to search")
ap.add_argument("--baseline", action="store_true", help="dense-only (rerank off)")
args = ap.parse_args()

os.environ["EVAL_DIMS"] = str(args.dims)
os.environ["EVAL_DB"] = args.db

# run_eval.main() runs its own argparse over sys.argv; hand it only the flags it
# understands.
sys.argv = [sys.argv[0]] + (["--baseline"] if args.baseline else [])

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_eval  # noqa: E402  (import after env is set so EMBEDDING_DIMENSIONS binds correctly)

raise SystemExit(asyncio.run(run_eval.main()))
