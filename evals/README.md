# Consulting-graph retrieval evals

Golden-set regression harness for the deployed search pipeline (dense + BM25
hybrid, RRF fusion, Cohere rerank). 66 gold queries (`golden.json`), 52
reachable against the current corpus.

```bash
.venv/bin/python evals/run_eval.py               # full pipeline vs live DB
.venv/bin/python evals/run_eval.py --baseline    # dense-only, for deltas
.venv/bin/python evals/run_eval_mrl.py --db consulting_eval_2048 --dims 2048
.venv/bin/python evals/bench_latency.py          # dense-stage latency + EXPLAIN
.venv/bin/python evals/ablation.py               # stage ablation
```

Quality gate: **R@5 >= 0.65 and R@20 >= 0.85**. Run the full eval before
merging anything that touches retrieval (the 2026-07 master merge regressed
R@20 from 0.904 to 0.673 via a well-intentioned SQL dedup; only this harness
caught it).

## Matryoshka dimensions experiment (2026-07-13)

qwen3-embedding-8b is MRL-trained: truncating a stored 4096-dim vector to its
first N dims (+ L2 renorm) yields a valid N-dim embedding, so lower-dim
indexes cost **zero re-embedding**. `make_mrl_index.py` builds them from the
live index. Measured on 6,154 chunks / 1,033 documents:

| | 4096 (current) | 2048 | 1024 |
|---|---|---|---|
| R@1 | 0.423 | 0.404 | 0.423 |
| R@5 | 0.750 | 0.750 | 0.769 |
| R@10 | 0.788 | 0.788 | 0.808 |
| R@20 | 0.865 | 0.865 | 0.846 |
| Quality gate | pass | **pass** | fail (one query) |
| Dense p50 | 68ms | 38ms | 30ms |
| Table size | 140MB | 74MB | 52MB |

Findings:

- **2048 reproduces 4096's gate quality exactly** at half the storage and
  ~45% lower dense-stage latency.
- **1024 beats 4096 at every k <= 10** but drops one tail query out of the
  top-20; since results feed an LLM reading the whole top-20, tail recall
  matters. Likely recoverable with a wider rerank pool (75-100): untested.
- **HNSW is irrelevant at this corpus size.** The planner picks a sequential
  scan at ~6k rows even with the index available (`enable_seqscan=off`
  proves the index itself works, ~7ms). Vector WIDTH, not indexing, drives
  latency today. HNSW (pgvector caps it at 2000 dims, so 1024 only) becomes
  meaningful around 10x corpus growth.

## Decision (2026-07-13): stay on 4096, documented option to switch

At today's corpus size the gains are modest (66MB disk; 30ms on a pipeline
dominated by the rerank API call), so the switch is deferred. Revisit when
the corpus grows several-fold or dense-stage latency starts to matter.

### Migration recipe (2048), when wanted

1. Teach the query path to truncate: `OllamaEmbedder`/`OpenRouterEmbedder`
   must return 2048-dim query embeddings (truncate first 2048 + L2 renorm,
   exactly as `make_mrl_index.py` does for documents). Verify with a unit
   test against a recorded 4096 vector.
2. Build the target DB: `make_mrl_index.py --dims 2048 --target-db <name>`
   (a validated copy already exists as `consulting_eval_2048`).
3. Point the stack at it: set `OPENROUTER_EMBED_DIMS=2048` and
   `EMBEDDING_DIMENSIONS=2048` in `docker-compose.consulting.yml` and switch
   `POSTGRES_DB` (or rename databases). Keep the 4096 DB until verified.
4. The startup `embedding_signature` check will flag the change until the
   first clean reconcile records `openrouter:qwen/qwen3-embedding-8b:2048`.
5. Verify: full eval run must pass the gate; then drop the old DB when
   comfortable.

Rollback at any point = repoint `POSTGRES_DB` at the untouched 4096 DB.
