# GPT-2 Pure C (Alpine / musl) — Minimal Reference

See SETUP.md for conversion & build steps. See `CLAUDE.md` for the full build/serve/training-pipeline reference.

## Recent additions (2026-08)

- **`vectorcache`** (S150-02, Go port of `docs/reference/vector_cache.md`) — semantic LLM-context
  cache with Merkle-style per-node content hashing (query+response+embedding+children).
- **towerprint-augmented training records** (S150-01) — teaches a fine-tuned checkpoint to
  natively produce the house transform (the 2020 original's technique: feed the tower back to
  the model as part of its own training data).
- **`corpus_stats` provenance audit mode** (S146-05) — HQ-SPEC-AI-103 §7's contamination-audit
  metric (must be zero findings) plus verdict/source-pipeline/license-class breakdown for a
  snapshot corpus.
- **`gpt2-serve.service`** (`ops/systemd/`) — processifies `scripts/serve.py`, replacing manual
  restarts of the inference API server.
