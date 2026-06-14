# gpt2-alpine-c Changelog

## 2026-06-14 (session 5 — C warnings + NORTHSTAR + emily train stats)

- fix(c): tokenizer.c fread return value checks — build now warning-clean (-Wall -Wextra); exits with diagnostic on read failure
- docs(northstar): Milestone 1.5 DONE; corpus stats added (327 LM / 456 instruct records); PPL baseline 116.76 reference; updated key files table
- feat(emily.cli): emily train stats subcommand — shells to corpus_stats.py; auto-discovers gpt2-alpine-c; -v for source breakdown

## 2026-06-14 (session 4 — corpus quality + eval hardening)

- feat(corpus): scripts/corpus_stats.py — pre-upload quality checker; token estimate, Colab training time, source breakdown, duplicate detection; exits non-zero on quality failure
- fix(corpus): retain _source metadata in output JSONL (harmless to Trainer; needed by corpus_stats.py source breakdown); LM corpus 327 records / ~110k tokens; instruct 456 records / ~133k tokens
- feat(eval): eval_perplexity.py --memory-efficient flag (batch=1 / max_length=64 / eval_frac≤5%); prevents CPU OOM on GPT-2 forward pass
- eval: updated PPL baseline — 16-example run: loss=4.7601, PPL=116.76 (WebText baseline: ~29; target post-FT: <60)

## 2026-06-14 (session 3 — S26-06)

- feat(corpus/S26-06): prime_directive_to_instruct() — 8 hardcoded Emily identity/protocol Q&A pairs + section-level pairs from prime directive ## headings; instruct corpus: 128 pairs / 455 records / 568KB
- eval(S26-06): GPT-2 base perplexity on Emily corpus = 130.33 (max_length=64, seed=42); saved to var/perplexity-baseline.json; post-fine-tune target PPL < 60
- test: 6 new tests for prime_directive_to_instruct; 28 total, all green
- Apple #488 filed; S26-06 marked done in EMILY/BACKLOG.md

## 2026-06-14 (session 2)

- feat(corpus): Apple log as corpus source #6 — reads APPLES git repo JSON files; auto-discovers APPLES sibling; adds --apples-dir / --max-apples CLI flags; 321 → 327 records on live repo
- feat(eval): scripts/eval_perplexity.py — HuggingFace perplexity eval; compare base vs fine-tuned checkpoint; per-source breakdown; preps S26-05 entropy validation
- test: tests/test_dataset.py — 22 unit tests for chunk_text, make_lm_records, backlog_to_instruct, parse_golden_index, apples_to_records, build_corpus; all green
- build: Makefile add 'make test' target (compile + Python tests)
- chore: commit tokenizer.json + tokenizer_config.json (BPE tokenizer, required by C engine, previously untracked)

## 2026-06-14

- feat(training): prime_directive_dataset.py — builds JSONL training corpus from Emily golden docs + prime directive + RSI task history
- feat(training): drive_sync.py — uploads/downloads training artifacts via IDUNA Drive API (EMILY-TRAINING agent auth)
- feat(training): convert_ft_checkpoint.py — converts HuggingFace fine-tuned checkpoint to C binary weight format (extends existing convert_checkpoint.py)
- feat(training): gpt2_finetune_colab.ipynb — complete Colab notebook: mounts Drive, loads training JSONL, HuggingFace Trainer fine-tune, saves checkpoints
- docs: NORTHSTAR.md — repo identity, architecture diagram, milestone table, entropy source integration
- docs: CLAUDE.md — agent instructions + training pipeline steps

## 2026-06-03

- Initial C inference engine: GPT-2 small (12L/12H/768D) forward pass, BPE tokenizer, entropy stats
- convert_checkpoint.py: HuggingFace gpt2 base checkpoint → flat binary C format
- Dockerfile: Alpine/musl build environment (zero runtime dependencies)
- Makefile: make → gpt2_run binary
