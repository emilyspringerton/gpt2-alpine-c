# gpt2-alpine-c Changelog

## 2026-08-13

- Added gpt2-serve.service systemd unit (processified the manual serve.py restart from the reboot runbook) (sess-20260813-2154-dda37e8b)


## 2026-07-19

- Local GPT-2 training round abandoned mid-run (500 steps, ETA ~54h at observed rate — too slow/risky for this box's CPU-only setup). Confirms the existing COLAB_RUNBOOK.md guidance: real fine-tuning belongs on Colab, not local CPU training beyond a 50-step smoke test.


## 2026-07-17
- feat(colab): `scripts/colab_train.py` — moved all fine-tuning logic (corpus load, tokenize, `TrainingArguments`/`Trainer`, save+archive, perplexity eval, generation smoke test) out of the notebook and into a versioned, git-tracked script with CLI/env-var config. `notebooks/gpt2_finetune_colab.ipynb` collapsed from 11 cells to 3: markdown intro, one bootstrap cell (mount Drive, `git clone`-or-`pull` this repo, run `colab_train.py`), and the existing "Next Steps" markdown. Directly addresses founder's ask to stop pasting/managing training cells in Colab ad hoc — the bootstrap cell never changes; new training behavior ships as a commit and the next `git pull` in the same cell picks it up automatically. `docs/COLAB_RUNBOOK.md` §2 rewritten to match.
- docs(training): `docs/COLAB_RUNBOOK.md` — concrete step-by-step for Milestone 2 (Colab T4 fine-tune), written after two local `train_local.py` attempts were silently OOM-killed within seconds of starting (confirmed: FatBaby pipeline processes consume ~3GB RSS on this 3.8GB VM, leaving no headroom for even the memory-conscious bfloat16/LoRA local trainer). Covers corpus upload (manual, since `GOOGLE_DRIVE_SERVICE_ACCOUNT_JSON` isn't configured — Option B documents setting it up), notebook cell-by-cell walkthrough, checkpoint download, conversion (`convert_ft_checkpoint.py`), validation (`eval_perplexity.py` + `--entropy-stats`) against the existing +0.5 nats target, and the commit/Apple flow. Corpus rebuilt fresh same day: 1048 records reflecting the full current golden-docs-index (up from the stale 15:36 build, now including NORN/PRIME-097/TYLER additions).

## 2026-07-16

- feat(corpus/S146-01..04): `--fable-eps` preset, snapshot manifest v1, EPS-headline extraction, eval tombstoning — `prime_directive_dataset.py` gains `eps_headlines_to_records()` (var/eps/articles.ndjson joined against oracle.ndjson by source_identity, one record per article never chunked, per-record provenance including source_event_hash/oracle_case_id/verdict/license_class; verdict-gated — only "confirmed" cases are FABLE-eligible, pending/contradicted excluded and counted, not silently dropped), `write_snapshot()` (immutable content-addressed pair under var/snapshots/, snapshot_id = sha256 over sorted record hashes, manifest records builder git rev + args + tombstone-list hash — the contract Go `fabledata` inherits per HQ-SPEC-AI-103 §4a), `load_eval_tombstones()`/`apply_tombstones()` (var/eval-tombstones.json, applied after generation before dedupe/write). `--fable-eps` bypasses the general corpus builder entirely — EPS records only, implies `--snapshot`. 12 new tests (40 total, all green); verified live against the real (currently empty-of-confirmed-cases) EPS oracle store — correctly reports 0 confirmed / 2 pending, an honest result, not a bug.
- feat(towerprint/S147-01): new Go package `pkg/towerprint` (repo's first Go code; go.mod added, module joined to root go.work) — port of the 2020 QUEENSALLYONLINEBOOKOFMAGIFICATIONANDUNICOR squish/tower/gematria pipeline: `Squish`/`U2V`/`Tower`/`MatrixTower`/`ClassicTower`/`Codzeify` (big.Int, AZ/ZA dual) /`MagicCode`+`MagicTower` (3×8 VVV grid, lookup table now derived not hand-written) /`Compute` composite `Fingerprint`/`FortMinute` decimal-time seed. 13 table-driven tests pinned to vectors from the original Python + the VOIDONX artifact + executed COR.ipynb cells; all green.
- docs(towerprint): docs/TOWERPRINT.md — archaeology verdict on the 2020 repo (GPT-2 divination loop; trxtwr/magicVVV family is the evolved final variant), transform spec, Apple-fingerprint rationale, S147-02 wiring design; position taken: async caller-side enrichment (Apple lands, emily-agent worker PATCHes fingerprint), never sync. Registered in EMILY golden-docs-index (tier 2).
- S146-06: rebuilt Track A general corpus (327 → 1228 records; golden-docs-index.md grew substantially since 2026-06-14, plus the six new HQ-SPEC docs auto-ingested) and refreshed var/perplexity-baseline.json (116.76 → 166.56 PPL — expected: larger, more specialized corpus is more out-of-distribution for base GPT-2, not a regression). Old baseline preserved in a `history` array. Post-fine-tune target unchanged: PPL < 60 on the same eval settings.
- docs(northstar): Corpus Evolution — Bridge to FABLE E0 section: two-track corpus policy (general Emily corpus unchanged / narrow `--fable-eps` snapshot), fabledata placement call (separate Go component per HQ-SPEC-AI-103 §4a; this repo implements manifest v1 first in Python), `eps_headlines_to_records()` design over PRRJECT_FATBABY/var/eps with per-record provenance + verdict gating, snapshot manifest v1 shape, eval-tombstone mechanism (var/eval-tombstones.json + build-time exclusion). Backlog: EMILY SECTION 146 (S146-01..06)

## 2026-06-27

- Add archetype engine (archetype.h/c) — 6 TYLER archetypes with natal/Hz/prefix; --archetype NAME steering, --classify perplexity scoring, --list-archetypes; Apple #4385


## 2026-06-23
- S26-05: entropy validation — base H_mean=4.4877 vs emily-ft H_mean=4.6602 (+0.17 nats); NORTHSTAR milestone 3 updated; target ≥0.5 nats requires Colab T4 full fine-tune

- feat: local LoRA fine-tune round 2026-06-23 — 20 steps, loss 5.336→4.724, 1036 corpus records, checkpoint-emily-ft saved


## 2026-06-18
- feat(game-ai): S39-03 --game-replays flag in prime_directive_dataset.py + build_game_corpus.py standalone aggregator; replay NDJSON → instruction pairs {prompt:state, completion:action} (Apple #1405)
- feat(game-ai): S39-01 game state serializer + action decoder — serialize_snapshot() PacketSnapshot→token string, decode_action() token string→UserCmd dict, encode_action() for replay logging; 14 unit tests pass (Apple #1401)

- docs(game-ai): GAME_AI_NORTHSTAR.md — GPT-2 as game policy network; 6-milestone track (state serializer → replay logger → fine-tune → GPT-2 policy → self-play → BedWars MOBA AI); NORTHSTAR.md Track 2 added (Apple #1264)


## 2026-06-16

- Git LFS: emily-ft.bin + model.bin + checkpoint-emily-ft/model.safetensors now tracked via LFS and pushed (Apple #589)


## 2026-06-15
- config/broker-routes.json — FatBaby broker routes for GPT-2 proxy on :8679 (tenant emily-prime, upstream :8088)

- S26: GPT-2 inference API (:8088) + --prompt C CLI + tokenizer binary builder


## 2026-06-14 (session 7 — entropy fix + smoke test)
- fix(gpt2.c): token_entropy_from_logits() — log-sum-exp in double before softmax; base H_mean=4.49 nats, H_max=8.62 nats (was 0.0 due to float32 underflow of non-max probs). Apple #519.
- feat(train_local): 20-step CPU smoke test validates end-to-end pipeline: build-dataset → train → convert → entropy. Loss 6.40→5.73. Fine-tuned H_mean=4.66 vs base 4.49 (+0.17 nats). Apple #520. Checkpoint: checkpoint-emily-ft/. Binary: weights/emily-ft.bin.

## 2026-06-14 (session 6 — colab preset + deduplication)
- feat(corpus): --colab preset in prime_directive_dataset.py — Emily operational text only (no SEC/press/TYLER raw crawl), deduped, max 1500 records; produces 466 records / 154k tokens / 2.2 min T4 (was 7184 records / 32M tokens / 469 min)
- feat(corpus): hash-based deduplication (--dedupe / --no-dedupe; default on)
- feat(corpus): stratified sampling cap (--max-records N; preserves source distribution)
- feat(corpus): --no-sec, --no-press-releases, --no-tyler exclusion flags
- fix(corpus): PRRJECT_FATBABY and TYLER no longer auto-discovered — must be explicit via --fatbaby-root / --tyler-root (prevents corpus explosion)
- feat(emily.cli): emily train build-dataset defaults to --colab preset; --no-colab to get full corpus
- docs(northstar): Milestone 1.5 updated with accurate corpus numbers

## 2026-06-14 (session 5 — C warnings + NORTHSTAR + emily train stats)
- add SEC filing, press release, and TYLER corpus sources to prime_directive_dataset.py; auto-discover PRRJECT_FATBABY and TYLER sibling repos; --fatbaby-root, --tyler-root, --max-sec-docs, --max-pr-docs flags

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
