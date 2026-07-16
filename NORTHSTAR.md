# gpt2-alpine-c — Northstar

*Last updated: 2026-07-16*

---

## Three-Sentence Version

gpt2-alpine-c is EINHORN_INDUSTRIAL's minimal C implementation of the GPT-2 transformer,
designed to run on Alpine Linux / musl libc with zero Python/CUDA dependency at inference time.
It is the on-device inference engine for Emily Prime's self-hosted models — the end state of the
training pipeline that runs on Google Colab and syncs via IDUNA's Drive API.
The repo also contains all tooling needed to build training datasets from Emily's golden docs
and fine-tune via HuggingFace Trainer on Colab, then convert checkpoints back to C binary format.

---

## Architecture

```
Training Pipeline (Colab):
  Emily golden docs + prime directive
    → scripts/prime_directive_dataset.py (build JSONL corpus)
    → scripts/drive_sync.py (upload to Drive via IDUNA API)
    → notebooks/gpt2_finetune_colab.ipynb (HuggingFace fine-tune on Colab GPU)
    → scripts/drive_sync.py --download (fetch checkpoint)
    → scripts/convert_ft_checkpoint.py (HuggingFace → C binary)
    → weights/model.bin (deployed to C engine)

Inference Engine (C, Alpine/musl):
  src/gpt2.c          — transformer forward pass (12L/12H/768D GPT-2 small)
  src/checkpoint_loader.c — loads weights/model.bin flat binary
  src/tokenizer.c     — BPE tokenizer (tokenizer.json)
  src/main.c          — CLI: generate N tokens, optionally print entropy stats
  → gpt2_run <weights.bin> [--tokens N] [--entropy] [--entropy-stats]
```

---

## Key Files

| Path | What it is |
|------|-----------|
| `src/gpt2.c` | Full GPT-2 transformer (12 layers, attention, MLP, layer norm) |
| `src/checkpoint_loader.c` | Loads flat binary weights written by convert_checkpoint.py |
| `src/tokenizer.c` | BPE tokenizer reading tokenizer.json |
| `src/main.c` | CLI: generate tokens + entropy stats (used as RSI entropy source) |
| `convert_checkpoint.py` | Convert HuggingFace gpt2 checkpoint → flat binary (inference) |
| `scripts/convert_ft_checkpoint.py` | Convert fine-tuned HF checkpoint → flat binary (training output) |
| `scripts/prime_directive_dataset.py` | Build JSONL from golden docs + prime directive + Apple log + instruct pairs |
| `scripts/corpus_stats.py` | Pre-upload quality check (token count, source breakdown, Colab estimate) |
| `scripts/eval_perplexity.py` | Perplexity eval base vs fine-tuned; `--memory-efficient` for CPU |
| `scripts/drive_sync.py` | Upload/download training artifacts via IDUNA Drive API |
| `notebooks/gpt2_finetune_colab.ipynb` | Colab notebook: HF Trainer fine-tune + save checkpoint |
| `var/perplexity-baseline.json` | Base GPT-2 PPL on Emily corpus: 116.76 (target post-FT: <60) |
| `weights/model.bin` | GPT-2 small weights (binary, not in git) |
| `Makefile` | `make` → builds gpt2_run; `make test` → compile + 28 Python unit tests |
| `Dockerfile` | Alpine/musl build environment |

---

## Training Dataset: The Prime Directive Corpus

Emily's training corpus is built from:
1. **Golden docs** — all Tier 1 + Tier 2 docs from `EMILY/context/golden-docs-index.md` (39 sources, ~321 chunks)
2. **Prime directive** — `EMILY/docs/emily-prime-directive-data-collection.md` (~29 chunks)
3. **RSI task history** — `EMILY/var/training-data/*.jsonl` (collected passively)
4. **Apple log** — APPLES git repo JSON files (auto-discovered as sibling of EMILY; `--apples-dir`)
5. **Prime directive instruct pairs** — 24 Q&A pairs: 8 hardcoded identity/protocol + section-level (`--mode instruct`)
6. **BACKLOG done items** — 104 instruction pairs from `- [x]` entries (`--mode instruct`)

Colab corpus (--colab preset, Emily operational only): ~466 records / ~154k tokens / ~2.2 min T4 training
Full corpus (--no-colab, all sources): ~7184 records / ~32M tokens / ~469 min T4 training (not for Colab)

Format: JSONL, each line `{"text": "..."}` for language modeling or
`{"prompt": "...", "completion": "..."}` for instruction fine-tuning.

The corpus teaches the model:
- EINHORN_INDUSTRIAL domain vocabulary (Apples, RSI, FatBaby signals, HEIMDAL)
- Emily's output format (BACKLOG items, CHANGELOG entries, Apple bodies)
- The planning→implementation→audit pattern

---

## Corpus Evolution — Bridge to FABLE E0 (S146)

*Added 2026-07-16. Backlog: EMILY/BACKLOG.md SECTION 146. Governing spec: HQ-SPEC-AI-103 (FABLE), §4a data engine, §8 build order.*

FABLE (BACKLOG SECTION 145) reuses this repo's validated fine-tune pipeline for its E0 rung:
fine-tune published GPT-2 weights on EPS headlines, stand the whole stack up end-to-end. But
FABLE's data requirements are stricter than what `prime_directive_dataset.py` produces today:
per-record provenance tracing to a reality-rooted oracle (the filed 8-K), immutable
content-addressed snapshots, and contamination tombstoning of eval records by hash. This section
is the path from the current corpus builder to both purposes without breaking either.

### Two tracks, one builder

**Track A — general Emily corpus (unchanged).** The existing behavior — all Tier 1+2 golden docs,
prime directive, BACKLOG, Apples, SEC filings, press releases, TYLER, game replays, dedup,
stratified sampling — stays exactly as is. This corpus's value *is* its breadth: it teaches
Emily-domain vocabulary and serves as the RSI entropy source (S26). The six freshly-registered
HQ-SPEC docs (098–103, Tier 2) flow in automatically on the next build, which is correct: they
are self-description, and self-description is this track's purpose. The perplexity baseline
(116.76) predates them, so the next rebuild refreshes it.

**Track B — FABLE E0 snapshot (new, narrow).** A separate build preset (`--fable-eps`, analogous
to `--colab`) that emits EPS-headline records *only*: no golden docs, no TYLER, no Apples, no
chunked SEC filings. Spec documents describing FABLE must never be FABLE's task training data,
and they are useless for headline generation anyway. FABLE's thesis is auditability, not breadth
— mixing the tracks would break both. Two corpora, one builder, distinguished by preset.

### Where fabledata lives — the call

**`fabledata` is a separate Go component (per HQ-SPEC-AI-103 §4a), not this repo. This repo
implements the snapshot-manifest contract first, in Python, so E0 doesn't wait for the Go
rewrite.** Rationale:

1. The spec is explicit: fabledata is Go with house patterns (NDJSON readers, Iduna-schema'd
   config). It belongs with the other Go services, not inside a C inference repo.
2. E0's whole point is speed through reuse. This repo's extraction paths are already validated
   (S26-02/05); blocking E0 on a Go rewrite inverts the build order in §8.
3. The durable interface is not the code — it's the **snapshot manifest format** (below). This
   repo defines and produces manifest v1 for E0; when Go `fabledata` lands (S145-01), it
   inherits the format and takes over as producer. This repo's training pipeline then becomes a
   *consumer* of fabledata snapshots — training never reads live stores either way.

### What the EPS source actually is

`sec_filings_to_records()` is **not** EPS-headline-scoped and is not the starting point for
Track B. It reads generic secwatch `source_document_persisted` events, chunks `cleaned_text` at
1500 chars, and carries no oracle linkage — chunking destroys record identity, which provenance
requires. It stays as-is for Track A.

The real EPS-headline corpus is `PRRJECT_FATBABY/var/eps/`:
- `articles.ndjson` — one record per EPS headline: `source_identity` (e.g. `pr:302803511`),
  ticker, headline, dek, body, period, `eps_value`, `is_gaap`, `publish_at`.
- `oracle.ndjson` — one case per article: `case_id`, `extracted_eps`, `filed_eps`, `verdict`
  (`pending` until eps-reconciler grades it against the filed 8-K).

Track B needs a **new function** `eps_headlines_to_records()`: read both files, join article to
oracle case by `source_identity`, emit one record per article (never chunked), and attach
provenance. Only `verdict: confirmed` cases are FABLE-eligible — the label must be reality-rooted
before the record trains anything (Löbian rule 1). `pending`/contradicted cases are excluded and
counted. The store holds ~1 record today; volume accrues from eps-processor as earnings seasons
pass — the pipeline shape is what gets built now.

Per-record provenance fields (matching §4a): `source_event_hash` (sha256 of the raw article
NDJSON line), `oracle` (eps-reconciler case id, plus 8-K accession when the reconciler carries
it), `label_date` (case `recorded_at`), `license_class` (`own-exhaust` — the whole store is
EINHORN's own event exhaust).

### Snapshot manifest v1

A snapshot is a pair of files under `var/snapshots/`, immutable once written:
`eps-<YYYYMMDD>-<shorthash>.jsonl` (the records) + `.manifest.json`. The manifest is
content-addressed: `snapshot_id` = sha256 over the sorted per-record hashes. Minimal v1 shape:

```json
{
  "manifest_version": 1,
  "snapshot_id": "sha256:3f9a…",
  "created_at": "2026-07-16T00:00:00Z",
  "builder": {
    "script": "scripts/prime_directive_dataset.py",
    "git_rev": "bc45295",
    "args": ["--fable-eps"]
  },
  "tombstone_list": {
    "path": "var/eval-tombstones.json",
    "sha256": "sha256:0e12…"
  },
  "record_count": 1,
  "records": [
    {
      "sha256": "sha256:a1b2…",
      "source": "eps-headlines",
      "provenance": {
        "source_event_hash": "sha256:77cd…",
        "source_identity": "pr:302803511",
        "oracle": "eps-reconciler/eps:8bd28b7b713deb01",
        "verdict": "confirmed",
        "label_date": "2026-06-17",
        "license_class": "own-exhaust"
      }
    }
  ]
}
```

That is the full v1 scope — sized to what E0 needs, not FABLE's eventual minhash-dedup/PII-scrub
filter stack (that arrives with Go fabledata). Track A output can optionally gain a manifest
later; it is not required for S26.

### Contamination tombstoning — the minimal mechanism

HQ-SPEC-AI-103 §8 step 2: the fableeval EPS suite freezes *before any training*, and its records
are excluded from every training snapshot by hash — mechanical, not procedural. The smallest
thing that satisfies this:

- **`var/eval-tombstones.json`** in this repo: a git-committed flat list of
  `{sha256, suite, frozen_at}` entries. The fableeval freeze step (S145-02) produces the hashes;
  this file is where the builder reads them.
- The carve-out happens **inside `prime_directive_dataset.py` at snapshot-build time**: after
  record generation, before dedup/write, drop any record whose content hash appears in the
  tombstone list. The manifest records the tombstone list's own hash (see above), so any
  snapshot proves which exclusion set it was built under.
- `corpus_stats.py` grows a check: zero tombstoned hashes present in the corpus, or fail — the
  "contamination audit results (must be zero findings)" metric from §7.

No service, no database, no flag soup — one file, one exclusion pass, one audit check.

### Order of work

1. **S146-01** — manifest v1 writer (`--snapshot` behavior in the builder). The contract Go
   fabledata inherits; feeds S145-01.
2. **S146-02** — `eps_headlines_to_records()` with provenance + verdict gating.
3. **S146-03** — `--fable-eps` preset producing the immutable snapshot pair; feeds S145-03 (E0
   training input).
4. **S146-04** — tombstone list + exclusion pass + manifest linkage; consumes S145-02's frozen
   suite hashes.
5. **S146-05** — corpus_stats provenance/contamination audit mode.
6. **S146-06** — Track A rebuild with HQ-SPEC 098–103 ingested; refresh perplexity baseline so
   S26-04's eventual Colab run measures against current corpus numbers.

Open per HQ-SPEC-AI-103 §9, deliberately not decided here: GPU posture, corpus mix beyond
own-exhaust for E1 pretraining, distillation stance. Track B's E0 diet is 100% own-exhaust,
which sidesteps all three for now.

---

## Milestones

| Milestone | Status | Description |
|-----------|--------|-------------|
| 0: C inference engine | DONE | GPT-2 small forward pass, tokenizer, entropy stats |
| 1: Training tooling | DONE | Dataset builder, Drive sync, Colab notebook, FT converter |
| 1.5: Corpus quality | DONE | Colab preset (466 records/2.2 min T4), deduplication, stats tool, 28 unit tests |
| 2: Prime directive fine-tune | PENDING (manual) | Run notebooks/gpt2_finetune_colab.ipynb on Colab T4 GPU with Colab corpus |
| 3: Checkpoint validation | DONE | Base: H_mean=4.4877 nats. Fine-tuned (emily-ft.bin): H_mean=4.6602 nats (+0.17). Target ≥0.5 not met — Colab T4 full fine-tune required. |
| 4: Emily domain vocabulary | NOT STARTED | Compare PPL base (116.76) vs fine-tuned on same eval set |
| 5: Emily Prime deployment | NOT STARTED | Replace haiku calls with local model for routine classification |

---

## Entropy Source (RSI Loop Integration)

```bash
./gpt2_run weights/model.bin --entropy-stats
# → entropy_mean_nats=4.4877 entropy_max_nats=8.6231 tokens=64  (base GPT-2, 2026-06-23)

./gpt2_run weights/emily-ft.bin --entropy-stats
# → entropy_mean_nats=4.6602 entropy_max_nats=8.6346 tokens=64  (emily-ft 300-step, 2026-06-23)
# delta: +0.17 nats (target ≥0.5 — full Colab T4 run required to meet target)
```

Emily's RSI loop uses the model's per-token entropy as a signal for creative
entropy injection (TYLER entropy phase in rsi-loop.sh). High entropy = more
creative; low entropy = more deterministic.

---

---

## Track 2: Game AI (GPT-2 as Policy Network)

Full spec: `docs/GAME_AI_NORTHSTAR.md`

GPT-2's token generation maps directly onto sequential game decisions: encode game state as
natural language tokens, generate action tokens, decode to game inputs. The testbed is
SHANKPIT (emily-bot as policy host); the MOBA-scale target is BedWars.

| Milestone | Status | Description |
|-----------|--------|-------------|
| 6: State serializer + action decoder | NOT STARTED | `scripts/game_state.py` — SHANKPIT snapshot → token string → UserCmd |
| 7: Replay logger in emily-bot | NOT STARTED | Log (state, action) NDJSON per tick; `scripts/build_game_corpus.py` |
| 8: Fine-tune on replay corpus | NOT STARTED | `--game-replays` flag in dataset builder; Colab fine-tune |
| 9: GPT-2 policy in emily-bot | NOT STARTED | Replace heuristic `think()` with inference server call |
| 10: Self-play loop | NOT STARTED | 4 bots play → replay data → re-fine-tune → next generation |
| 11: BedWars AI | NOT STARTED | MOBA-scale state/action vocabulary; strategic decisions (economy, objectives) |

**The "our own League of Legends" milestone is M11.** BedWars has the same strategic primitives
as a MOBA: economy, objectives, team elimination, map control. The architecture paper writes itself.

---

## Related Repos

| Repo | Relationship |
|------|-------------|
| `EMILY` | Training data source; prime directive + golden docs + RSI task history |
| `IDUNA` | Drive API for artifact sync (upload datasets, download checkpoints) |
| `PRRJECT_FATBABY` | Domain text source (governance signals, Apple bodies) |
| `SHANKPIT` | Game testbed; emily-bot hosts the GPT-2 policy; BedWars is the MOBA-scale target |
