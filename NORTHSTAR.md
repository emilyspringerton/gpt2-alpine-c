# gpt2-alpine-c — Northstar

*Last updated: 2026-06-14*

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
| `scripts/prime_directive_dataset.py` | Build training JSONL from Emily golden docs + prime directive |
| `scripts/drive_sync.py` | Upload/download training artifacts via IDUNA Drive API |
| `notebooks/gpt2_finetune_colab.ipynb` | Colab notebook: HF Trainer fine-tune + save checkpoint |
| `weights/model.bin` | GPT-2 small weights (binary, downloaded separately) |
| `Makefile` | `make` → builds gpt2_run |
| `Dockerfile` | Alpine/musl build environment |

---

## Training Dataset: The Prime Directive Corpus

Emily's training corpus is built from:
1. **Golden docs** — all Tier 1 + Tier 2 docs from `EMILY/context/golden-docs-index.md`
2. **Prime directive** — `EMILY/docs/emily-prime-directive-data-collection.md`
3. **RSI task history** — `EMILY/var/training-data/*.jsonl` (collected passively)
4. **Apple log** — recent Apple bodies from IDUNA (structured domain text)
5. **BACKLOG + DONE** — backlog items as instruction-following pairs

Format: JSONL, each line `{"text": "..."}` for language modeling or
`{"prompt": "...", "completion": "..."}` for instruction fine-tuning.

The corpus teaches the model:
- EINHORN_INDUSTRIAL domain vocabulary (Apples, RSI, FatBaby signals, HEIMDAL)
- Emily's output format (BACKLOG items, CHANGELOG entries, Apple bodies)
- The planning→implementation→audit pattern

---

## Milestones

| Milestone | Status | Description |
|-----------|--------|-------------|
| 0: C inference engine | DONE | GPT-2 small forward pass, tokenizer, entropy stats |
| 1: Training tooling | DONE | Dataset builder, Drive sync, Colab notebook, FT converter |
| 2: Prime directive fine-tune | IN PROGRESS | First experimental fine-tune on Colab |
| 3: Checkpoint validation | NOT STARTED | Run fine-tuned model through gpt2_run; validate entropy |
| 4: Emily domain vocabulary | NOT STARTED | Measure perplexity on Emily operational text |
| 5: Emily Prime deployment | NOT STARTED | Replace haiku calls with local model for routine classification |

---

## Entropy Source (RSI Loop Integration)

```bash
./gpt2_run weights/model.bin --entropy-stats
# → entropy_mean_nats: 3.42, entropy_max_nats: 7.81
```

Emily's RSI loop uses the model's per-token entropy as a signal for creative
entropy injection (TYLER entropy phase in rsi-loop.sh). High entropy = more
creative; low entropy = more deterministic.

---

## Related Repos

| Repo | Relationship |
|------|-------------|
| `EMILY` | Training data source; prime directive + golden docs + RSI task history |
| `IDUNA` | Drive API for artifact sync (upload datasets, download checkpoints) |
| `PRRJECT_FATBABY` | Domain text source (governance signals, Apple bodies) |
