# gpt2-alpine-c — Agent Instructions

C inference engine + GPT-2 fine-tuning pipeline for Emily Prime.

## What this repo is

- **C inference engine**: minimal GPT-2 small forward pass, Alpine/musl, zero runtime dependencies
- **Training tooling**: builds JSONL corpus from Emily golden docs, syncs via IDUNA Drive API, fine-tunes on Colab

## Build

```bash
make                   # → ./gpt2_run
make docker            # builds Alpine container
./gpt2_run weights/model.bin --entropy-stats   # test entropy output
```

## Training Pipeline (sequential steps)

```bash
# 1. Build training corpus from Emily golden docs
python3 scripts/prime_directive_dataset.py \
  --emily-root /home/fatbaby/EMILY \
  --output /tmp/emily-corpus.jsonl

# 2. Upload corpus to Drive via IDUNA
IDUNA_BASE_URL=http://localhost:8080 \
IDUNA_AGENT_NAME=EMILY-TRAINING \
IDUNA_AGENT_SECRET=<from var/agent-secrets.env> \
python3 scripts/drive_sync.py --upload /tmp/emily-corpus.jsonl

# 3. Run notebook on Colab (see notebooks/gpt2_finetune_colab.ipynb)

# 4. Download fine-tuned checkpoint
python3 scripts/drive_sync.py --download --pattern "checkpoint-*.tar.gz"

# 5. Convert checkpoint to C binary
python3 scripts/convert_ft_checkpoint.py \
  --checkpoint ./checkpoint-final \
  --output ./weights/emily-ft.bin
```

## Env Vars

```
IDUNA_BASE_URL      — IDUNA server URL (default: http://localhost:8080)
IDUNA_AGENT_NAME    — agent name (EMILY-TRAINING)
IDUNA_AGENT_SECRET  — agent secret (from IDUNA var/agent-secrets.env)
```

## Key Files

| File | Purpose |
|------|---------|
| `scripts/prime_directive_dataset.py` | Build JSONL training corpus from Emily golden docs |
| `scripts/drive_sync.py` | Upload/download artifacts via IDUNA Drive API |
| `scripts/convert_ft_checkpoint.py` | HuggingFace checkpoint → C binary weights |
| `notebooks/gpt2_finetune_colab.ipynb` | Colab fine-tuning notebook (run on GPU) |
| `convert_checkpoint.py` | Existing: HF gpt2 base → binary (not fine-tuned) |
| `src/gpt2.c` | Transformer forward pass |
| `src/main.c` | CLI: generate + entropy stats |
| `weights/model.bin` | GPT-2 base weights (binary) |

## CHANGELOG Protocol

After any change, append a dated bullet to CHANGELOG.md:
```
## YYYY-MM-DD
- <what changed>
```

## Apple Protocol

After completing backlog items:
```bash
emily apples post -t completion "<title>" "<body with details>"
```

## Golden Doc Registration

After creating docs, register in EMILY/context/golden-docs-index.md:
```
| GPT2-NORTH | gpt2-alpine-c/NORTHSTAR.md | 2 | 0 | GPT-2 fine-tuning pipeline for Emily Prime |
```

## Related

- `EMILY` — golden docs, prime directive, RSI training data sources
- `IDUNA` — Drive API at `/api/v1/drive/*`, EMILY-TRAINING agent
- `EMILY/context/golden-docs-index.md` — register new docs here
