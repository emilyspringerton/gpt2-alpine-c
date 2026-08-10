# gpt2-alpine-c — Agent Instructions

C inference engine + GPT-2 fine-tuning pipeline for Emily Prime.

## What this repo is

- **C inference engine**: minimal GPT-2 small forward pass, Alpine/musl, zero runtime dependencies
- **Training tooling**: builds JSONL corpus from Emily golden docs, syncs via IDUNA Drive API, fine-tunes on Colab

## Build

```bash
make                        # → ./gpt2_run (C inference binary)
make tokenizer              # → weights/tokenizer.bin (required for --prompt mode)
make docker                 # builds Alpine container

# Text prompt generation (requires tokenizer.bin):
./gpt2_run weights/emily-ft.bin --prompt "Emily Prime:" --tokens 50

# Entropy stats only (no tokenizer needed):
./gpt2_run weights/emily-ft.bin --entropy-stats
```

## Inference API Server

```bash
# Start the HTTP inference server (loads checkpoint once, keeps in memory):
python3 scripts/serve.py                     # fine-tuned model on :8088 (default)
python3 scripts/serve.py --model base        # base GPT-2
python3 scripts/serve.py --port 8089         # custom port

# Health check:
curl http://localhost:8088/health

# Generate:
curl -X POST http://localhost:8088/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Emily Prime:", "max_tokens": 100, "temperature": 0.8}'

# Via emily-agent proxy (requires emily-agent running on :8086):
curl -X POST http://localhost:8086/api/v1/gpt2/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Emily Prime:", "max_tokens": 100}'
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
| `src/main.c` | CLI: generate + entropy stats + `--prompt` text mode |
| `scripts/build_tokenizer_bin.py` | Convert tokenizer.json → tokenizer.bin (C binary format) |
| `scripts/serve.py` | HTTP inference server (:8088) — loads HF checkpoint, serves /generate |
| `weights/model.bin` | GPT-2 base weights (binary) |
| `weights/tokenizer.bin` | Binary tokenizer (gitignored; `make tokenizer` to rebuild) |

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

## Founder Real-Time Direction

Whenever the founder gives real-time direction — a new ask, a correction, a "can we also..." —
route it through `emily observe -s info "Founder real-time: <summary>"` first, even if it isn't
this repo's usual domain, then sprint-plan it into `EMILY/BACKLOG.md` (`emily backlog curate`,
scoped into a real SECTION/sub-item, not just a one-line log), and only then implement. See
`EMILY/docs/THE_EMILY_WAY.md` Principle 18 ("Pave the Cow Paths").

## Frame-Break Reframing

Founder-sourced prompting technique (REDGARDEN/NORTHSTAR.md §28, full origin in
REDGARDEN/docs2/MULTI_AGENT_RD_RESEARCH_NOTES.md §5): given a request, name the underlying
structural/systemic pattern it's one instance of — one level of abstraction up — as an added
lens during planning/triage/judgment calls. Use it to spot the general case behind a specific
ask. It augments judgment, it does not replace doing the work: direct, concrete execution of
the literal task asked for still happens every time.

## Commit Protocol (standing instruction)

Always commit and push completed work immediately — don't wait to be asked. This is the default for every repo in this monorepo.

Every commit — human-written or produced by automated code paths (git-commit helpers in emily-agent, emily.cli, IDUNA handlers, etc.) — must carry the active `emily session` fingerprint as a `session: <tag>` trailer (blank line, then the trailer). This was silently missing from several independently-implemented automated commit helpers across the monorepo until an audit on 2026-08-10 (founder, real-time: "where in the fuck is my llm session id anywhere"). If you add a new automated git-commit code path anywhere, wire in the session tag the same way — don't assume an existing helper already does it.
