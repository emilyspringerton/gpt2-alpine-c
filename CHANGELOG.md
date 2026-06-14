# gpt2-alpine-c Changelog

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
