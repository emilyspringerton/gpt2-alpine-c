#!/usr/bin/env bash
# post_train_commit.sh — called after a training run completes.
# Commits updated model weights via LFS and pushes.
# Usage: bash scripts/post_train_commit.sh [steps] [apple_id]

set -e
export PATH="$HOME/.local/bin:$PATH"

STEPS="${1:-300}"
APPLE_ID="${2:-}"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"

cd "$REPO_DIR"

echo "→ Checking git LFS status ..."
git lfs status

echo "→ Staging updated weights ..."
git add weights/emily-ft.bin checkpoint-emily-ft/

echo "→ Checking what changed ..."
git diff --cached --stat

LABEL=""
[ -n "$APPLE_ID" ] && LABEL=" (Apple #${APPLE_ID})"

git commit -m "feat(weights): update fine-tuned model — ${STEPS}-step local LoRA run${LABEL}

Training: LoRA rank-8 on Emily Prime corpus (578 records, bfloat16 CPU).
Overwrites weights/emily-ft.bin + checkpoint-emily-ft/model.safetensors.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>" || echo "nothing to commit"

echo "→ Pushing via LFS ..."
git push origin main

echo "✓ Done — updated fine-tune committed and pushed"
