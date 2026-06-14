#!/usr/bin/env python3
"""
Evaluate perplexity of a GPT-2 model on the Emily Prime corpus.

Compares base GPT-2 vs fine-tuned Emily model on the same evaluation set.
Used for S26-05: entropy validation after Colab fine-tune.

Usage:
    # Base model perplexity (before fine-tuning)
    python3 scripts/eval_perplexity.py \
        --corpus /tmp/emily-corpus.jsonl \
        --model gpt2

    # Fine-tuned checkpoint
    python3 scripts/eval_perplexity.py \
        --corpus /tmp/emily-corpus.jsonl \
        --checkpoint ./checkpoint-final

    # Compare base vs fine-tuned
    python3 scripts/eval_perplexity.py \
        --corpus /tmp/emily-corpus.jsonl \
        --model gpt2 \
        --checkpoint ./checkpoint-final \
        --compare

    # Per-source breakdown (requires _source field in corpus)
    python3 scripts/eval_perplexity.py \
        --corpus /tmp/emily-corpus.jsonl \
        --model gpt2 \
        --per-source
"""

import argparse
import json
import math
import sys
from pathlib import Path
from collections import defaultdict


def load_corpus(jsonl_path: Path) -> list[dict]:
    records = []
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                records.append(obj)
            except json.JSONDecodeError:
                continue
    return records


def record_text(rec: dict) -> str:
    if "text" in rec:
        return rec["text"]
    if "prompt" in rec and "completion" in rec:
        return f"{rec['prompt']}\n\n### Response:\n{rec['completion']}"
    return ""


def eval_model(model, tokenizer, texts: list[str], max_length: int = 512,
               batch_size: int = 8, device: str = "cpu") -> dict:
    """Evaluate mean perplexity over a list of texts."""
    import torch

    model.eval()
    model.to(device)

    total_loss = 0.0
    total_tokens = 0
    n_examples = 0

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=True,
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            labels = input_ids.clone()
            # Mask padding tokens in loss
            labels[attention_mask == 0] = -100

            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            # out.loss is mean over non-ignored tokens in the batch
            n_tokens = (labels != -100).sum().item()
            total_loss += out.loss.item() * n_tokens
            total_tokens += n_tokens
            n_examples += len(batch)

            if (i // batch_size) % max(1, (len(texts) // batch_size // 5)) == 0:
                done = min(i + batch_size, len(texts))
                print(f"  {done}/{len(texts)} examples...", end="\r", flush=True)

    print(f"  {len(texts)}/{len(texts)} done.     ")
    mean_loss = total_loss / max(total_tokens, 1)
    perplexity = math.exp(mean_loss)
    return {
        "loss": mean_loss,
        "perplexity": perplexity,
        "n_examples": n_examples,
        "n_tokens": total_tokens,
    }


def load_hf_model(model_name_or_path: str):
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    tokenizer = GPT2TokenizerFast.from_pretrained(model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(model_name_or_path)
    return model, tokenizer


def print_results(label: str, results: dict) -> None:
    print(f"\n{'='*50}")
    print(f"  Model:       {label}")
    print(f"  Examples:    {results['n_examples']}")
    print(f"  Tokens:      {results['n_tokens']}")
    print(f"  Loss:        {results['loss']:.4f}")
    print(f"  Perplexity:  {results['perplexity']:.2f}")
    print(f"{'='*50}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate GPT-2 perplexity on Emily corpus")
    parser.add_argument("--corpus", required=True, help="Path to JSONL corpus")
    parser.add_argument("--model", default="gpt2",
                        help="HuggingFace model name or path (default: gpt2)")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to fine-tuned HuggingFace checkpoint directory")
    parser.add_argument("--compare", action="store_true",
                        help="Run both --model and --checkpoint and compare")
    parser.add_argument("--per-source", action="store_true",
                        help="Show per-source perplexity breakdown (requires _source in corpus)")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-frac", type=float, default=0.1,
                        help="Fraction of corpus to use for eval (default: 0.1 = 10%%)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--memory-efficient", action="store_true",
                        help="Use CPU-safe settings (batch=1, max_length=64) for low-RAM machines")
    args = parser.parse_args()

    if args.memory_efficient:
        args.batch_size = 1
        args.max_length = 64
        args.eval_frac = min(args.eval_frac, 0.05)

    try:
        import torch
    except ImportError:
        print("ERROR: torch not available. Install: pip install torch transformers", file=sys.stderr)
        sys.exit(1)
    try:
        import transformers  # noqa: F401
    except ImportError:
        print("ERROR: transformers not available. Install: pip install transformers", file=sys.stderr)
        sys.exit(1)

    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        print(f"ERROR: corpus not found: {corpus_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading corpus: {corpus_path}")
    records = load_corpus(corpus_path)
    print(f"  {len(records)} records loaded")

    # Deterministic eval split
    import random
    rng = random.Random(args.seed)
    rng.shuffle(records)
    n_eval = max(1, int(len(records) * args.eval_frac))
    eval_records = records[:n_eval]
    print(f"  Eval set: {n_eval} records ({args.eval_frac*100:.0f}% of corpus)")

    eval_texts = [record_text(r) for r in eval_records]
    eval_texts = [t for t in eval_texts if t.strip()]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    # --- Base model ---
    print(f"\nLoading base model: {args.model}")
    base_model, tokenizer = load_hf_model(args.model)
    print("  Evaluating...")
    base_results = eval_model(base_model, tokenizer, eval_texts,
                               args.max_length, args.batch_size, device)
    print_results(args.model, base_results)

    # --- Fine-tuned checkpoint ---
    ft_results = None
    if args.checkpoint or args.compare:
        ckpt = args.checkpoint
        if not ckpt:
            print("ERROR: --compare requires --checkpoint", file=sys.stderr)
            sys.exit(1)
        ckpt_path = Path(ckpt)
        if not ckpt_path.exists():
            print(f"ERROR: checkpoint not found: {ckpt_path}", file=sys.stderr)
            sys.exit(1)
        print(f"\nLoading fine-tuned checkpoint: {ckpt_path}")
        ft_model, ft_tokenizer = load_hf_model(str(ckpt_path))
        print("  Evaluating...")
        ft_results = eval_model(ft_model, ft_tokenizer, eval_texts,
                                 args.max_length, args.batch_size, device)
        print_results(str(ckpt_path), ft_results)

    # --- Comparison ---
    if ft_results:
        delta_ppl = ft_results["perplexity"] - base_results["perplexity"]
        pct = (delta_ppl / base_results["perplexity"]) * 100
        direction = "lower (better domain fit)" if delta_ppl < 0 else "higher (domain shift?)"
        print(f"\n  Δ perplexity: {delta_ppl:+.2f} ({pct:+.1f}%) — {direction}")
        print(f"  GPT-2 base on WebText: ~29 (reference)")

    # --- Per-source breakdown ---
    if args.per_source:
        sources = defaultdict(list)
        for rec, text in zip(eval_records, eval_texts):
            src = rec.get("_source", "unknown")
            if text.strip():
                sources[src].append(text)

        print("\n  Per-source perplexity (base model):")
        for src, src_texts in sorted(sources.items()):
            if not src_texts:
                continue
            r = eval_model(base_model, tokenizer, src_texts,
                           args.max_length, args.batch_size, device)
            print(f"    {src:<30}  ppl={r['perplexity']:6.2f}  n={r['n_examples']}")

    print("\nDone.")


if __name__ == "__main__":
    main()
