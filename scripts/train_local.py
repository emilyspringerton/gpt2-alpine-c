#!/usr/bin/env python3
"""
Local LoRA fine-tuning for Emily Prime GPT-2.

Designed for memory-constrained CPU (≤550 MB free RAM).

Root causes of previous timeouts:
  1. Full GPT-2 float32 (523 MB) + Adam optimizer states (~1 GB) → OOM
  2. `datasets` library missing → ImportError mid-run (silent crash)
  3. No HF offline gate → any DNS probe stalls indefinitely

Fixes applied here:
  1. bfloat16 base (~261 MB) + LoRA-only optimizer (~14 MB) → fits in RAM
  2. Reads JSONL directly — no `datasets` library used
  3. TRANSFORMERS_OFFLINE=1 set before any HF import → zero network calls

Usage:
  python3 scripts/train_local.py                        # 500 steps, auto-discovers corpus
  python3 scripts/train_local.py --corpus /tmp/emily-corpus.jsonl --steps 200
  python3 scripts/train_local.py --steps 50 --no-convert   # quick smoke test

Output:
  checkpoint-emily-ft/         HuggingFace checkpoint (merged LoRA)
  weights/emily-ft.bin         C binary for gpt2_run (unless --no-convert)
"""

import os
# Must be set before any transformers import — prevents all HF hub calls.
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np


# ---------------------------------------------------------------------------
# Dataset — reads JSONL directly, no datasets library
# ---------------------------------------------------------------------------

def load_corpus(jsonl_path: Path, max_records: int = 0) -> list[str]:
    texts = []
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "text" in rec:
                t = rec["text"].strip()
            elif "prompt" in rec and "completion" in rec:
                t = f"{rec['prompt']}\n\n### Response:\n{rec['completion']}".strip()
            else:
                continue
            if len(t) >= 50:
                texts.append(t)
            if max_records > 0 and len(texts) >= max_records:
                break
    return texts


class TokenDataset(torch.utils.data.Dataset):
    def __init__(self, texts: list[str], tokenizer, max_length: int):
        self.examples = []
        for text in texts:
            enc = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            ids = enc["input_ids"][0]
            if len(ids) < 8:
                continue
            self.examples.append(ids)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ids = self.examples[idx]
        return {"input_ids": ids, "labels": ids.clone()}


def collate_fn(batch):
    # Pad to the longest sequence in the batch
    max_len = max(x["input_ids"].shape[0] for x in batch)
    input_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    for i, x in enumerate(batch):
        n = x["input_ids"].shape[0]
        input_ids[i, :n] = x["input_ids"]
        labels[i, :n] = x["labels"]
    return {"input_ids": input_ids, "labels": labels}


# ---------------------------------------------------------------------------
# C binary conversion (inline — no subprocess needed)
# ---------------------------------------------------------------------------

def write_f32(f, arr):
    f.write(arr.astype(np.float32).flatten().tobytes())


def convert_to_c_binary(model, output_path: Path) -> None:
    """Write GPT-2 weights to flat binary format expected by gpt2_run."""
    model.eval()
    sd = model.state_dict()

    def get(key) -> np.ndarray:
        if key not in sd:
            raise KeyError(f"Missing key: {key}")
        return sd[key].detach().float().cpu().numpy()

    n_layer = model.config.n_layer
    print(f"  Converting {n_layer}-layer model to {output_path} ...")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("wb") as f:
        write_f32(f, get("transformer.wte.weight"))
        write_f32(f, get("transformer.wpe.weight"))
        for l in range(n_layer):
            p = f"transformer.h.{l}."
            write_f32(f, get(p + "ln_1.weight"))
            write_f32(f, get(p + "ln_1.bias"))
            write_f32(f, get(p + "attn.c_attn.weight").T)
            write_f32(f, get(p + "attn.c_attn.bias"))
            write_f32(f, get(p + "attn.c_proj.weight").T)
            write_f32(f, get(p + "attn.c_proj.bias"))
            write_f32(f, get(p + "ln_2.weight"))
            write_f32(f, get(p + "ln_2.bias"))
            write_f32(f, get(p + "mlp.c_fc.weight").T)
            write_f32(f, get(p + "mlp.c_fc.bias"))
            write_f32(f, get(p + "mlp.c_proj.weight").T)
            write_f32(f, get(p + "mlp.c_proj.bias"))
        write_f32(f, get("transformer.ln_f.weight"))
        write_f32(f, get("transformer.ln_f.bias"))

    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"  ✓ {output_path} ({size_mb:.1f} MB)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    repo_root = Path(__file__).parent.parent
    default_corpus = Path("/tmp/emily-corpus.jsonl")
    if not default_corpus.exists():
        default_corpus = repo_root.parent / "PRRJECT_FATBABY" / "var" / "emily-observations" / "latest.json"

    parser = argparse.ArgumentParser(
        description="Local LoRA fine-tuning for Emily Prime GPT-2"
    )
    parser.add_argument("--corpus", default=str(default_corpus),
                        help="Path to JSONL training corpus (default: /tmp/emily-corpus.jsonl)")
    parser.add_argument("--output-dir", default=str(repo_root / "checkpoint-emily-ft"),
                        help="HuggingFace checkpoint output dir")
    parser.add_argument("--binary-output", default=str(repo_root / "weights" / "emily-ft.bin"),
                        help="C binary output path")
    parser.add_argument("--steps", type=int, default=500,
                        help="Training steps (default: 500; use 50 for a quick smoke test)")
    parser.add_argument("--max-length", type=int, default=128,
                        help="Max token sequence length (default: 128; lower = less RAM)")
    parser.add_argument("--lora-r", type=int, default=8,
                        help="LoRA rank (default: 8; lower = fewer trainable params)")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate (default: 3e-4)")
    parser.add_argument("--grad-accum", type=int, default=8,
                        help="Gradient accumulation steps (effective batch = 1 × grad-accum)")
    parser.add_argument("--max-records", type=int, default=3000,
                        help="Max corpus records to load (default: 3000; 0=all)")
    parser.add_argument("--log-every", type=int, default=10,
                        help="Print loss every N steps (default: 10)")
    parser.add_argument("--no-convert", action="store_true",
                        help="Skip C binary conversion after training")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        print(f"ERROR: corpus not found: {corpus_path}", file=sys.stderr)
        print("  Build it first: emily train build-dataset", file=sys.stderr)
        sys.exit(1)

    print(f"◈ Emily Prime GPT-2 Local LoRA Fine-Tuning")
    print(f"  Corpus:      {corpus_path}")
    print(f"  Steps:       {args.steps}")
    print(f"  Max length:  {args.max_length}")
    print(f"  LoRA rank:   {args.lora_r}")
    print(f"  Grad accum:  {args.grad_accum} (effective batch = {args.grad_accum})")
    print(f"  Output:      {args.output_dir}")
    print(f"  OFFLINE:     HF hub calls disabled")
    print()

    # -----------------------------------------------------------------------
    # Load model in bfloat16 to halve memory (~261 MB vs 523 MB float32)
    # -----------------------------------------------------------------------
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    from peft import LoraConfig, get_peft_model, TaskType

    print("Loading tokenizer from cache ...")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token

    print("Loading GPT-2 base model in bfloat16 (~261 MB) ...")
    t0 = time.time()
    model = GPT2LMHeadModel.from_pretrained(
        "gpt2",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    print(f"  Loaded in {time.time()-t0:.1f}s")

    # LoRA: inject trainable low-rank matrices into attention projections only.
    # With r=8: ~3.5M trainable params out of 124M total → optimizer uses ~28 MB.
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        target_modules=["c_attn"],   # Q/K/V projection — the highest-signal target
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    print()

    # -----------------------------------------------------------------------
    # Build dataset
    # -----------------------------------------------------------------------
    print(f"Loading corpus (max {args.max_records} records) ...")
    texts = load_corpus(corpus_path, args.max_records)
    print(f"  {len(texts)} texts loaded")

    print("Tokenizing ...")
    t0 = time.time()
    dataset = TokenDataset(texts, tokenizer, args.max_length)
    print(f"  {len(dataset)} examples (tokenized in {time.time()-t0:.1f}s)")

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=collate_fn,
        generator=torch.Generator().manual_seed(args.seed),
    )

    # -----------------------------------------------------------------------
    # Optimizer — only LoRA parameters (much smaller state)
    # -----------------------------------------------------------------------
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)

    # Cosine LR schedule with warmup (50 steps)
    warmup = min(50, args.steps // 10)
    def lr_lambda(step):
        if step < warmup:
            return step / max(warmup, 1)
        progress = (step - warmup) / max(args.steps - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    model.train()
    step = 0
    accum_loss = 0.0
    accum_count = 0
    optimizer.zero_grad()

    print(f"Training for {args.steps} steps (log every {args.log_every}) ...")
    print(f"  Step 0/{args.steps} — starting", flush=True)

    epoch = 0
    t_start = time.time()

    while step < args.steps:
        epoch += 1
        for batch in loader:
            if step >= args.steps:
                break

            input_ids = batch["input_ids"]
            labels = batch["labels"]

            out = model(input_ids=input_ids, labels=labels)
            loss = out.loss / args.grad_accum
            loss.backward()

            accum_loss += loss.item() * args.grad_accum
            accum_count += 1

            if accum_count == args.grad_accum:
                nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                step += 1
                accum_loss_avg = accum_loss / accum_count
                accum_loss = 0.0
                accum_count = 0

                if step % args.log_every == 0 or step == 1 or step == args.steps:
                    elapsed = time.time() - t_start
                    steps_per_sec = step / max(elapsed, 0.001)
                    eta_sec = (args.steps - step) / max(steps_per_sec, 0.001)
                    lr_now = scheduler.get_last_lr()[0] * args.lr
                    print(
                        f"  Step {step:>4}/{args.steps}  "
                        f"loss={accum_loss_avg:.4f}  "
                        f"lr={lr_now:.2e}  "
                        f"eta={eta_sec/60:.1f}min",
                        flush=True,
                    )

        if step < args.steps:
            pass  # continue to next epoch

    elapsed_total = time.time() - t_start
    print(f"\n✓ Training complete in {elapsed_total/60:.1f} min ({step} steps)")

    # -----------------------------------------------------------------------
    # Save checkpoint: merge LoRA into base model, save as HF dir
    # -----------------------------------------------------------------------
    output_dir = Path(args.output_dir)
    print(f"\nMerging LoRA adapters and saving to {output_dir} ...")
    merged = model.merge_and_unload()
    merged.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"  ✓ Checkpoint saved")

    # -----------------------------------------------------------------------
    # Convert to C binary for gpt2_run
    # -----------------------------------------------------------------------
    if not args.no_convert:
        binary_path = Path(args.binary_output)
        print(f"\nConverting to C binary: {binary_path}")
        # Cast to float32 for the C engine (layer by layer in the write loop)
        convert_to_c_binary(merged, binary_path)
        print(f"\n  Test with:")
        print(f"    ./gpt2_run {binary_path} --entropy-stats")
        print(f"    ./gpt2_run {binary_path} --tokens 40")

    print(f"\n◈ Done.")
    print(f"  Checkpoint: {output_dir}")
    if not args.no_convert:
        print(f"  C binary:   {args.binary_output}")


if __name__ == "__main__":
    main()
