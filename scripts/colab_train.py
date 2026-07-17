#!/usr/bin/env python3
"""
Colab fine-tuning driver — the versioned logic behind the notebook's single
bootstrap cell (notebooks/gpt2_finetune_colab.ipynb).

The notebook cell only mounts Drive and does `git clone`/`git pull` +
`python scripts/colab_train.py`. All actual training logic lives here, in
git, so a fresh run always executes the latest version without anyone
re-pasting cells into Colab. Change training behavior by editing this file
and pushing — the next "Run all" in Colab picks it up via git pull.

Config is read from environment variables (set by the notebook cell) with
defaults matching this project's established values, so it also runs
standalone for local smoke-testing (`python scripts/colab_train.py --help`).
"""

import argparse
import json
import math
import os
import subprocess
import sys
import tarfile


def pip_install():
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q",
        "transformers", "datasets", "accelerate",
    ])


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--drive-folder", default=os.environ.get(
        "DRIVE_FOLDER", "/content/drive/MyDrive/emily-training"))
    p.add_argument("--corpus-file", default=None,
                   help="default: <drive-folder>/emily-corpus.jsonl")
    p.add_argument("--output-dir", default=None,
                   help="default: <drive-folder>/checkpoint-final")
    p.add_argument("--model-name", default=os.environ.get("MODEL_NAME", "gpt2"))
    p.add_argument("--max-length", type=int, default=int(os.environ.get("MAX_LENGTH", 512)))
    p.add_argument("--batch-size", type=int, default=int(os.environ.get("BATCH_SIZE", 4)))
    p.add_argument("--grad-accum", type=int, default=int(os.environ.get("GRADIENT_ACCUMULATION", 4)))
    p.add_argument("--epochs", type=int, default=int(os.environ.get("NUM_EPOCHS", 3)))
    p.add_argument("--learning-rate", type=float, default=float(os.environ.get("LEARNING_RATE", 5e-5)))
    p.add_argument("--warmup-steps", type=int, default=int(os.environ.get("WARMUP_STEPS", 100)))
    p.add_argument("--save-steps", type=int, default=int(os.environ.get("SAVE_STEPS", 250)))
    p.add_argument("--skip-pip-install", action="store_true",
                   help="local smoke-testing only — Colab always installs")
    return p.parse_args()


def record_to_text(rec):
    if "text" in rec:
        return rec["text"]
    if "prompt" in rec and "completion" in rec:
        return f"{rec['prompt']}\n\n### Response:\n{rec['completion']}"
    return ""


def main():
    args = parse_args()
    corpus_file = args.corpus_file or os.path.join(args.drive_folder, "emily-corpus.jsonl")
    output_dir = args.output_dir or os.path.join(args.drive_folder, "checkpoint-final")

    print(f"Corpus: {corpus_file}")
    print(f"Output: {output_dir}")
    if not os.path.exists(corpus_file):
        raise FileNotFoundError(
            f"Corpus not found: {corpus_file}\n"
            "Run scripts/prime_directive_dataset.py locally and sync it to Drive "
            "with scripts/drive_sync.py --upload, or drop it in the Drive folder by hand."
        )

    if not args.skip_pip_install:
        pip_install()

    # Imports deferred until after pip_install so a fresh Colab runtime
    # doesn't need these packages pre-installed to even parse this file.
    import torch
    from datasets import Dataset
    from transformers import (
        DataCollatorForLanguageModeling, GPT2LMHeadModel, GPT2TokenizerFast,
        Trainer, TrainingArguments, pipeline,
    )

    records = []
    with open(corpus_file) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"Loaded {len(records)} records")

    texts = [record_to_text(r) for r in records if record_to_text(r).strip()]
    dataset = Dataset.from_dict({"text": texts})
    print(f"Dataset size: {len(dataset)} examples")

    tokenizer = GPT2TokenizerFast.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize(batch):
        out = tokenizer(batch["text"], truncation=True,
                         max_length=args.max_length, padding="max_length")
        out["labels"] = out["input_ids"].copy()
        return out

    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])
    tokenized.set_format("torch")
    print(f"Tokenized: {len(tokenized)} examples, max_length={args.max_length}")

    model = GPT2LMHeadModel.from_pretrained(args.model_name)
    split = tokenized.train_test_split(test_size=0.1, seed=42)
    train_ds, eval_ds = split["train"], split["test"]
    print(f"Train: {len(train_ds)}, Eval: {len(eval_ds)}")

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        per_device_eval_batch_size=args.batch_size,
        warmup_steps=args.warmup_steps,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        logging_dir="/content/logs",
        logging_steps=50,
        evaluation_strategy="steps",
        eval_steps=args.save_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=train_ds,
                       eval_dataset=eval_ds, data_collator=data_collator)

    print("Starting fine-tuning...")
    trainer.train()
    print("Training complete.")

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved to {output_dir}")

    tar_path = output_dir + ".tar.gz"
    with tarfile.open(tar_path, "w:gz") as tf:
        tf.add(output_dir, arcname="checkpoint-final")
    size_mb = os.path.getsize(tar_path) / 1024 / 1024
    print(f"Archived: {tar_path} ({size_mb:.1f} MB)")

    eval_results = trainer.evaluate()
    perplexity = math.exp(eval_results["eval_loss"])
    print(f"Eval loss:   {eval_results['eval_loss']:.4f}")
    print(f"Perplexity:  {perplexity:.2f}")
    print("GPT-2 small base perplexity on WebText: ~29 (lower = more adapted)")

    if torch.cuda.is_available():
        gen = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)
        prompts = [
            "Emily Prime is the chief of staff for EINHORN_INDUSTRIAL.",
            "The RSI loop begins when obs-watcher detects",
            "An Apple is filed after every backlog completion:",
        ]
        print("\n--- Generation samples ---")
        for prompt in prompts:
            out = gen(prompt, max_new_tokens=60, do_sample=True, temperature=0.8, top_p=0.95)
            print(f"PROMPT: {prompt}")
            print(f"OUTPUT: {out[0]['generated_text']}\n")

    print("=" * 60)
    print(f"DONE. Checkpoint: {tar_path}")
    print("Next: download from Drive (or `scripts/drive_sync.py --download "
          "--pattern 'checkpoint*.tar.gz'`), then "
          "`python3 scripts/convert_ft_checkpoint.py --checkpoint <tar> --output weights/emily-ft.bin`")
    print("=" * 60)


if __name__ == "__main__":
    main()
