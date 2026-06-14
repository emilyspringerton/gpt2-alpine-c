#!/usr/bin/env python3
"""
Convert a HuggingFace fine-tuned GPT-2 checkpoint to C binary format.

Extends convert_checkpoint.py to handle fine-tuned checkpoints (from
gpt2_finetune_colab.ipynb or any HuggingFace Trainer output directory)
and produce the same flat binary format the C inference engine expects.

The binary layout is identical to convert_checkpoint.py — the C engine
treats base and fine-tuned weights identically.

Usage:
    python3 scripts/convert_ft_checkpoint.py \
        --checkpoint ./checkpoint-final \
        --output ./weights/emily-ft.bin \
        [--tokenizer gpt2]

    # Or from a .tar.gz downloaded from Drive:
    python3 scripts/convert_ft_checkpoint.py \
        --checkpoint checkpoint-final.tar.gz \
        --output ./weights/emily-ft.bin
"""

import argparse
import os
import sys
import tarfile
import tempfile
from pathlib import Path

import numpy as np


def tensor_to_np(t):
    return t.detach().cpu().float().numpy()


def write_f32(f, arr):
    f.write(arr.astype(np.float32).flatten().tobytes())


def load_model(checkpoint_path: Path):
    from transformers import GPT2LMHeadModel
    return GPT2LMHeadModel.from_pretrained(str(checkpoint_path))


def convert(checkpoint_path: Path, output_bin: Path) -> None:
    print(f"Loading checkpoint: {checkpoint_path}")
    model = load_model(checkpoint_path)
    model.eval()
    sd = model.state_dict()

    def get(key):
        if key in sd:
            return tensor_to_np(sd[key])
        raise KeyError(f"Missing weight key: {key}")

    n_layer = model.config.n_layer
    n_embd = model.config.n_embd
    vocab_size = model.config.vocab_size
    print(f"  layers={n_layer}  embd={n_embd}  vocab={vocab_size}")

    output_bin.parent.mkdir(parents=True, exist_ok=True)
    with output_bin.open("wb") as f:
        write_f32(f, get("transformer.wte.weight"))   # (V, D)
        write_f32(f, get("transformer.wpe.weight"))   # (T, D)

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

    size_mb = output_bin.stat().st_size / 1024 / 1024
    print(f"Wrote {output_bin} ({size_mb:.1f} MB)")


def extract_tar(tar_path: Path, dest: Path) -> Path:
    print(f"Extracting {tar_path}...")
    with tarfile.open(tar_path) as tf:
        tf.extractall(dest)
    # Find the checkpoint directory inside
    candidates = sorted(dest.iterdir())
    for c in candidates:
        if c.is_dir() and (c / "config.json").exists():
            return c
    # Maybe extracted flat
    if (dest / "config.json").exists():
        return dest
    raise FileNotFoundError(f"Could not find config.json in extracted {tar_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert HuggingFace fine-tuned GPT-2 checkpoint to C binary"
    )
    parser.add_argument("--checkpoint", required=True,
                        help="Path to HuggingFace checkpoint dir or .tar.gz")
    parser.add_argument("--output", default="weights/emily-ft.bin",
                        help="Output binary path (default: weights/emily-ft.bin)")
    parser.add_argument("--tokenizer", default=None,
                        help="Tokenizer source (default: same as checkpoint)")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    output_bin = Path(args.output)

    if not checkpoint_path.exists():
        print(f"ERROR: checkpoint not found: {checkpoint_path}", file=sys.stderr)
        sys.exit(1)

    # Handle .tar.gz input
    if checkpoint_path.name.endswith(".tar.gz") or checkpoint_path.name.endswith(".tgz"):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            checkpoint_path = extract_tar(checkpoint_path, tmp_path)
            convert(checkpoint_path, output_bin)
    else:
        convert(checkpoint_path, output_bin)

    # Save tokenizer files if requested or if not already present
    tok_source = args.tokenizer or str(checkpoint_path)
    tok_dir = output_bin.parent
    if not (tok_dir / "tokenizer.json").exists():
        try:
            from transformers import GPT2TokenizerFast
            tok = GPT2TokenizerFast.from_pretrained(tok_source)
            tok.save_pretrained(str(tok_dir))
            print(f"Wrote tokenizer files to {tok_dir}/")
        except Exception as e:
            print(f"WARNING: could not save tokenizer: {e}", file=sys.stderr)

    print("Done. Test with:")
    print(f"  ./gpt2_run {output_bin} --entropy-stats")


if __name__ == "__main__":
    main()
