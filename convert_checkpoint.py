#!/usr/bin/env python3
"""
Convert an OpenAI-format GPT-2 checkpoint (via HuggingFace) to a flat binary
weight file for the C inference engine.

Weight write order (float32, all row-major in C convention):
  wte  (V × D)
  wpe  (T × D)
  [for each layer l in 0..11]:
    ln_1_gamma (D), ln_1_beta (D)
    c_attn_w   (3D × D)  ← Conv1D weight transposed
    c_attn_b   (3D)
    c_proj_w   (D × D)   ← Conv1D weight transposed
    c_proj_b   (D)
    ln_2_gamma (D), ln_2_beta (D)
    mlp_fc_w   (4D × D)  ← Conv1D weight transposed
    mlp_fc_b   (4D)
    mlp_proj_w (D × 4D)  ← Conv1D weight transposed
    mlp_proj_b (D)
  ln_f_gamma (D), ln_f_beta (D)

GPT-2's Conv1D stores weight as (in_features, out_features), i.e., transposed
relative to nn.Linear. We transpose on export so the C engine can use standard
row-major matmul: y = W * x + b, W shape (out, in).
"""
import sys
import numpy as np


def tensor_to_np(t):
    return t.detach().cpu().float().numpy()


def write_f32(f, arr):
    f.write(arr.astype(np.float32).flatten().tobytes())


def main(model_name: str, output_bin: str):
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()
    sd = model.state_dict()

    def get(key):
        if key in sd:
            return tensor_to_np(sd[key])
        raise KeyError(f"Missing weight key: {key}")

    n_layer = model.config.n_layer
    print(f"Model: {model_name}  layers={n_layer}  "
          f"embd={model.config.n_embd}  vocab={model.config.vocab_size}")

    with open(output_bin, "wb") as f:
        # Global embeddings
        write_f32(f, get("transformer.wte.weight"))   # (V, D)
        write_f32(f, get("transformer.wpe.weight"))   # (T, D)

        for l in range(n_layer):
            p = f"transformer.h.{l}."

            # Layer norm 1
            write_f32(f, get(p + "ln_1.weight"))   # gamma (D,)
            write_f32(f, get(p + "ln_1.bias"))     # beta  (D,)

            # Attention: Conv1D weights are (in, out) → transpose to (out, in)
            write_f32(f, get(p + "attn.c_attn.weight").T)  # (3D, D)
            write_f32(f, get(p + "attn.c_attn.bias"))      # (3D,)
            write_f32(f, get(p + "attn.c_proj.weight").T)  # (D, D)
            write_f32(f, get(p + "attn.c_proj.bias"))      # (D,)

            # Layer norm 2
            write_f32(f, get(p + "ln_2.weight"))
            write_f32(f, get(p + "ln_2.bias"))

            # MLP: same Conv1D transpose convention
            write_f32(f, get(p + "mlp.c_fc.weight").T)    # (4D, D)
            write_f32(f, get(p + "mlp.c_fc.bias"))        # (4D,)
            write_f32(f, get(p + "mlp.c_proj.weight").T)  # (D, 4D)
            write_f32(f, get(p + "mlp.c_proj.bias"))      # (D,)

        # Final layer norm
        write_f32(f, get("transformer.ln_f.weight"))
        write_f32(f, get("transformer.ln_f.bias"))

    size_mb = __import__("os").path.getsize(output_bin) / 1024 / 1024
    print(f"Wrote {output_bin} ({size_mb:.1f} MB)")

    # Save tokenizer files alongside the binary
    tok = GPT2TokenizerFast.from_pretrained(model_name)
    tok.save_pretrained(".")
    print("Wrote tokenizer files (vocab.json, merges.txt) in current directory.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_checkpoint.py gpt2 weights/model.bin")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
