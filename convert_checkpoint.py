\
    #!/usr/bin/env python3
    import sys
    import os
    import numpy as np
    import torch
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    def tensor_to_np(t):
        return t.detach().cpu().numpy().astype(np.float32)

    def main(model_name, output_bin):
        model = GPT2LMHeadModel.from_pretrained(model_name)
        sd = model.state_dict()

        # Basic expected keys; this converter writes a minimal subset.
        with open(output_bin, "wb") as f:
            # token embeddings
            wte = tensor_to_np(sd.get('transformer.wte.weight') or sd.get('model.wte.weight'))
            f.write(wte.tobytes())
            # position embeddings
            wpe = tensor_to_np(sd.get('transformer.wpe.weight') or sd.get('model.wpe.weight'))
            f.write(wpe.tobytes())

            # For each layer write a canonical subset (update keys if your checkpoint differs)
            for l in range(12):
                prefix = f'transformer.h.{l}.'
                def get(k):
                    kk = prefix + k
                    if kk in sd:
                        return tensor_to_np(sd[kk])
                    raise KeyError(f"Missing key: {kk}")

                f.write(get('ln_1.weight').tobytes())
                f.write(get('ln_1.bias').tobytes())
                f.write(get('attn.c_attn.weight').flatten().tobytes())
                f.write(get('attn.c_attn.bias').tobytes())
                f.write(get('attn.c_proj.weight').flatten().tobytes())
                f.write(get('attn.c_proj.bias').tobytes())
                f.write(get('ln_2.weight').tobytes())
                f.write(get('ln_2.bias').tobytes())
                f.write(get('mlp.c_fc.weight').flatten().tobytes())
                f.write(get('mlp.c_fc.bias').tobytes())
                f.write(get('mlp.c_proj.weight').flatten().tobytes())
                f.write(get('mlp.c_proj.bias').tobytes())

            f.write(tensor_to_np(sd['transformer.ln_f.weight']).tobytes())
            f.write(tensor_to_np(sd['transformer.ln_f.bias']).tobytes())

        # Save tokenizer files (vocab.json, merges.txt)
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        tokenizer.save_pretrained('.')
        print("Wrote", output_bin, "and tokenizer files in current directory.")

    if __name__ == "__main__":
        if len(sys.argv) != 3:
            print("Usage: python convert_checkpoint.py gpt2 weights.bin")
            sys.exit(1)
        main(sys.argv[1], sys.argv[2])
