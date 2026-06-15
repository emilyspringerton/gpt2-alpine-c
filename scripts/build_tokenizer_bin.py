#!/usr/bin/env python3
"""
Convert weights/tokenizer.json (HuggingFace GPT-2 BPE) to weights/tokenizer.bin
(the binary format expected by src/tokenizer.c).

Binary layout:
  int32   vocab_size          (50257)
  for each token 0..vocab_size-1:
    int32  len
    len bytes  raw bytes of the token
  int32   num_merges
  for each merge:
    int32 first_id
    int32 second_id
    int32 result_id

Re-indexing:
  Token IDs 0-255  = single bytes (ID == byte value, matching how gpt2_encode initialises tokens)
  Token IDs 256-50255 = merged tokens in BPE merge order
  Token ID  50256  = <|endoftext|>
"""

import json
import struct
import sys
from pathlib import Path

REPO = Path(__file__).parent.parent
TOK_JSON = REPO / "weights" / "tokenizer.json"
TOK_BIN  = REPO / "weights" / "tokenizer.bin"
VOCAB_SIZE = 50257


def bytes_to_unicode():
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("¡"), ord("¬") + 1))
          + list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


def main():
    b2u = bytes_to_unicode()
    u2b = {v: k for k, v in b2u.items()}

    def gpt2_str_to_bytes(s: str) -> bytes:
        return bytes([u2b[c] for c in s])

    with open(TOK_JSON) as f:
        tok = json.load(f)

    vocab  = tok["model"]["vocab"]   # str → gpt2_id
    merges = tok["model"]["merges"]  # [[str_a, str_b], ...]

    # --- Build re-indexed token table ---
    # new_id → raw bytes
    id_to_bytes: dict[int, bytes] = {}
    # token str → new_id
    str_to_id: dict[str, int] = {}

    # Step 1: single-byte tokens (new_id == byte value)
    for s, _gpt2_id in vocab.items():
        if s == "<|endoftext|>":
            continue
        raw = gpt2_str_to_bytes(s)
        if len(raw) == 1:
            new_id = raw[0]
            id_to_bytes[new_id] = raw
            str_to_id[s] = new_id

    # Step 2: multi-byte tokens in merge order (new_ids 256..50255)
    next_id = 256
    for pair in merges:
        s_a, s_b = pair[0], pair[1]
        s_result = s_a + s_b
        if s_result in str_to_id:
            continue  # already assigned (shouldn't happen but guard)
        raw = gpt2_str_to_bytes(s_result)
        id_to_bytes[next_id] = raw
        str_to_id[s_result] = next_id
        next_id += 1

    # Step 3: <|endoftext|> = 50256
    id_to_bytes[50256] = b"<|endoftext|>"
    str_to_id["<|endoftext|>"] = 50256

    assert len(id_to_bytes) == VOCAB_SIZE, \
        f"expected {VOCAB_SIZE} tokens, got {len(id_to_bytes)}"

    # --- Build merge table ---
    merge_table: list[tuple[int, int, int]] = []
    for pair in merges:
        s_a, s_b = pair[0], pair[1]
        s_result = s_a + s_b
        if s_a not in str_to_id or s_b not in str_to_id or s_result not in str_to_id:
            continue
        merge_table.append((str_to_id[s_a], str_to_id[s_b], str_to_id[s_result]))

    # --- Write binary file ---
    with open(TOK_BIN, "wb") as f:
        f.write(struct.pack("<i", VOCAB_SIZE))
        for i in range(VOCAB_SIZE):
            raw = id_to_bytes.get(i, b"")
            f.write(struct.pack("<i", len(raw)))
            f.write(raw)
        f.write(struct.pack("<i", len(merge_table)))
        for first, second, result in merge_table:
            f.write(struct.pack("<iii", first, second, result))

    print(f"wrote {TOK_BIN}")
    print(f"  {VOCAB_SIZE} tokens, {len(merge_table)} merges")


if __name__ == "__main__":
    main()
