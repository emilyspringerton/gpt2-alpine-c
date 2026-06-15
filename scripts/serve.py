#!/usr/bin/env python3
"""
Emily Prime GPT-2 inference server.

Loads the fine-tuned checkpoint once at startup and serves it via HTTP on :8088.
No external web framework required — uses stdlib http.server.

Endpoints:
  GET  /health          -> {"status": "ok", "model": "..."}
  POST /generate        -> {"prompt": "...", "max_tokens": 100, "temperature": 0.8}
                       <-  {"text": "...", "prompt": "...", "tokens": N, "model": "..."}

Run:
  python3 scripts/serve.py                          # fine-tuned model (default)
  python3 scripts/serve.py --model base             # base GPT-2
  python3 scripts/serve.py --checkpoint path/to/cp  # explicit checkpoint path
  python3 scripts/serve.py --port 8089
"""

import argparse
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
REPO_DIR   = SCRIPT_DIR.parent

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

# Loaded at startup — shared across all requests.
_model     = None
_tokenizer = None
_model_tag = None
_lock      = threading.Lock()


def load_model(checkpoint_path: str, base: bool = False):
    global _model, _tokenizer, _model_tag
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    if base:
        tag = "gpt2-base"
        print(f"[serve] loading base GPT-2 from HF cache ...", flush=True)
        _tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        _model     = GPT2LMHeadModel.from_pretrained("gpt2")
    else:
        tag = f"emily-ft ({Path(checkpoint_path).name})"
        print(f"[serve] loading fine-tuned checkpoint: {checkpoint_path} ...", flush=True)
        _tokenizer = GPT2Tokenizer.from_pretrained(checkpoint_path)
        _model     = GPT2LMHeadModel.from_pretrained(checkpoint_path)

    _tokenizer.pad_token = _tokenizer.eos_token
    _model.eval()
    _model_tag = tag
    print(f"[serve] model ready: {_model_tag}", flush=True)


def generate(prompt: str, max_tokens: int = 100, temperature: float = 0.8) -> tuple[str, int]:
    import torch
    max_tokens = max(1, min(max_tokens, 512))
    temperature = max(0.01, min(temperature, 2.0))

    inputs = _tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    prompt_len = input_ids.shape[1]

    with torch.no_grad():
        output = _model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            do_sample=(temperature > 0.01),
            temperature=temperature,
            pad_token_id=_tokenizer.eos_token_id,
        )

    new_ids = output[0][prompt_len:]
    text = _tokenizer.decode(new_ids, skip_special_tokens=True)
    return text, len(new_ids)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # suppress default access log
        pass

    def _send_json(self, code: int, obj: dict):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict | None:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        try:
            return json.loads(self.rfile.read(length))
        except Exception:
            return None

    def do_GET(self):
        if self.path == "/health":
            self._send_json(200, {"status": "ok", "model": _model_tag})
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/generate":
            self._send_json(404, {"error": "not found"})
            return

        body = self._read_json()
        if body is None:
            self._send_json(400, {"error": "invalid JSON"})
            return

        prompt     = str(body.get("prompt", "Emily Prime:"))
        max_tokens = int(body.get("max_tokens", 100))
        temperature = float(body.get("temperature", 0.8))

        try:
            with _lock:
                text, n_tokens = generate(prompt, max_tokens, temperature)
            self._send_json(200, {
                "text":    text,
                "prompt":  prompt,
                "tokens":  n_tokens,
                "model":   _model_tag,
            })
        except Exception as e:
            self._send_json(500, {"error": str(e)})


def main():
    parser = argparse.ArgumentParser(description="Emily Prime GPT-2 inference server")
    parser.add_argument("--port",       type=int, default=8088)
    parser.add_argument("--host",       default="0.0.0.0")
    parser.add_argument("--model",      choices=["ft", "base"], default="ft",
                        help="ft = fine-tuned emily-ft, base = vanilla GPT-2")
    parser.add_argument("--checkpoint", default=str(REPO_DIR / "checkpoint-emily-ft"),
                        help="path to HuggingFace checkpoint directory")
    args = parser.parse_args()

    use_base = (args.model == "base")
    load_model(args.checkpoint, base=use_base)

    server = HTTPServer((args.host, args.port), Handler)
    print(f"[serve] listening on http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[serve] shutting down", flush=True)


if __name__ == "__main__":
    main()
