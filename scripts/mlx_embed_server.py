#!/usr/bin/env python3
"""
Minimal persistent MLX embedding server (Apple Silicon).

Loads a 4-bit Qwen3-Embedding model once and serves an Ollama-compatible
POST /api/embed endpoint, so OllamaEmbedder can point at it unchanged and get
~10x the throughput of Ollama's own runtime on an M-series Mac.

  MLX_EMBED_MODEL=mlx-community/Qwen3-Embedding-4B-4bit-DWQ \
  MLX_EMBED_PORT=11435 .venv/bin/python scripts/mlx_embed_server.py

Request:  {"model": "...", "input": "text" | ["t1","t2"]}
Response: {"embeddings": [[...], ...]}   (raw; the client L2-normalizes)
"""

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import mlx.core as mx
from mlx_embeddings import generate, load

MODEL = os.getenv("MLX_EMBED_MODEL", "mlx-community/Qwen3-Embedding-4B-4bit-DWQ")
PORT = int(os.getenv("MLX_EMBED_PORT", "11435"))
# Bind 127.0.0.1 for local-only; set 0.0.0.0 to serve peers over the tailnet.
HOST = os.getenv("MLX_EMBED_HOST", "127.0.0.1")

print(f"[mlx-embed] loading {MODEL} ...", flush=True)
_model, _tokenizer = load(MODEL)
print(f"[mlx-embed] loaded; serving on http://{HOST}:{PORT}/api/embed", flush=True)


def embed_texts(texts: list[str]) -> list[list[float]]:
    out = generate(_model, _tokenizer, texts=texts)
    emb = getattr(out, "text_embeds", out)
    mx.eval(emb)
    return emb.tolist()


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802
        if self.path != "/api/embed":
            self.send_response(404)
            self.end_headers()
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            inp = body.get("input")
            texts = [inp] if isinstance(inp, str) else list(inp)
            vectors = embed_texts(texts)
            payload = json.dumps({"embeddings": vectors}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        except Exception as e:  # noqa: BLE001 - return error to client, keep serving
            msg = json.dumps({"error": str(e)}).encode()
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(msg)))
            self.end_headers()
            self.wfile.write(msg)

    def log_message(self, *args):  # silence per-request logging
        pass


if __name__ == "__main__":
    try:
        ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
    except KeyboardInterrupt:
        sys.exit(0)
