"""Minimal OpenAI-compatible /v1/embeddings server on CPU for ShinkaEvolve's code-novelty filter.

  python embed_server.py [port]        (default 8001; the GPU stays with the vLLM proposal model)
Uses sentence-transformers all-MiniLM-L6-v2 when it is installed (averaged over 240-token windows so the
whole program counts), else a hashed character-trigram embedding (deterministic, no download). The chosen backend is printed at start and served as the model id.
"""
import hashlib
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    BACKEND = "all-MiniLM-L6-v2"

    def embed(texts):
        """Mean of window embeddings over the whole text: MiniLM reads only 256 word pieces, and a policy is
        ~12k characters, so one call would see the docstring alone and every program would look identical."""
        tok = MODEL.tokenizer
        out = []
        for text in texts:
            ids = tok(text, add_special_tokens=False)["input_ids"] or [tok.unk_token_id]
            windows = [tok.decode(ids[i:i + 240]) for i in range(0, len(ids), 200)]  # 40-token overlap
            v = MODEL.encode(windows, normalize_embeddings=True).mean(axis=0)
            out.append((v / (np.linalg.norm(v) or 1)).tolist())
        return out
except Exception:  # no sentence-transformers or no network: hashed trigrams
    BACKEND = "hashed-trigram-1024"

    def embed(texts):
        out = []
        for text in texts:
            v = np.zeros(1024)
            for i in range(max(len(text) - 2, 0)):
                v[int.from_bytes(hashlib.blake2b(text[i:i + 3].encode(), digest_size=4).digest(), "little") % 1024] += 1
            out.append((v / (np.linalg.norm(v) or 1)).tolist())
        return out


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def reply(self, body, code=200):
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self.reply({"object": "list", "data": [{"id": BACKEND, "object": "model"}]})

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        texts = req.get("input", "")
        texts = [texts] if isinstance(texts, str) else list(texts)
        vecs = embed(texts)
        self.reply({"object": "list", "model": req.get("model", BACKEND),
                    "data": [{"object": "embedding", "index": i, "embedding": v} for i, v in enumerate(vecs)],
                    "usage": {"prompt_tokens": sum(len(t) // 4 for t in texts),
                              "total_tokens": sum(len(t) // 4 for t in texts)}})


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8001
    print("embedding backend:", BACKEND, "port", port, flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
