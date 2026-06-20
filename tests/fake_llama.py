"""A minimal stand-in for llama-server's HTTP surface, for tests.

Usage: python fake_llama.py --port N [--ready-after SECONDS]
Serves GET /health (503 until ready-after elapsed, then 200), GET /v1/models,
and POST /v1/chat/completions. Ignores any extra (llama-server-like) flags.
"""

import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, HTTPServer


def _make_handler(start: float, ready_after: float):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # silence
            pass

        def _send(self, code: int, body: dict):
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/health":
                ready = (time.monotonic() - start) >= ready_after
                self._send(200 if ready else 503,
                           {"status": "ok" if ready else "loading"})
            elif self.path == "/v1/models":
                # real llama.cpp reports the runtime context under data[].meta.n_ctx
                self._send(200, {"data": [{"id": "fake",
                          "meta": {"n_ctx": 4096, "n_ctx_train": 8192}}]})
            else:
                self._send(404, {})

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0) or 0)
            self.rfile.read(length)
            if self.path == "/v1/chat/completions":
                self._send(200, {"choices": [{"message":
                          {"role": "assistant", "content": "ok"}}]})
            else:
                self._send(404, {})

    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--ready-after", type=float, default=0.0)
    args, _ = parser.parse_known_args()
    httpd = HTTPServer(("127.0.0.1", args.port),
                       _make_handler(time.monotonic(), args.ready_after))
    httpd.serve_forever()


if __name__ == "__main__":
    main()
