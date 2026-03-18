#!/usr/bin/env python3
"""
ANE Server — Persistent CoreML inference server for background vault analysis
===============================================================================

Keeps the Qwen3-1.7B CoreML model loaded on ANE and processes analysis tasks
from a queue. Zero GPU cost — runs entirely on Neural Engine + CPU.

Two modes:
  1. Generative mode (--meta): Full Qwen3-1.7B on ANE at 57 tok/s
     - Analyzes entity profiles, generates insights, classifies facts
     - Uses ANEMLL chat.py infrastructure for CoreML inference
  2. Heuristic mode (fallback): Regex-based classification only
     - Zero model cost, instant response
     - Used when CoreML model not available

Architecture:
    ANE Server
      ├── Model loader (ANEMLL chat.py infrastructure)
      ├── Unix socket listener (for enricher/daemon to submit tasks)
      ├── HTTP API (for dashboard/monitoring)
      └── Task queue (thread-safe)

Usage:
    # Start generative server:
    python ane_server.py --meta ~/Desktop/cowork/anemll/models/qwen3-1.7b-coreml/meta.yaml

    # Start heuristic-only server:
    python ane_server.py

    # Test mode (run prompts then exit):
    python ane_server.py --meta <path> --test

    # Client ping:
    python ane_server.py --ping

Endpoints:
    POST /analyze  {"prompt": "...", "max_tokens": 150} -> {"result": "...", "elapsed_ms": 123}
    POST /classify {"text": "..."}                      -> {"type": "...", "confidence": 0.85}
    GET  /health                                        -> {"status": "ok", "backend": "ane|heuristic"}
    GET  /stats                                         -> {"tasks_completed": 42, "uptime": 3600}
"""

import argparse
import json
import logging
import os
import re
import signal
import socket
import struct
import sys
import time
import threading
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ane_server")

# ── Constants ─────────────────────────────────────────────────

SOCKET_PATH = "/tmp/orion-ane-server.sock"
HTTP_PORT = 8423
DEFAULT_META = os.path.expanduser(
    "~/Desktop/cowork/anemll/models/qwen3-1.7b-coreml/meta.yaml"
)
DEFAULT_MAX_TOKENS = 150

# ── Heuristic classifier (from daemon.py FactExtractor) ──────

DECISION_MARKERS = [
    "decided", "chose", "selected", "will use", "switched to",
    "going with", "agreed", "confirmed", "settled on", "set at",
    "set to", "specified", "established",
    "we'll go with", "final answer", "approved",
]
TASK_MARKERS = [
    "need to", "todo", "next step", "will do",
    "plan to", "going to", "must", "remember to",
    "deadline", "due by", "by friday", "by monday",
    "by end of", "action item",
]
PREFERENCE_MARKERS = [
    "prefer", "we like", "we want", "always use",
    "never use", "our standard", "our policy", "we require",
    "we typically", "we usually", "our approach",
]
QUANTITY_PATTERN = re.compile(
    r'(?:USD|GBP|EUR|\$|£|€)\s*[\d,]+(?:\.\d+)?(?:\s*(?:million|billion|M|B|K))?\b'
    r'|[\d,]+(?:\.\d+)?%'
    r'|\b\d+\s+(?:days?|months?|years?|business days?)\b',
    re.IGNORECASE,
)


def heuristic_classify(text: str) -> tuple:
    """Classify text using keyword heuristics. Returns (type, confidence)."""
    s_lower = text.lower()
    for markers, label, conf in [
        (DECISION_MARKERS, "decision", 0.80),
        (TASK_MARKERS, "task", 0.80),
        (PREFERENCE_MARKERS, "preference", 0.80),
    ]:
        matched = sum(1 for m in markers if m in s_lower)
        if matched:
            return label, min(conf + matched * 0.05, 0.95)
    if QUANTITY_PATTERN.search(text):
        return "quantitative", 0.85
    return "general", 0.60


# ── ANE Generative Model ─────────────────────────────────────

class ANEModel:
    """Wraps ANEMLL model loading and generation into a clean interface.

    Keeps CoreML models loaded on ANE — zero GPU contention.
    Thread-safe via lock (one generation at a time).
    """

    def __init__(self, meta_path: str, max_tokens: int = DEFAULT_MAX_TOKENS):
        self.meta_path = meta_path
        self.max_tokens = max_tokens
        self.embed_model = None
        self.ffn_models = None
        self.lmhead_model = None
        self.tokenizer = None
        self.metadata = None
        self.state = None
        self.causal_mask = None
        self._lock = threading.Lock()
        self._loaded = False

        # Add ANEMLL to path
        anemll_tests = os.path.expanduser("~/Desktop/cowork/anemll/tests")
        if anemll_tests not in sys.path:
            sys.path.insert(0, anemll_tests)

    def load(self):
        """Load CoreML models onto ANE. Takes ~5-10s on first call."""
        import yaml
        import torch
        from chat import (
            load_models, create_unified_state,
            initialize_causal_mask, initialize_tokenizer,
            build_stop_token_ids,
        )

        log.info("Loading models from %s ...", self.meta_path)
        t0 = time.time()

        with open(self.meta_path) as f:
            meta = yaml.safe_load(f)

        params = meta["model_info"]["parameters"]
        model_dir = Path(self.meta_path).parent

        # Build args namespace for chat.py
        args = argparse.Namespace(
            d=str(model_dir),
            embed=str(model_dir / params["embeddings"]),
            ffn=str(model_dir / params["ffn"]),
            lmhead=str(model_dir / params["lm_head"]),
            tokenizer=str(model_dir),
            context_length=params["context_length"],
            batch_size=params.get("batch_size", 64),
            split_lm_head=params.get("split_lm_head", 16),
            num_logits=params.get("split_lm_head", 8),
            vocab_size=params.get("vocab_size", None),
            lm_head_chunk_sizes=params.get("lm_head_chunk_sizes", None),
            cpu=False,
            eval=True,
            debug=False,
            argmax_in_model=False,
            debug_argmax=False,
            split_rotate=False,
            pf=None,
            sliding_window=None,
            attention_size=params["context_length"],
            mem_report=False,
        )

        self.tokenizer = initialize_tokenizer(str(model_dir), True)
        if self.tokenizer is None:
            raise RuntimeError("Failed to load tokenizer")

        self.metadata = {}
        self.embed_model, self.ffn_models, self.lmhead_model, self.metadata = load_models(args, self.metadata)

        self.metadata.update({
            "context_length": params["context_length"],
            "state_length": params["context_length"],
            "batch_size": params.get("batch_size", 64),
            "split_lm_head": params.get("split_lm_head", 16),
            "debug": False,
            "argmax_in_model": False,
            "debug_argmax": False,
            "vocab_size": params.get("vocab_size", None),
        })
        if params.get("lm_head_chunk_sizes"):
            self.metadata["lm_head_chunk_sizes"] = params["lm_head_chunk_sizes"]
        if params.get("prefill_dynamic_slice"):
            self.metadata["prefill_dynamic_slice"] = True

        self.state = create_unified_state(
            self.ffn_models, self.metadata["context_length"], True, metadata=self.metadata
        )
        attention_size = self.metadata.get("attention_size", self.metadata["context_length"])
        self.causal_mask = initialize_causal_mask(attention_size, True)
        self.stop_token_ids = build_stop_token_ids(self.tokenizer)
        self._loaded = True

        elapsed = time.time() - t0
        log.info("Models loaded in %.1fs", elapsed)

    def warmup(self):
        """Run warmup inference to prime ANE dispatch."""
        log.info("Warming up ANE...")
        for _ in range(2):
            self.generate("Hello", max_tokens=10)
        log.info("Warmup complete")

    def generate(self, prompt: str, max_tokens: int = None) -> str:
        """Generate text from a prompt. Thread-safe via lock."""
        if not self._loaded:
            raise RuntimeError("Model not loaded")

        import torch
        from chat import (
            run_prefill, generate_next_token, create_unified_state,
        )

        max_tokens = max_tokens or self.max_tokens

        with self._lock:
            context_length = self.metadata["context_length"]
            batch_size = self.metadata.get("batch_size", 64)
            update_mask_prefill = self.metadata.get("update_mask_prefill", False)
            prefill_dynamic_slice = self.metadata.get("prefill_dynamic_slice", False)
            single_token_mode = not (update_mask_prefill or prefill_dynamic_slice)

            # Fresh state for each generation
            self.state = create_unified_state(
                self.ffn_models, context_length, True, metadata=self.metadata
            )

            # Tokenize with chat template
            try:
                messages = [{"role": "user", "content": prompt}]
                input_ids = self.tokenizer.apply_chat_template(
                    messages, return_tensors="pt",
                    add_generation_prompt=True,
                    enable_thinking=False,
                ).to(torch.int32)
            except Exception:
                input_ids = self.tokenizer(
                    prompt, return_tensors="pt", add_special_tokens=True
                ).input_ids.to(torch.int32)

            context_pos = input_ids.size(1)
            if context_pos >= context_length - 2:
                return "[ERROR: prompt too long for context window]"

            # Prefill
            run_prefill(
                self.embed_model, self.ffn_models, input_ids,
                context_pos, context_length, batch_size,
                self.state, self.causal_mask, None,
                single_token_mode=single_token_mode,
                use_update_mask=update_mask_prefill,
            )

            # Generate
            generated_ids = []
            pos = context_pos

            while pos < context_length - 1 and len(generated_ids) < max_tokens:
                next_token = generate_next_token(
                    self.embed_model, self.ffn_models, self.lmhead_model,
                    input_ids, pos, context_length,
                    self.metadata, self.state, self.causal_mask,
                )

                if next_token in self.stop_token_ids:
                    break

                generated_ids.append(next_token)

                if pos < input_ids.size(1):
                    input_ids[0, pos] = next_token
                else:
                    input_ids = torch.cat([
                        input_ids,
                        torch.tensor([[next_token]], dtype=torch.int32)
                    ], dim=1)

                pos += 1

            text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

            # Strip think tags
            if "<think>" in text:
                text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

            return text


# ── Socket Server ─────────────────────────────────────────────

class SocketServer:
    """Unix socket server for enricher/daemon to submit tasks."""

    def __init__(self, model: ANEModel = None, socket_path: str = SOCKET_PATH):
        self.model = model
        self.socket_path = socket_path
        self._running = False
        self._server_socket = None
        self._stats = {
            "tasks_completed": 0,
            "tasks_failed": 0,
            "uptime_start": None,
        }

    def start(self):
        """Start listening in a background thread."""
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)

        self._server_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._server_socket.bind(self.socket_path)
        self._server_socket.listen(5)
        self._server_socket.settimeout(1.0)
        self._running = True
        self._stats["uptime_start"] = time.time()

        self._thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._thread.start()
        log.info("Socket server listening on %s", self.socket_path)

    def stop(self):
        self._running = False
        if self._server_socket:
            self._server_socket.close()
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)

    def _accept_loop(self):
        while self._running:
            try:
                conn, _ = self._server_socket.accept()
                threading.Thread(target=self._handle, args=(conn,), daemon=True).start()
            except socket.timeout:
                continue
            except Exception as e:
                if self._running:
                    log.error("Socket accept error: %s", e)

    def _handle(self, conn):
        try:
            raw_len = self._recv_exact(conn, 4)
            if not raw_len:
                return
            msg_len = struct.unpack("!I", raw_len)[0]
            raw_msg = self._recv_exact(conn, msg_len)
            if not raw_msg:
                return

            request = json.loads(raw_msg.decode("utf-8"))
            cmd = request.get("cmd", "analyze")

            if cmd == "analyze" and self.model:
                prompt = request.get("prompt", "")
                max_tokens = request.get("max_tokens", DEFAULT_MAX_TOKENS)
                t0 = time.time()
                result = self.model.generate(prompt, max_tokens=max_tokens)
                elapsed = time.time() - t0
                response = {
                    "status": "ok",
                    "result": result,
                    "elapsed_ms": round(elapsed * 1000),
                }
                self._stats["tasks_completed"] += 1

            elif cmd == "classify":
                text = request.get("text", "")
                if self.model:
                    t0 = time.time()
                    prompt = (
                        "Classify this text into exactly one category. "
                        "Reply with ONLY the category name, nothing else.\n"
                        "Categories: decision, task, preference, quantitative, general\n"
                        f"Text: {text}\n"
                        "Category:"
                    )
                    result = self.model.generate(prompt, max_tokens=5)
                    elapsed = time.time() - t0
                    valid = {"decision", "task", "preference", "quantitative", "general"}
                    category = "general"
                    for word in result.lower().split():
                        cleaned = word.strip("*.,;:\"'()[]")
                        if cleaned in valid:
                            category = cleaned
                            break
                    response = {"status": "ok", "type": category, "confidence": 0.90, "elapsed_ms": round(elapsed * 1000)}
                else:
                    fact_type, confidence = heuristic_classify(text)
                    response = {"status": "ok", "type": fact_type, "confidence": confidence}
                self._stats["tasks_completed"] += 1

            elif cmd == "ping":
                response = {"status": "ok", "uptime": time.time() - self._stats["uptime_start"]}

            elif cmd == "stats":
                response = {"status": "ok", **self._stats, "uptime": time.time() - self._stats["uptime_start"]}

            else:
                response = {"status": "error", "error": f"Unknown command: {cmd}"}

        except Exception as e:
            response = {"status": "error", "error": str(e)}
            self._stats["tasks_failed"] += 1

        try:
            resp_bytes = json.dumps(response).encode("utf-8")
            conn.sendall(struct.pack("!I", len(resp_bytes)) + resp_bytes)
        except Exception:
            pass
        finally:
            conn.close()

    @staticmethod
    def _recv_exact(sock, n):
        data = b""
        while len(data) < n:
            chunk = sock.recv(n - len(data))
            if not chunk:
                return None
            data += chunk
        return data


# ── HTTP Server (for dashboard/monitoring) ────────────────────

class HTTPServer:
    """Lightweight HTTP API for monitoring and browser-based access."""

    def __init__(self, model: ANEModel = None, port: int = HTTP_PORT):
        self.model = model
        self.port = port
        self.start_time = time.time()
        self._stats = {"tasks_completed": 0}

    def start(self):
        """Start HTTP server in background thread."""
        from aiohttp import web
        import asyncio

        async def handle_analyze(request):
            try:
                body = await request.json()
            except Exception:
                return web.json_response({"error": "Invalid JSON"}, status=400)

            prompt = body.get("prompt", "").strip()
            if not prompt:
                return web.json_response({"error": "Missing prompt"}, status=400)

            max_tokens = body.get("max_tokens", DEFAULT_MAX_TOKENS)
            t0 = time.time()

            if self.model:
                result = self.model.generate(prompt, max_tokens=max_tokens)
            else:
                result = "[No generative model loaded — heuristic only]"

            elapsed = time.time() - t0
            self._stats["tasks_completed"] += 1
            return web.json_response({
                "result": result,
                "elapsed_ms": round(elapsed * 1000),
                "backend": "ane" if self.model else "none",
            })

        async def handle_classify(request):
            try:
                body = await request.json()
            except Exception:
                return web.json_response({"error": "Invalid JSON"}, status=400)
            text = body.get("text", "").strip()
            if not text:
                return web.json_response({"error": "Missing text"}, status=400)
            fact_type, confidence = heuristic_classify(text)
            return web.json_response({
                "type": fact_type, "confidence": round(confidence, 3),
                "backend": "ane" if self.model else "heuristic",
            })

        async def handle_health(request):
            return web.json_response({
                "status": "ok",
                "backend": "ane" if self.model else "heuristic",
                "uptime": round(time.time() - self.start_time, 1),
                "tasks_completed": self._stats["tasks_completed"],
            })

        def _run():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            app = web.Application()
            app.router.add_post("/analyze", handle_analyze)
            app.router.add_post("/classify", handle_classify)
            app.router.add_get("/health", handle_health)
            # handle_signals=False because we're in a non-main thread
            web.run_app(app, host="0.0.0.0", port=self.port, print=None, handle_signals=False)

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        log.info("HTTP server on http://localhost:%d", self.port)


# ── Client ────────────────────────────────────────────────────

class ANEClient:
    """Client for submitting tasks to the ANE server via Unix socket."""

    def __init__(self, socket_path: str = SOCKET_PATH):
        self.socket_path = socket_path

    def _send(self, request: dict, timeout: float = 120.0) -> dict:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        try:
            sock.connect(self.socket_path)
            msg = json.dumps(request).encode("utf-8")
            sock.sendall(struct.pack("!I", len(msg)) + msg)

            raw_len = b""
            while len(raw_len) < 4:
                chunk = sock.recv(4 - len(raw_len))
                if not chunk:
                    raise ConnectionError("Server closed connection")
                raw_len += chunk
            resp_len = struct.unpack("!I", raw_len)[0]

            raw_resp = b""
            while len(raw_resp) < resp_len:
                chunk = sock.recv(resp_len - len(raw_resp))
                if not chunk:
                    raise ConnectionError("Server closed connection")
                raw_resp += chunk

            return json.loads(raw_resp.decode("utf-8"))
        finally:
            sock.close()

    def analyze(self, prompt: str, max_tokens: int = DEFAULT_MAX_TOKENS) -> str:
        """Submit an analysis prompt and get the result text."""
        resp = self._send({"cmd": "analyze", "prompt": prompt, "max_tokens": max_tokens})
        if resp["status"] == "ok":
            return resp["result"]
        raise RuntimeError(resp.get("error", "Unknown error"))

    def classify(self, text: str) -> tuple:
        """Classify text. Returns (type, confidence)."""
        resp = self._send({"cmd": "classify", "text": text})
        if resp["status"] == "ok":
            return resp["type"], resp["confidence"]
        raise RuntimeError(resp.get("error", "Unknown error"))

    def ping(self) -> dict:
        return self._send({"cmd": "ping"}, timeout=5.0)

    def stats(self) -> dict:
        return self._send({"cmd": "stats"}, timeout=5.0)

    @staticmethod
    def is_running(socket_path: str = SOCKET_PATH) -> bool:
        if not os.path.exists(socket_path):
            return False
        try:
            ANEClient(socket_path).ping()
            return True
        except Exception:
            return False


# ── CLI ───────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Orion ANE Server — 1.7B on Neural Engine")
    parser.add_argument("--meta", default=None, help="Path to meta.yaml for generative mode")
    parser.add_argument("--socket", default=SOCKET_PATH, help="Unix socket path")
    parser.add_argument("--port", type=int, default=HTTP_PORT, help="HTTP port")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--test", action="store_true", help="Run test prompts then exit")
    parser.add_argument("--ping", action="store_true", help="Ping running server")
    parser.add_argument("--stats", action="store_true", help="Get server stats")
    args = parser.parse_args()

    # Client-only commands
    if args.ping:
        client = ANEClient(args.socket)
        try:
            resp = client.ping()
            print(f"ANE Server: alive (uptime {resp['uptime']:.0f}s)")
        except Exception as e:
            print(f"ANE Server: not running ({e})")
        return

    if args.stats:
        client = ANEClient(args.socket)
        resp = client.stats()
        for k, v in resp.items():
            print(f"  {k}: {v}")
        return

    # Server mode
    print("=" * 60)
    print("  ORION ANE SERVER — Qwen3-1.7B on Neural Engine")
    print("=" * 60)

    model = None
    meta_path = args.meta or DEFAULT_META

    if os.path.exists(meta_path):
        try:
            model = ANEModel(meta_path, max_tokens=args.max_tokens)
            model.load()
            model.warmup()
            log.info("Backend: ANE generative (Qwen3-1.7B)")
        except Exception as e:
            log.warning("ANE model load failed: %s — falling back to heuristic", e)
            model = None
    else:
        log.info("No meta.yaml found at %s — heuristic mode only", meta_path)

    if args.test:
        if not model:
            print("ERROR: --test requires a loaded model (--meta)")
            return
        prompts = [
            "Classify this fact as decision/task/preference/quantitative/general: The cross-default threshold is $50M including affiliates.",
            "What entities are related in this text: JPMorgan and Goldman Sachs both have ISDA Master Agreements with cross-default provisions.",
            "Summarize this entity profile: Counterparty Alpha — cross-default threshold $50M, minimum transfer amount $500K, eligible collateral US Treasuries with 2% haircut.",
        ]
        for p in prompts:
            print(f"\n{'─' * 60}")
            print(f"PROMPT: {p[:80]}...")
            t0 = time.time()
            result = model.generate(p)
            elapsed = time.time() - t0
            print(f"RESULT ({elapsed:.1f}s): {result}")
        return

    # Start socket + HTTP servers
    sock_server = SocketServer(model, args.socket)
    sock_server.start()

    http_server = HTTPServer(model, args.port)
    http_server.start()

    log.info("Ready — Socket: %s, HTTP: http://localhost:%d", args.socket, args.port)
    log.info("Press Ctrl+C to stop")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        log.info("Shutting down...")
        sock_server.stop()


if __name__ == "__main__":
    main()
