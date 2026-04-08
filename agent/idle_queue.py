#!/usr/bin/env python3
"""
GPU idle-time work queue for Subconscious.

After 70B responds and no new message for 5 seconds, runs:
1. Contradiction scan: compare recent memories, 70B resolves conflicts
2. Retrieval scoring: adjust relevance based on usage

Preemption: if user sends a message during idle processing,
abandon immediately. Generation always wins.

Copyright 2026 Nick Lo. MIT License.
"""

import json
import logging
import threading
import time
import urllib.request

log = logging.getLogger("idle_queue")

MLX_URL = "http://localhost:8899/v1/chat/completions"
IDLE_DELAY = 5.0  # seconds before idle tasks start
MAX_PAIRS = 5     # max contradiction pairs per idle cycle


class IdleQueue:
    def __init__(self, memory_bridge):
        self.memory = memory_bridge
        self._timer = None
        self._running = False
        self._cancelled = False
        self._last_injected = []  # memories injected in last turn
        self._last_response = ""  # 70B response text from last turn
        self._stats = {"scans": 0, "contradictions": 0, "relevance_updates": 0}

    def schedule(self, injected_memories=None, response_text=""):
        """Schedule idle tasks after 70B responds. Call on every generation."""
        self.cancel()  # Cancel any pending idle work
        self._last_injected = injected_memories or []
        self._last_response = response_text
        self._cancelled = False
        self._timer = threading.Timer(IDLE_DELAY, self._run_idle)
        self._timer.daemon = True
        self._timer.start()

    def cancel(self):
        """Cancel pending idle work. Call when new user message arrives."""
        self._cancelled = True
        if self._timer:
            self._timer.cancel()
            self._timer = None

    def _run_idle(self):
        """Execute idle tasks. Checks for cancellation between steps."""
        if self._cancelled:
            return

        self._running = True
        try:
            # Task 1: Retrieval quality scoring (fast, no 70B needed)
            if not self._cancelled:
                self._score_retrieval()

            # Task 2: Contradiction scan (uses 70B, slower)
            if not self._cancelled:
                self._scan_contradictions()
        except Exception as e:
            log.debug("Idle task error: %s", e)
        finally:
            self._running = False

    def _score_retrieval(self):
        """Score injected memories based on whether 70B used them."""
        if not self._last_injected or not self._last_response:
            return

        response_lower = self._last_response.lower()
        updates = 0

        for mem in self._last_injected:
            if self._cancelled:
                return

            mem_text = mem.get("text", "")
            # Extract significant words from memory
            mem_words = set(w.lower() for w in mem_text.split() if len(w) > 4)

            # Check if response references memory content
            overlap = sum(1 for w in mem_words if w in response_lower)
            was_used = overlap >= 3  # at least 3 significant words in common

            # Update relevance in ChromaDB
            try:
                score_delta = 0.10 if was_used else -0.05
                # We'd need the memory ID to update — for now, log the intent
                log.debug("Retrieval score: %s%+.2f (%s)",
                         mem_text[:50], score_delta,
                         "used" if was_used else "unused")
                updates += 1
            except Exception:
                pass

        if updates:
            self._stats["relevance_updates"] += updates
            log.info("Retrieval scoring: %d memories scored", updates)

    def _scan_contradictions(self):
        """Find and resolve contradictions via 70B."""
        if not self.memory or not hasattr(self.memory, 'daemon') or not self.memory.daemon:
            return

        store = self.memory.daemon.store
        # Main 24/25: LocalMemoryStore exposes count() not _counter.
        try:
            store_count = store.count()
        except AttributeError:
            store_count = getattr(store, "_counter", 0)
        if not store or store_count < 10:
            return

        # Get 10 most recent memories
        try:
            recent = store.collection.get(
                include=["documents", "metadatas"],
                limit=10,
                offset=max(0, store_count - 10)
            )
        except Exception:
            return

        if not recent["ids"] or len(recent["ids"]) < 2:
            return

        # Find candidate contradiction pairs
        pairs_checked = 0
        for i in range(len(recent["ids"])):
            if self._cancelled or pairs_checked >= MAX_PAIRS:
                break

            meta_i = recent["metadatas"][i]
            if meta_i.get("superseded_by"):
                continue

            doc_i = recent["documents"][i]
            entities_i = set(json.loads(meta_i.get("entities", "[]")))

            # Query store for similar memories with shared entities
            try:
                emb = store.emb_model.encode([doc_i], normalize_embeddings=True)[0]
                results = store.collection.query(
                    query_embeddings=[emb.tolist()],
                    n_results=5,
                )
            except Exception:
                continue

            for j in range(len(results["ids"][0])):
                if self._cancelled:
                    return
                if results["ids"][0][j] == recent["ids"][i]:
                    continue

                other_doc = results["documents"][0][j]
                other_meta = results["metadatas"][0][j]
                if other_meta.get("superseded_by"):
                    continue

                # Check entity overlap
                entities_j = set(json.loads(other_meta.get("entities", "[]")))
                if not entities_i or not entities_j:
                    continue
                if not entities_i & entities_j:
                    continue

                # Potential contradiction — ask 70B
                sim = 1 - results["distances"][0][j]
                if sim < 0.60 or sim > 0.94:
                    continue

                pairs_checked += 1
                verdict = self._ask_70b_contradiction(
                    doc_i, meta_i.get("timestamp", ""),
                    other_doc, other_meta.get("timestamp", ""))

                if verdict and verdict.startswith("SUPERSEDES:"):
                    winner = verdict.split(":")[1].strip()
                    if winner == "A":
                        loser_id = results["ids"][0][j]
                    elif winner == "B":
                        loser_id = recent["ids"][i]
                    else:
                        continue

                    try:
                        loser_meta = store.collection.get(ids=[loser_id], include=["metadatas"])
                        if loser_meta["metadatas"]:
                            m = loser_meta["metadatas"][0].copy()
                            m["superseded_by"] = "idle_contradiction_scan"
                            store.collection.update(ids=[loser_id], metadatas=[m])
                            self._stats["contradictions"] += 1
                            log.info("Contradiction resolved: superseded %s", loser_id)
                    except Exception:
                        pass

        self._stats["scans"] += 1

    def _ask_70b_contradiction(self, doc_a, ts_a, doc_b, ts_b):
        """Ask 70B to resolve a potential contradiction."""
        if self._cancelled:
            return None

        prompt = (
            f"Memory A: {doc_a} (from {ts_a})\n"
            f"Memory B: {doc_b} (from {ts_b})\n\n"
            "Do these contradict each other? Respond with exactly one:\n"
            "COMPATIBLE — no contradiction\n"
            "SUPERSEDES:A — Memory A is correct, B is outdated\n"
            "SUPERSEDES:B — Memory B is correct, A is outdated"
        )

        try:
            data = json.dumps({
                "model": "mlx-community/Qwen2.5-72B-Instruct-4bit",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 20,
                "temperature": 0.0,
            }).encode()
            req = urllib.request.Request(MLX_URL, data=data,
                headers={"Content-Type": "application/json"})
            resp = urllib.request.urlopen(req, timeout=15)
            result = json.loads(resp.read())
            text = result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()

            for line in text.split("\n"):
                line = line.strip()
                if line.startswith("COMPATIBLE") or line.startswith("SUPERSEDES:"):
                    return line

        except Exception:
            pass
        return None

    @property
    def stats(self):
        return self._stats.copy()
