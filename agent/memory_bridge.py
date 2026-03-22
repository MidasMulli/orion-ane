"""
Shared MemoryBridge — single source of truth for agent_v2 and telegram_bot.
"""

import json
import os
import sys
import time
from datetime import datetime


VAULT_PATH = "/Users/midas/Desktop/cowork/vault"


class MemoryBridge:
    """Direct Python bridge to the memory daemon."""

    def __init__(self):
        self.daemon = None
        self._started = False

    def start(self, enable_enricher=True, enricher_interval=300):
        try:
            from phantom_memory.daemon import MemoryDaemon
        except ImportError:
            daemon_dir = os.path.join(os.path.dirname(__file__), "..", "memory")
            sys.path.insert(0, os.path.abspath(daemon_dir))
            from daemon import MemoryDaemon

        db_path = os.path.join(os.path.dirname(__file__), "..", "memory", "chromadb_live")
        self.daemon = MemoryDaemon(
            vault_path=VAULT_PATH, db_path=db_path,
            enable_enricher=enable_enricher, enricher_interval=enricher_interval,
        )
        self.daemon.start()
        self._started = True

    def ingest(self, role, text):
        if not self._started:
            return {"error": "daemon not started"}
        self.daemon.ingest(role, text)
        time.sleep(0.3)
        s = self.daemon.stats
        return {"status": "stored", "extracted": s["extracted"],
                "stored": s["stored"], "total_memories": s["total_memories"]}

    def recall(self, query, n_results=5, type_filter=""):
        if not self._started:
            return {"error": "daemon not started"}
        fv = type_filter if type_filter in ("decision", "task", "preference", "quantitative", "general") else None
        memories = self.daemon.store.recall(query, n_results=n_results, type_filter=fv)
        results = []
        for m in memories:
            meta = m["metadata"]
            results.append({
                "text": m["text"], "type": meta.get("type", "unknown"),
                "score": round(m["score"], 3),
                "entities": json.loads(meta.get("entities", "[]")),
                "timestamp": meta.get("timestamp", ""),
            })
        return {"query": query, "results": results, "total_memories": self.daemon.store.count()}

    def stats(self):
        if not self._started:
            return {"error": "daemon not started"}
        s = self.daemon.stats
        return {"session": self.daemon.session_id, "ingested": s["ingested"],
                "extracted": s["extracted"], "stored": s["stored"],
                "deduped": s["deduped"], "superseded": s.get("superseded", 0),
                "total_memories": s["total_memories"]}

    def get_insights(self):
        if not self._started:
            return {"error": "daemon not started"}
        vault_path = self.daemon.vault.vault_path
        lines = []
        heartbeat_path = os.path.join(vault_path, "midas", ".enricher_heartbeat")
        if os.path.exists(heartbeat_path):
            try:
                with open(heartbeat_path) as f:
                    ts = f.read().strip()
                hb_time = datetime.fromisoformat(ts)
                age_min = (datetime.now() - hb_time).total_seconds() / 60
                lines.append(f"Enricher: {'running' if age_min < 10 else 'stale'} (heartbeat {age_min:.0f}m ago)")
            except Exception:
                lines.append("Enricher: unknown")
        rel_path = os.path.join(vault_path, "memory", "relationships.md")
        if os.path.exists(rel_path):
            with open(rel_path) as f:
                lines.append(f"Relationships: {f.read().count('## ')} entities")
        insights_dir = os.path.join(vault_path, "memory", "insights")
        if os.path.exists(insights_dir):
            files = sorted([f for f in os.listdir(insights_dir) if f.startswith("patterns-")], reverse=True)
            if files:
                with open(os.path.join(insights_dir, files[0])) as f:
                    headers = [l.replace("## ", "").strip() for l in f if l.startswith("## ")]
                if headers:
                    lines.append(f"Insights ({files[0]}): {', '.join(headers[:5])}")
        return {"summary": "\n".join(lines)}

    def stop(self):
        if self._started and self.daemon:
            self.daemon.stop()
