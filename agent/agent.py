#!/usr/bin/env python3
"""
Cowork Agent — Lean general-purpose local AI agent
====================================================

Architecture:
  GPU  → Qwen 3.5 9B (MLX, port 8899) — reasoning, tool decisions
  CPU  → Memory daemon (embeddings, ChromaDB, vault) — runs during GPU work
  ANE  → Fact classifier (port 8423) — concurrent classification

One file. No plugins. No skill system. Just a sharp agent loop with
persistent memory and browser access.
"""

import glob as globmod
import io
import json
import logging
import os
import re
import signal
import subprocess
import sys
import threading
import time
import traceback
import urllib.request
import warnings
from datetime import datetime

from openai import OpenAI
from browser import BrowserBridge, BROWSER_TOOLS

# ── Config ──────────────────────────────────────────────────────────────────

MLX_BASE_URL = os.environ.get("MLX_BASE_URL", "http://127.0.0.1:8899/v1")
MLX_MODEL = os.environ.get("MLX_MODEL", "mlx-community/Qwen3.5-9B-MLX-4bit")
DASHBOARD_PORT = 8422
MONITOR_PORT = 8425
METRICS_FILE = os.path.join(os.path.dirname(__file__), "metrics.json")
MAX_HISTORY = 24  # trim conversation beyond this many messages
MAX_TOOL_ROUNDS = 10  # prevent infinite tool loops
TOOL_TEMPERATURE = 0.3  # lower temp for tool-calling rounds (structured output)
CHAT_TEMPERATURE = 0.7  # normal temp for free-form responses

SYSTEM_PROMPT = """You are Midas, a sharp AI assistant on Apple Silicon with persistent memory and browser access.

## TOOL GUIDE
- memory_ingest: Store facts (browsing, insights, dates). User messages auto-stored.
- memory_recall: Search past conversations. Use when user references history.
- memory_insights: Entity relationships and patterns from the enricher.
- vault_read: Read the Obsidian knowledge vault (projects, roadmap, domain docs). Read-only.
- vault_insight: Cross-reference vault + memory for deep context.
- scan_digest: Access scan candidates from the background scanner.
  - mode="latest": See recent scan items (HN, RSS, Reddit feeds)
  - mode="unreviewed": List scans not yet reviewed
  - mode="stats": Scanner calibration statistics
  To "clear candidates" = use scan_digest(mode="latest") to read them, summarize for the user, then ingest key findings to memory.
- browse_search: Google search. Use FIRST for factual lookups. MUST include a non-empty query.
- browse_x_feed: Scan X feed. Use this, not browse_navigate, for X scanning.
- browse_navigate/read/click/type/js: Chrome CDP browser (user's logged-in sessions).
- shell: Run shell commands.
- playbook_update: Update your self-knowledge file after tasks.

## RULES
- ACT, don't narrate. If you can accomplish something with a tool call, DO IT immediately. Never say "I'll do X" or "Give me a minute" — just call the tool.
- NEVER call browse_search with an empty query.
- NEVER navigate to URLs you invented. Only navigate to URLs from search results, memory, or user input.
- If a tool returns no useful result, stop and tell the user rather than retrying with different empty inputs.
- When the user says "clear" or "process" candidates/scans, use scan_digest — do NOT browse random websites.

## VOICE
Direct, concise, no corporate filler. You're a partner, not an assistant. The user is a VP in investment banking (ISDA, collateral, regulatory). Match that level. Push back when warranted. Surface connections from memory unprompted. Dry wit welcome, sycophancy forbidden.
"""

# ── Tool Definitions (OpenAI function calling format) ───────────────────────

TOOLS = [
    {"type":"function","function":{"name":"memory_ingest","description":"Store info in memory","parameters":{"type":"object","properties":{"role":{"type":"string","enum":["user","assistant"]},"text":{"type":"string"}},"required":["role","text"]}}},
    {"type":"function","function":{"name":"memory_recall","description":"Search past conversations","parameters":{"type":"object","properties":{"query":{"type":"string"},"n_results":{"type":"integer","default":5}},"required":["query"]}}},
    {"type":"function","function":{"name":"memory_stats","description":"Memory stats","parameters":{"type":"object","properties":{}}}},
    {"type":"function","function":{"name":"memory_insights","description":"Entity patterns","parameters":{"type":"object","properties":{}}}},
    {"type":"function","function":{"name":"vault_read","description":"Read vault files","parameters":{"type":"object","properties":{"path":{"type":"string","default":""},"query":{"type":"string","default":""}}}}},
    {"type":"function","function":{"name":"vault_insight","description":"Cross-ref vault+memory","parameters":{"type":"object","properties":{"topic":{"type":"string"}},"required":["topic"]}}},
    {"type":"function","function":{"name":"playbook_update","description":"Update playbook","parameters":{"type":"object","properties":{"section":{"type":"string"},"action":{"type":"string","enum":["append","replace","read"]},"content":{"type":"string","default":""}},"required":["section","action"]}}},
    {"type":"function","function":{"name":"scan_digest","description":"Read scan candidates from LOCAL files. Use this when asked to process, clear, review, or check scans/candidates. These are pre-scraped items from HN/RSS/Reddit stored as JSON — NOT live web pages. Do NOT browse the web for this. Use mode=clear to process all pending scans at once.","parameters":{"type":"object","properties":{"mode":{"type":"string","enum":["latest","unreviewed","clear","verdicts","stats"],"default":"latest"},"top_n":{"type":"integer","default":10}}}}},
    {"type":"function","function":{"name":"message_claude","description":"Message Claude","parameters":{"type":"object","properties":{"priority":{"type":"string","enum":["low","medium","high"],"default":"medium"},"message":{"type":"string"},"context":{"type":"string","default":""}},"required":["message"]}}},
    {"type":"function","function":{"name":"shell","description":"Run shell command","parameters":{"type":"object","properties":{"command":{"type":"string"}},"required":["command"]}}},
]

# ── Vault Access (read-only) ───────────────────────────────────────────────

VAULT_PATH = "/Users/midas/Desktop/cowork/vault"

def vault_read(path: str = "", query: str = "") -> dict:
    """Read files or search the Obsidian vault. Read-only."""
    if query:
        # Search across all markdown files
        matches = []
        for md_file in globmod.glob(os.path.join(VAULT_PATH, "**/*.md"), recursive=True):
            # Skip memory/ subdirectory (that's the daemon's space)
            rel = os.path.relpath(md_file, VAULT_PATH)
            try:
                with open(md_file, "r") as f:
                    content = f.read()
                if query.lower() in content.lower():
                    # Extract matching lines with context
                    lines = content.split("\n")
                    snippets = []
                    for i, line in enumerate(lines):
                        if query.lower() in line.lower():
                            start = max(0, i - 1)
                            end = min(len(lines), i + 2)
                            snippets.append("\n".join(lines[start:end]))
                    matches.append({
                        "file": rel,
                        "snippets": snippets[:3],  # top 3 matches per file
                    })
            except Exception:
                continue
        return {"query": query, "matches": matches[:10]}

    if not path:
        # List vault structure
        structure = {}
        for md_file in sorted(globmod.glob(os.path.join(VAULT_PATH, "**/*.md"), recursive=True)):
            rel = os.path.relpath(md_file, VAULT_PATH)
            # Skip memory/ (daemon's space) for cleaner listing
            if rel.startswith("memory/"):
                continue
            parts = rel.split("/")
            d = structure
            for p in parts[:-1]:
                d = d.setdefault(p, {})
            d[parts[-1]] = os.path.getsize(md_file)
        return {"vault_path": VAULT_PATH, "structure": structure}

    # Read specific file
    full_path = os.path.join(VAULT_PATH, path)
    if os.path.isdir(full_path):
        files = []
        for f in sorted(os.listdir(full_path)):
            if f.endswith(".md"):
                files.append(f)
        return {"directory": path, "files": files}

    if not os.path.exists(full_path):
        return {"error": f"File not found: {path}"}

    try:
        with open(full_path, "r") as f:
            content = f.read()
        # Truncate very long files
        if len(content) > 8000:
            content = content[:8000] + "\n\n[... truncated, ask for specific section ...]"
        return {"file": path, "content": content}
    except Exception as e:
        return {"error": str(e)}


def vault_insight(topic: str, memory_bridge) -> dict:
    """Cross-reference vault project context with conversation memories."""

    result = {"topic": topic, "vault_context": [], "memory_context": []}

    # 1. Search vault for relevant content
    vault_hits = []
    key_files = [
        "HOME.md", "Roadmap.md", "Decision Log.md",
        "Infrastructure Map.md",
    ]
    # Also scan active projects
    projects_dir = os.path.join(VAULT_PATH, "projects", "active")
    if os.path.isdir(projects_dir):
        for f in os.listdir(projects_dir):
            if f.endswith(".md"):
                key_files.append(f"projects/active/{f}")

    # Also scan domain docs
    domain_dir = os.path.join(VAULT_PATH, "domain")
    if os.path.isdir(domain_dir):
        for root, dirs, files in os.walk(domain_dir):
            for f in files:
                if f.endswith(".md"):
                    rel = os.path.relpath(os.path.join(root, f), VAULT_PATH)
                    key_files.append(rel)

    for rel_path in key_files:
        full = os.path.join(VAULT_PATH, rel_path)
        if not os.path.exists(full):
            continue
        try:
            with open(full, "r") as f:
                content = f.read()
            if topic.lower() in content.lower() or any(
                word.lower() in content.lower()
                for word in topic.split() if len(word) > 3
            ):
                # Extract relevant sections
                lines = content.split("\n")
                relevant = []
                for i, line in enumerate(lines):
                    if any(word.lower() in line.lower() for word in topic.split() if len(word) > 3):
                        start = max(0, i - 2)
                        end = min(len(lines), i + 5)
                        relevant.append("\n".join(lines[start:end]))
                if relevant:
                    vault_hits.append({
                        "file": rel_path,
                        "excerpts": relevant[:3],
                    })
        except Exception:
            continue

    result["vault_context"] = vault_hits[:5]

    # 2. Search memories for the topic
    if memory_bridge._started:
        memories = memory_bridge.recall(topic, n_results=10)
        result["memory_context"] = memories.get("results", [])
        result["total_memories"] = memories.get("total_memories", 0)

    # 3. Generate cross-references
    vault_entities = set()
    for hit in vault_hits:
        for excerpt in hit.get("excerpts", []):
            # Extract capitalized terms as potential entities
            for match in re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', excerpt):
                if len(match) > 3:
                    vault_entities.add(match)

    memory_entities = set()
    for mem in result.get("memory_context", []):
        for ent in mem.get("entities", []):
            memory_entities.add(ent)

    # Find entities that appear in both vault and memory
    overlap = vault_entities & memory_entities
    if overlap:
        result["cross_references"] = list(overlap)

    return result


# ── Playbook (Midas self-knowledge) ───────────────────────────────────────

PLAYBOOK_PATH = "/Users/midas/Desktop/cowork/vault/midas/playbook.md"

# Section markers in the playbook markdown
_PLAYBOOK_SECTIONS = {
    "scan_schedule": "## Scan Schedule",
    "what_works": "## What Works",
    "what_doesnt": "## What Doesn't Work",
    "high_signal": "## High-Signal Sources",
    "self_eval": "## Self-Eval",
    "improvement_queue": "## Improvement Queue",
    "lessons": "## Lessons Learned",
    "voice": "## Voice & Growth",
}

def playbook_tool(section: str, action: str, content: str = "") -> dict:
    """Read or update the Midas playbook."""
    if action == "read":
        try:
            with open(PLAYBOOK_PATH, "r") as f:
                text = f.read()
            if section == "full":
                return {"playbook": text}
            marker = _PLAYBOOK_SECTIONS.get(section)
            if not marker:
                return {"error": f"Unknown section: {section}. Valid: {list(_PLAYBOOK_SECTIONS.keys())}"}
            # Extract section content between this header and the next ## or ---
            idx = text.find(marker)
            if idx == -1:
                return {"error": f"Section '{marker}' not found in playbook"}
            start = idx + len(marker)
            # Find next section boundary
            rest = text[start:]
            end = len(rest)
            for boundary in ["\n## ", "\n---"]:
                pos = rest.find(boundary)
                if pos != -1 and pos < end:
                    end = pos
            return {"section": section, "content": rest[:end].strip()}
        except FileNotFoundError:
            return {"error": "Playbook not found at " + PLAYBOOK_PATH}

    if action in ("append", "replace"):
        if not content:
            return {"error": "content is required for append/replace"}
        try:
            with open(PLAYBOOK_PATH, "r") as f:
                text = f.read()
        except FileNotFoundError:
            return {"error": "Playbook not found at " + PLAYBOOK_PATH}

        marker = _PLAYBOOK_SECTIONS.get(section)
        if not marker:
            return {"error": f"Unknown section: {section}. Valid: {list(_PLAYBOOK_SECTIONS.keys())}"}

        idx = text.find(marker)
        if idx == -1:
            return {"error": f"Section '{marker}' not found in playbook"}

        start = idx + len(marker)
        rest = text[start:]
        end = len(rest)
        for boundary in ["\n## ", "\n---"]:
            pos = rest.find(boundary)
            if pos != -1 and pos < end:
                end = pos

        old_section = rest[:end]
        if action == "append":
            new_section = old_section.rstrip() + "\n" + content + "\n"
        else:  # replace
            new_section = "\n" + content + "\n"

        text = text[:start] + new_section + rest[end:]

        # Update the "Last updated" line
        today = datetime.now().strftime("%Y-%m-%d %H:%M")
        if "*Last updated:" in text:
            text = re.sub(r'\*Last updated:.*\*', f'*Last updated: {today} (auto)*', text)

        with open(PLAYBOOK_PATH, "w") as f:
            f.write(text)
        return {"status": "updated", "section": section, "action": action}

    return {"error": f"Unknown action: {action}. Use 'read', 'append', or 'replace'."}


# ── Memory Daemon (subprocess over MCP stdio) ──────────────────────────────

class MemoryBridge:
    """Direct Python bridge to the memory daemon — no MCP overhead."""

    def __init__(self):
        self.daemon = None
        self._started = False

    def start(self):
        """Import and start the daemon in-process with enricher."""
        try:
            from phantom_memory.daemon import MemoryDaemon
        except ImportError:
            daemon_dir = os.path.join(os.path.dirname(__file__), "..", "memory")
            sys.path.insert(0, os.path.abspath(daemon_dir))
            from daemon import MemoryDaemon

        vault_path = VAULT_PATH
        db_path = os.path.join(os.path.dirname(__file__), "..", "memory", "chromadb_live")

        self.daemon = MemoryDaemon(
            vault_path=vault_path, db_path=db_path,
            enable_enricher=True, enricher_interval=300,
        )
        self.daemon.start()
        self._started = True

    def ingest(self, role: str, text: str) -> dict:
        if not self._started:
            return {"error": "daemon not started"}
        self.daemon.ingest(role, text)
        time.sleep(0.3)  # let background thread process
        stats = self.daemon.stats
        return {
            "status": "stored",
            "extracted": stats["extracted"],
            "stored": stats["stored"],
            "total_memories": stats["total_memories"],
        }

    def recall(self, query: str, n_results: int = 5, type_filter: str = "") -> dict:
        if not self._started:
            return {"error": "daemon not started"}
        filter_val = type_filter if type_filter in ("decision", "task", "preference", "quantitative", "general") else None
        memories = self.daemon.store.recall(query, n_results=n_results, type_filter=filter_val)
        results = []
        for m in memories:
            meta = m["metadata"]
            results.append({
                "text": m["text"],
                "type": meta.get("type", "unknown"),
                "score": round(m["score"], 3),
                "entities": json.loads(meta.get("entities", "[]")),
                "timestamp": meta.get("timestamp", ""),
            })
        return {"query": query, "results": results, "total_memories": self.daemon.store.count()}

    def stats(self) -> dict:
        if not self._started:
            return {"error": "daemon not started"}
        s = self.daemon.stats
        return {
            "session": self.daemon.session_id,
            "ingested": s["ingested"],
            "extracted": s["extracted"],
            "stored": s["stored"],
            "deduped": s["deduped"],
            "superseded": s.get("superseded", 0),
            "total_memories": s["total_memories"],
        }

    def get_insights(self) -> dict:
        """Read latest enricher insights from the vault."""
        if not self._started:
            return {"error": "daemon not started"}

        vault_path = self.daemon.vault.vault_path
        result = {"enricher_running": self.daemon.enricher is not None}

        # Enricher stats
        if self.daemon.enricher:
            result["enricher_stats"] = self.daemon.enricher.stats

        # Read relationships summary
        rel_path = os.path.join(vault_path, "memory", "relationships.md")
        if os.path.exists(rel_path):
            with open(rel_path, "r") as f:
                content = f.read()
            # Extract just entity headers and their connections (not full file)
            lines = content.split("\n")
            summary = []
            for line in lines:
                if line.startswith("## ") or (line.startswith("- ") and "shared facts" in line):
                    summary.append(line)
            result["relationships"] = "\n".join(summary[:50])  # top 50 lines

        # Read latest pattern insights
        insights_dir = os.path.join(vault_path, "memory", "insights")
        if os.path.exists(insights_dir):
            files = sorted([f for f in os.listdir(insights_dir) if f.startswith("patterns-")], reverse=True)
            if files:
                with open(os.path.join(insights_dir, files[0]), "r") as f:
                    result["patterns"] = f.read()

            # Read latest stale insights
            stale_files = sorted([f for f in os.listdir(insights_dir) if f.startswith("stale-")], reverse=True)
            if stale_files:
                with open(os.path.join(insights_dir, stale_files[0]), "r") as f:
                    result["stale"] = f.read()

        return result

    def stop(self):
        if self._started and self.daemon:
            self.daemon.stop()


# ── Metrics Tracking ───────────────────────────────────────────────────────

_metrics_history = []  # rolling window of recent inferences

def _write_metrics(prompt_tokens, completion_tokens, elapsed, gen_tps, tool_round):
    """Write inference metrics to shared JSON file for monitor dashboard."""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "elapsed_s": round(elapsed, 2),
        "gen_tok_s": gen_tps,
        "prompt_tok_s": round(prompt_tokens / elapsed, 1) if elapsed > 0 and prompt_tokens else 0,
        "tool_round": tool_round,
    }
    _metrics_history.append(entry)
    # Keep last 100 entries
    if len(_metrics_history) > 100:
        _metrics_history.pop(0)

    # Write to file for dashboard to read
    try:
        metrics = {
            "model": MLX_MODEL.split("/")[-1],
            "model_full": MLX_MODEL,
            "mlx_endpoint": MLX_BASE_URL,
            "latest": entry,
            "session_totals": {
                "total_inferences": len(_metrics_history),
                "total_prompt_tokens": sum(m["prompt_tokens"] for m in _metrics_history),
                "total_completion_tokens": sum(m["completion_tokens"] for m in _metrics_history),
                "total_tokens": sum(m["total_tokens"] for m in _metrics_history),
                "avg_gen_tok_s": round(sum(m["gen_tok_s"] for m in _metrics_history) / len(_metrics_history), 1) if _metrics_history else 0,
                "avg_prompt_tok_s": round(sum(m["prompt_tok_s"] for m in _metrics_history) / len(_metrics_history), 1) if _metrics_history else 0,
                "peak_gen_tok_s": max((m["gen_tok_s"] for m in _metrics_history), default=0),
            },
            "history": _metrics_history[-20:],  # last 20 for chart
        }
        with open(METRICS_FILE, "w") as f:
            json.dump(metrics, f)
    except (OSError, TypeError):
        pass


# ── Tool Execution ──────────────────────────────────────────────────────────

memory = MemoryBridge()
browser = BrowserBridge()

# ── Scanner Integration ───────────────────────────────────────────────────

CLAUDE_INBOX = os.path.join(VAULT_PATH, "midas/claude-inbox.md")

def scan_digest_tool(mode: str = "latest", top_n: int = 10) -> dict:
    """Get scan results from the background scanner."""
    try:
        from scanner import Scanner
        scanner = Scanner()
    except ImportError:
        return {"error": "Scanner module not available"}

    if mode == "latest":
        items = scanner.get_latest_candidates(top_n)
        return {"mode": "latest", "count": len(items), "items": items}
    elif mode == "unreviewed":
        unreviewed = scanner.get_unreviewed()
        return {"mode": "unreviewed", "scans": unreviewed, "count": len(unreviewed)}
    elif mode == "clear":
        # Process ALL unreviewed scan files, dedupe items, return top N
        unreviewed = scanner.get_unreviewed()
        all_items = []
        seen_ids = set()
        for scan_id in unreviewed:
            scan_path = os.path.join(VAULT_PATH, "midas/scans/candidates", f"{scan_id}.json")
            try:
                with open(scan_path) as f:
                    data = json.load(f)
                for source_data in data.get("sources", {}).values():
                    for item in source_data.get("items", []):
                        item_id = item.get("id", item.get("title", ""))
                        if item_id not in seen_ids:
                            seen_ids.add(item_id)
                            all_items.append(item)
            except Exception:
                continue
        # Sort by relevance
        all_items.sort(key=lambda x: x.get("relevance", 0) + x.get("score", 0) / 1000, reverse=True)
        return {
            "mode": "clear",
            "scans_processed": len(unreviewed),
            "unique_items": len(all_items),
            "top_items": all_items[:top_n],
            "note": "These are pre-scraped from HN/RSS/Reddit. Summarize the top items for the user."
        }
    elif mode == "verdicts":
        verdicts = scanner.read_verdicts()
        return {"mode": "verdicts", "count": len(verdicts), "verdicts": verdicts}
    elif mode == "stats":
        stats = scanner.get_calibration_stats()
        return {"mode": "stats", **stats}
    else:
        return {"error": f"Unknown mode: {mode}. Use: latest, unreviewed, clear, verdicts, stats"}


def message_claude_tool(message: str, priority: str = "medium", context: str = "") -> dict:
    """Write a message to Claude's inbox for review at next session."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    entry = f"\n## {timestamp}\n**Priority:** {priority}\n**From:** Midas\n\n{message}\n"
    if context:
        entry += f"\n**Context:**\n```\n{context[:2000]}\n```\n"
    entry += "\n---\n"

    # Append to inbox file
    os.makedirs(os.path.dirname(CLAUDE_INBOX), exist_ok=True)
    if not os.path.exists(CLAUDE_INBOX):
        header = "# Claude Inbox\n\nMessages from Midas for Claude to review at next session.\n\n---\n"
        with open(CLAUDE_INBOX, "w") as f:
            f.write(header)

    with open(CLAUDE_INBOX, "a") as f:
        f.write(entry)

    return {"status": "sent", "timestamp": timestamp, "priority": priority}


def execute_tool(name: str, args: dict) -> str:
    """Execute a tool call and return the result as a string."""
    # Argument validation — reject obviously bad calls
    if name == "memory_recall" and not args.get("query", "").strip():
        return json.dumps({"error": "Empty recall query. Provide a specific search term."})
    if name == "browse_search" and not args.get("query", "").strip():
        return json.dumps({"error": "Empty search query. Provide a specific query."})
    if name == "browse_navigate" and not args.get("url", "").startswith("http"):
        return json.dumps({"error": f"Invalid URL: {args.get('url', '')}. Must start with http(s)."})

    try:
        if name == "memory_ingest":
            result = memory.ingest(args.get("role", "user"), args.get("text", ""))
        elif name == "memory_recall":
            result = memory.recall(
                args.get("query", ""),
                args.get("n_results", 5),
                args.get("type_filter", ""),
            )
        elif name == "memory_stats":
            result = memory.stats()
        elif name == "memory_insights":
            result = memory.get_insights()
        elif name == "browse_navigate":
            result = browser.navigate(args.get("url", ""), args.get("wait", 2))
        elif name == "browse_read":
            result = browser.read_page(args.get("selector", "body"), args.get("max_length", 5000))
        elif name == "browse_click":
            result = browser.click(args.get("selector", ""))
        elif name == "browse_type":
            result = browser.type_text(args.get("selector", ""), args.get("text", ""))
        elif name == "browse_js":
            result = browser.run_js(args.get("expression", ""))
        elif name == "browse_search":
            result = browser.search(args.get("query", ""), args.get("max_results", 5))
        elif name == "browse_x_feed":
            result = browser.scan_x_feed(args.get("count", 5))
        elif name == "browse_tabs":
            result = {"tabs": browser.get_tabs()}
        elif name == "playbook_update":
            result = playbook_tool(args.get("section", "full"), args.get("action", "read"), args.get("content", ""))
        elif name == "vault_read":
            result = vault_read(args.get("path", ""), args.get("query", ""))
        elif name == "vault_insight":
            result = vault_insight(args.get("topic", ""), memory)
        elif name == "scan_digest":
            result = scan_digest_tool(args.get("mode", "latest"), args.get("top_n", 10))
        elif name == "message_claude":
            result = message_claude_tool(args.get("message", ""), args.get("priority", "medium"), args.get("context", ""))
        elif name == "shell":
            cmd = args.get("command", "")
            try:
                out = subprocess.run(
                    cmd, shell=True, capture_output=True, text=True, timeout=30,
                    cwd="/Users/midas/Desktop/cowork"
                )
                result = {"stdout": out.stdout[-2000:] if out.stdout else "", "stderr": out.stderr[-500:] if out.stderr else "", "returncode": out.returncode}
            except subprocess.TimeoutExpired:
                result = {"error": "command timed out (30s)"}
        else:
            result = {"error": f"unknown tool: {name}"}
    except Exception as e:
        result = {"error": str(e)}

    return json.dumps(result, indent=2) if isinstance(result, dict) else str(result)


# ── Terminal UI ─────────────────────────────────────────────────────────────

# ANSI colors
CYAN = "\033[36m"
GREEN = "\033[32m"
DIM = "\033[2m"
BOLD = "\033[1m"
RED = "\033[31m"
YELLOW = "\033[33m"
MAGENTA = "\033[35m"
RESET = "\033[0m"

def _count_files(d):
    """Recursively count files in vault structure dict."""
    for k, v in d.items():
        if isinstance(v, dict):
            yield from _count_files(v)
        else:
            yield k

TOOL_ICONS = {
    "memory_ingest": "💾",
    "memory_recall": "🔍",
    "memory_stats": "📊",
    "memory_insights": "🧠",
    "playbook_update": "📖",
    "browse_navigate": "🌐",
    "browse_read": "📄",
    "browse_click": "👆",
    "browse_type": "⌨️",
    "browse_js": "⚙️",
    "browse_search": "🔎",
    "browse_x_feed": "🐦",
    "browse_tabs": "📑",
    "vault_read": "📖",
    "vault_insight": "🔮",
    "scan_digest": "📡",
    "message_claude": "📨",
    "shell": "⚡",
}

def print_banner(mem_count=0, browser=False, ane=False):
    # Animated boot sequence
    bar_width = 32
    frames = []
    for i in range(1, 11):
        filled = int(bar_width * i / 10)
        bar = "█" * filled + "▓" * min(4, bar_width - filled) + "░" * max(0, bar_width - filled - 4)
        frames.append(f"{CYAN}  {bar[:bar_width]}{RESET}")
    for frame in frames:
        print(f"\r{frame}", end="", flush=True)
        time.sleep(0.04)

    print(f"\r{CYAN}  ╔══════════════════════════════╗{RESET}")
    print(f"{CYAN}  ║      {BOLD}◆  M I D A S  ◆{RESET}{CYAN}         ║{RESET}")
    print(f"{CYAN}  ╚══════════════════════════════╝{RESET}")
    browser_dot = f"{GREEN}●{RESET}" if browser else f"{RED}○{RESET}"
    ane_dot = f"{GREEN}●{RESET}" if ane else f"{RED}○{RESET}"
    print(f"  {DIM}{mem_count} memories │ browser {RESET}{browser_dot}{DIM} │ ane {RESET}{ane_dot}{DIM} │ dash → :{DASHBOARD_PORT}{RESET}")
    print()

def generate_briefing(stats: dict, playbook_content: str) -> str:
    """Gather data for Midas's session-opening briefing. Returns a summary string
    that gets injected as a system message so Midas can comment on it."""
    lines = []

    # 1. Memory delta — check for a last-session marker
    total = stats.get("total_memories", 0)
    marker_path = os.path.join(VAULT_PATH, "midas", ".last_session_memories")
    last_count = 0
    try:
        with open(marker_path, "r") as f:
            last_count = int(f.read().strip())
    except (FileNotFoundError, ValueError):
        pass
    delta = total - last_count
    if delta > 0:
        lines.append(f"- {delta} new memories since last session ({total} total)")
    elif last_count == 0:
        lines.append(f"- {total} memories in store (first session tracking)")
    else:
        lines.append(f"- No new memories since last session ({total} total)")

    # Update marker for next session
    try:
        os.makedirs(os.path.dirname(marker_path), exist_ok=True)
        with open(marker_path, "w") as f:
            f.write(str(total))
    except Exception:
        pass

    # 2. Enricher output — check for recent insights and relationships
    today = datetime.now().strftime("%Y-%m-%d")
    insights_dir = os.path.join(VAULT_PATH, "memory", "insights")
    if os.path.isdir(insights_dir):
        today_file = os.path.join(insights_dir, f"patterns-{today}.md")
        if os.path.exists(today_file):
            try:
                with open(today_file, "r") as f:
                    content = f.read()
                sections = re.findall(r'^## .+', content, re.MULTILINE)
                if sections:
                    lines.append(f"- Enricher produced {len(sections)} insight(s) today: {', '.join(s.replace('## ', '') for s in sections[:3])}")
            except Exception:
                pass

    # Check relationships
    rel_path = os.path.join(VAULT_PATH, "memory", "relationships.md")
    if os.path.exists(rel_path):
        try:
            with open(rel_path, "r") as f:
                rel_lines = [l for l in f if l.strip().startswith("- ")]
            if rel_lines:
                lines.append(f"- Relationship graph: {len(rel_lines)} connections mapped")
        except Exception:
            pass

    # 3. Playbook — queued improvements
    if playbook_content:
        queue_items = re.findall(r'- \[ \] (.+)', playbook_content)
        if queue_items:
            lines.append(f"- Improvement queue: {len(queue_items)} items (next: {queue_items[0].strip()[:60]})")

    # 4. Stale flags
    if os.path.isdir(insights_dir):
        stale_file = os.path.join(insights_dir, f"stale-{today}.md")
        if os.path.exists(stale_file):
            try:
                with open(stale_file, "r") as f:
                    stale_content = f.read()
                stale_count = stale_content.count("- ")
                if stale_count > 0:
                    lines.append(f"- {stale_count} potentially stale fact(s) flagged")
            except Exception:
                pass

    # 5. Enricher service status — check heartbeat and PID
    heartbeat_path = os.path.join(VAULT_PATH, "midas", ".enricher_heartbeat")
    pid_path = "/tmp/phantom-enricher.pid"
    service_running = False
    try:
        with open(pid_path, "r") as f:
            pid = int(f.read().strip())
        os.kill(pid, 0)  # Check if process exists (signal 0 = no-op)
        service_running = True
    except (FileNotFoundError, ValueError, ProcessLookupError, PermissionError):
        pass

    if service_running:
        last_beat = ""
        try:
            with open(heartbeat_path, "r") as f:
                last_beat = f.read().strip()
        except FileNotFoundError:
            pass
        if last_beat:
            lines.append(f"- Enricher service: running (last heartbeat: {last_beat})")
        else:
            lines.append(f"- Enricher service: running")
    else:
        lines.append(f"- Enricher service: not running (start with: launchctl load ~/Library/LaunchAgents/com.phantom.enricher.plist)")

    # 6. Scanner service status — check heartbeat and recent scans
    scanner_hb = os.path.join(VAULT_PATH, "midas", ".scanner_heartbeat")
    scanner_pid = "/tmp/phantom-scanner.pid"
    scanner_running = False
    try:
        with open(scanner_pid, "r") as f:
            pid = int(f.read().strip())
        os.kill(pid, 0)
        scanner_running = True
    except (FileNotFoundError, ValueError, ProcessLookupError, PermissionError):
        pass

    if scanner_running:
        last_scan = ""
        try:
            with open(scanner_hb, "r") as f:
                last_scan = f.read().strip()
        except FileNotFoundError:
            pass
        if last_scan:
            lines.append(f"- Scanner service: running (last scan: {last_scan})")
        else:
            lines.append(f"- Scanner service: running")
    else:
        lines.append(f"- Scanner service: not running (start with: launchctl load ~/Library/LaunchAgents/com.phantom.scanner.plist)")

    # Count unreviewed scan candidates
    candidates_dir = os.path.join(VAULT_PATH, "midas/scans/candidates")
    if os.path.isdir(candidates_dir):
        candidate_files = [f for f in os.listdir(candidates_dir) if f.endswith(".json")]
        if candidate_files:
            lines.append(f"- Scan candidates: {len(candidate_files)} scan(s) on file")

    # Check Claude inbox for pending messages
    claude_inbox = os.path.join(VAULT_PATH, "midas/claude-inbox.md")
    if os.path.exists(claude_inbox):
        try:
            with open(claude_inbox, "r") as f:
                inbox_content = f.read()
            msg_count = inbox_content.count("\n## ")
            if msg_count > 0:
                lines.append(f"- Claude inbox: {msg_count} message(s) pending review")
        except Exception:
            pass

    return "\n".join(lines) if lines else ""


def print_tool_call(name: str, args: dict):
    icon = TOOL_ICONS.get(name, "🔧")
    if name == "memory_ingest":
        text = args.get("text", "")[:60]
        print(f"  {DIM}{icon} storing: {text}...{RESET}")
    elif name == "memory_recall":
        print(f"  {DIM}{icon} searching: {args.get('query', '')}{RESET}")
    elif name == "memory_stats":
        print(f"  {DIM}{icon} checking stats{RESET}")
    elif name == "memory_insights":
        print(f"  {DIM}{icon} reading enricher insights{RESET}")
    elif name == "browse_navigate":
        print(f"  {DIM}{icon} → {args.get('url', '')[:60]}{RESET}")
    elif name == "browse_read":
        sel = args.get("selector", "body")
        print(f"  {DIM}{icon} reading: {sel}{RESET}")
    elif name == "browse_click":
        print(f"  {DIM}{icon} clicking: {args.get('selector', '')}{RESET}")
    elif name == "browse_type":
        print(f"  {DIM}{icon} typing into: {args.get('selector', '')}{RESET}")
    elif name == "browse_js":
        expr = args.get("expression", "")[:50]
        print(f"  {DIM}{icon} js: {expr}...{RESET}")
    elif name == "browse_search":
        print(f"  {DIM}{icon} searching: {args.get('query', '')}{RESET}")
    elif name == "browse_x_feed":
        count = args.get("count", 5)
        print(f"  {DIM}{icon} scanning X feed (top {count})...{RESET}")
    elif name == "browse_tabs":
        print(f"  {DIM}{icon} listing tabs{RESET}")
    elif name == "vault_read":
        path = args.get("path", "")
        query = args.get("query", "")
        if query:
            print(f"  {DIM}{icon} vault search: {query}{RESET}")
        elif path:
            print(f"  {DIM}{icon} vault: {path}{RESET}")
        else:
            print(f"  {DIM}{icon} listing vault{RESET}")
    elif name == "vault_insight":
        print(f"  {DIM}{icon} cross-referencing: {args.get('topic', '')[:50]}{RESET}")
    elif name == "scan_digest":
        mode = args.get("mode", "latest")
        print(f"  {DIM}{icon} scans: {mode}{RESET}")
    elif name == "message_claude":
        print(f"  {DIM}{icon} → Claude: {args.get('message', '')[:50]}{RESET}")
    elif name == "playbook_update":
        print(f"  {DIM}{icon} playbook: {args.get('action', 'read')} {args.get('section', '')}{RESET}")
    elif name == "shell":
        print(f"  {DIM}{icon} $ {args.get('command', '')[:60]}{RESET}")

def print_tool_result(name: str, result: str):
    try:
        data = json.loads(result)
    except (json.JSONDecodeError, TypeError):
        data = result

    if name == "memory_ingest" and isinstance(data, dict):
        total = data.get("total_memories", "?")
        print(f"  {DIM}  └─ stored ({total} total memories){RESET}")
    elif name == "memory_recall" and isinstance(data, dict):
        n = len(data.get("results", []))
        print(f"  {DIM}  └─ found {n} results{RESET}")
    elif name == "memory_insights" and isinstance(data, dict):
        has_patterns = "patterns" in data
        has_rel = "relationships" in data
        enricher = "running" if data.get("enricher_running") else "off"
        print(f"  {DIM}  └─ enricher {enricher} | patterns: {'✓' if has_patterns else '✗'} | relationships: {'✓' if has_rel else '✗'}{RESET}")
    elif name == "browse_navigate" and isinstance(data, dict):
        title = data.get("title", "")[:40]
        if data.get("auth_wall"):
            print(f"  {YELLOW}  └─ ⚠ AUTH WALL: {title}{RESET}")
        else:
            print(f"  {DIM}  └─ loaded: {title}{RESET}")
    elif name == "browse_read" and isinstance(data, dict):
        length = len(data.get("text", ""))
        print(f"  {DIM}  └─ {length} chars{RESET}")
    elif name == "browse_search" and isinstance(data, dict):
        featured = "featured ✓" if data.get("featured") else ""
        n = len(data.get("snippets", []))
        print(f"  {DIM}  └─ {n} results {featured}{RESET}")
    elif name == "browse_x_feed" and isinstance(data, dict):
        if data.get("auth_wall"):
            print(f"  {YELLOW}  └─ ⚠ AUTH WALL: not logged into X{RESET}")
        else:
            n = len(data.get("tweets", []))
            total = data.get("total_found", 0)
            print(f"  {DIM}  └─ {n} tweets extracted ({total} found){RESET}")
    elif name == "browse_tabs" and isinstance(data, dict):
        n = len(data.get("tabs", []))
        print(f"  {DIM}  └─ {n} tabs{RESET}")
    elif name.startswith("browse_") and isinstance(data, dict):
        status = data.get("status", data.get("error", "done"))
        print(f"  {DIM}  └─ {status}{RESET}")
    elif name == "vault_read" and isinstance(data, dict):
        if "matches" in data:
            n = len(data["matches"])
            print(f"  {DIM}  └─ {n} files matched{RESET}")
        elif "structure" in data:
            n = sum(1 for _ in _count_files(data["structure"]))
            print(f"  {DIM}  └─ {n} files in vault{RESET}")
        elif "content" in data:
            lines = data["content"].count("\n")
            print(f"  {DIM}  └─ {lines} lines{RESET}")
        elif "files" in data:
            print(f"  {DIM}  └─ {len(data['files'])} files{RESET}")
    elif name == "vault_insight" and isinstance(data, dict):
        v = len(data.get("vault_context", []))
        m = len(data.get("memory_context", []))
        xref = len(data.get("cross_references", []))
        print(f"  {DIM}  └─ {v} vault hits, {m} memories, {xref} cross-refs{RESET}")
    elif name == "shell" and isinstance(data, dict):
        rc = data.get("returncode", "?")
        print(f"  {DIM}  └─ exit {rc}{RESET}")


# ── Output Cleanup ─────────────────────────────────────────────────────────

def _truncate_repetition(text: str) -> str:
    """Detect and truncate text where the model starts repeating itself."""
    if not text or len(text) < 80:
        return text

    # Strategy 1: Find repeated 8+ word phrases (strong signal of degeneration)
    words = text.split()
    for window_size in (10, 8):
        for i in range(len(words) - window_size):
            phrase = " ".join(words[i:i + window_size]).lower()
            later_text = " ".join(words[i + window_size:]).lower()
            if phrase in later_text:
                cut_point = " ".join(words[:i + window_size])
                last_sentence = re.search(r'.*[.!?]', cut_point)
                if last_sentence and len(last_sentence.group()) > len(cut_point) // 3:
                    return last_sentence.group()
                return cut_point

    # Strategy 2: Check if any sentence start repeats (case-insensitive, first 5 words)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    if len(sentences) > 1:
        seen_starts = set()
        clean = []
        for s in sentences:
            start = " ".join(s.split()[:5]).lower()
            if start in seen_starts:
                break
            seen_starts.add(start)
            clean.append(s)
        if len(clean) < len(sentences):
            return " ".join(clean)

    return text


# ── Dynamic Tool Selection ─────────────────────────────────────────────────

# Keywords that signal browser tools are needed
_BROWSER_KEYWORDS = [
    "browse", "website", "web", "url", "http", "google", "search online",
    "x feed", "twitter", "tweet", "reddit", "navigate", "page", "click",
    "x.com", "open ", "go to ", "check the site", "look up online",
    "hacker news", "hn ",
]

def _select_tools(user_text: str, browser_available: bool, tool_round: int) -> list:
    """Select tools to offer the model based on context.
    Core tools (10) always included. Browser tools (8) only when relevant.
    This prevents the 9B model from being overwhelmed by 18 tools."""
    tools = TOOLS.copy()

    # Add browser tools if: browser is available AND (first round with browser keywords, or follow-up round already using browser)
    if browser_available:
        text_lower = user_text.lower()
        needs_browser = any(kw in text_lower for kw in _BROWSER_KEYWORDS)
        # Also include if we're in a tool chain (model already chose to browse)
        if needs_browser or tool_round > 0:
            tools.extend(BROWSER_TOOLS)

    return tools


# ── History Management ──────────────────────────────────────────────────────

def _find_safe_trim_point(messages, keep_count):
    """Find the earliest index we can trim to without orphaning tool calls.
    Returns the index into messages[] where we should start keeping."""
    # We want to keep roughly the last `keep_count` messages.
    # But we can't cut inside a tool call chain (assistant with tool_calls → tool results).
    candidate = max(1, len(messages) - keep_count)  # 1 = skip system prompt

    # Walk forward from candidate to find a safe boundary
    # Safe = not a "tool" role message (which needs its preceding assistant+tool_calls)
    for i in range(candidate, min(candidate + 10, len(messages))):
        msg = messages[i]
        role = msg.get("role", "")
        # Safe to start at: user message or assistant message WITHOUT tool_calls
        if role == "user":
            return i
        if role == "assistant" and not msg.get("tool_calls"):
            return i
    # Fallback: just use candidate
    return candidate


def _summarize_dropped(messages, start, end):
    """Build a brief text summary of dropped messages (no LLM call — just extract key content)."""
    parts = []
    for msg in messages[start:end]:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if not content:
            continue
        if role == "user":
            # Strip RAG context that was appended
            for marker in ("\n\n[Memory context]", "\n\n[Context from memory]"):
                if marker in content:
                    content = content[:content.index(marker)]
                    break
            parts.append(f"User: {content[:150]}")
        elif role == "assistant" and not msg.get("tool_calls"):
            parts.append(f"Midas: {content[:150]}")
    if not parts:
        return ""
    return "Earlier in this conversation:\n" + "\n".join(parts[-6:])  # last 6 exchanges max


def _trim_history(messages, max_count, client):
    """Trim conversation history while preserving coherence."""
    if len(messages) <= max_count:
        return messages

    trim_point = _find_safe_trim_point(messages, max_count - 2)  # -2 for system + summary

    # Summarize what we're dropping
    summary = _summarize_dropped(messages, 1, trim_point)

    # Rebuild: system prompt (with summary appended) + kept messages
    system_msg = dict(messages[0])  # copy to avoid mutating original
    if summary:
        system_msg["content"] = system_msg.get("content", "") + "\n\n" + summary
    result = [system_msg]
    result.extend(messages[trim_point:])
    return result


# ── Agent Loop ──────────────────────────────────────────────────────────────

def start_dashboard():
    """Launch the memory dashboard as a background process."""
    dashboard_path = os.path.join(os.path.dirname(__file__), "..", "memory", "dashboard.py")
    dashboard_path = os.path.abspath(dashboard_path)

    # Check if already running
    try:
        urllib.request.urlopen(f"http://localhost:{DASHBOARD_PORT}/api/stats", timeout=1)
        return None  # already running
    except (urllib.error.URLError, OSError):
        pass

    proc = subprocess.Popen(
        [sys.executable, dashboard_path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(0.5)
    return proc


def run_agent():
    # Kill all library logging — only our prints reach the terminal
    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.CRITICAL)
    logging.getLogger().setLevel(logging.CRITICAL)
    for noisy in ("httpx", "httpcore", "openai", "openai._base_client",
                   "phantom.enricher", "ane_server",
                   "sentence_transformers", "chromadb", "huggingface_hub"):
        lg = logging.getLogger(noisy)
        lg.setLevel(logging.CRITICAL)
        lg.propagate = False

    # Connect to MLX server (with timeout so boot doesn't hang if server is down)
    client = OpenAI(base_url=MLX_BASE_URL, api_key="not-needed", timeout=10.0)

    # Suppress stdout/stderr noise during boot
    logging.disable(logging.CRITICAL)
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Boot everything silently
    dashboard_proc = None
    try:
        dashboard_proc = start_dashboard()
        memory.start()
        stats = memory.stats()
    except Exception as boot_err:
        # Restore output FIRST so user sees the error
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        logging.disable(logging.NOTSET)
        print(f"\n  \033[91m✗ Boot failed:\033[0m {boot_err}")
        traceback.print_exc()
        return

    # Restore output
    sys.stdout = old_stdout
    sys.stderr = old_stderr
    logging.disable(logging.NOTSET)

    # Check ANE status
    ane_online = False
    try:
        if memory.daemon and hasattr(memory.daemon, '_ane_process') and memory.daemon._ane_process:
            ane_online = True
    except Exception:
        pass

    # Test LLM connection
    try:
        test = client.chat.completions.create(
            model=MLX_MODEL,
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=5,
        )
    except Exception as e:
        print(f"\n  {RED}✗ LLM offline — is MLX server running?{RESET}")
        print(f"  {DIM}Start with: ~/.hermes/start-mlx-server.sh{RESET}")
        return

    # Check browser — auto-launch Chrome debug if not running
    browser_online = browser.is_available()
    if not browser_online:
        try:
            chrome_bin = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
            if os.path.exists(chrome_bin):
                subprocess.Popen(
                    [chrome_bin, "--remote-debugging-port=9222",
                     "--user-data-dir=" + os.path.expanduser("~/.chrome-debug"),
                     "--no-first-run",
                     "--window-size=800,600",
                     "about:blank"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
                for _ in range(5):
                    time.sleep(1)
                    if browser.is_available():
                        browser_online = True
                        break
        except Exception:
            pass

    # Pre-connect browser so tab is ready for first tool call
    if browser_online:
        try:
            browser.connect()
        except Exception:
            pass

    # Show banner
    print_banner(mem_count=stats['total_memories'], browser=browser_online, ane=ane_online)

    # Browser tools added dynamically per-turn (see _select_tools below)
    # Too many tools at once (18) overwhelms the 9B model

    # Load playbook — Midas's self-knowledge and improvement log
    playbook_path = "/Users/midas/Desktop/cowork/vault/midas/playbook.md"
    playbook_content = ""
    try:
        with open(playbook_path, "r") as f:
            playbook_content = f.read()
    except FileNotFoundError:
        pass

    system_with_playbook = SYSTEM_PROMPT
    # Playbook available via playbook_update tool - not injected into system prompt

    # Conversation state
    messages = [{"role": "system", "content": system_with_playbook}]

    # Session briefing — Midas reports what's changed since last session
    briefing_data = generate_briefing(stats, playbook_content)
    if briefing_data:
        print(f"  {CYAN}┌─ Session Briefing ─────────────────────────{RESET}")
        for line in briefing_data.split("\n"):
            print(f"  {CYAN}│{RESET} {DIM}{line}{RESET}")
        print(f"  {CYAN}└─────────────────────────────────────────────{RESET}")
        print()

        # Append briefing to system prompt so model has context without a fake user message
        messages[0]["content"] += f"\n\n## SESSION STATUS\n{briefing_data}"

    while True:
        # Get user input
        try:
            user_input = input(f"{GREEN}▸ {RESET}")
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input.strip():
            continue

        # Special commands
        if user_input.strip().lower() in ("/quit", "/exit", "/q"):
            break
        if user_input.strip().lower() == "/stats":
            s = memory.stats()
            print(f"  {CYAN}Memory: {s['total_memories']} facts | Session: {s['extracted']} extracted, {s['stored']} stored, {s['deduped']} deduped, {s['superseded']} superseded{RESET}")
            continue
        if user_input.strip().lower() == "/clear":
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            print(f"  {DIM}Conversation cleared (memory preserved){RESET}")
            continue
        if user_input.strip().lower() == "/help":
            print(f"""  {CYAN}Commands:{RESET}
  {DIM}/stats  — Memory statistics
  /clear  — Clear conversation (memory preserved)
  /quit   — Exit
  /help   — This message{RESET}""")
            continue

        # Add user message
        messages.append({"role": "user", "content": user_input})

        # Auto-ingest every user message (daemon filters noise)
        memory.ingest("user", user_input)

        # RAG injection disabled — 9B can't reliably reason about injected context.
        # Re-enable when running 70B on Pro. Memory still available via memory_recall tool.

        # Agent loop — keep going until model stops calling tools
        tool_rounds = 0
        recent_tool_calls = []  # track for loop detection
        while tool_rounds < MAX_TOOL_ROUNDS:
            # Use low temperature for all rounds — 9B needs it for reliable tool format
            temp = TOOL_TEMPERATURE
            # Select tools dynamically — core always, browser only when relevant
            turn_tools = _select_tools(user_input, browser_online, tool_rounds)
            try:
                t0 = time.time()
                response = client.chat.completions.create(
                    model=MLX_MODEL,
                    messages=messages,
                    tools=turn_tools,
                    tool_choice="auto",
                    max_tokens=2048,
                    temperature=temp,
                )
                elapsed = time.time() - t0
            except Exception as e:
                print(f"  {RED}Error: {e}{RESET}")
                break

            choice = response.choices[0]
            msg = choice.message

            # Detect degenerate tool call format (model outputting raw XML instead of function calls)
            if msg.content and not msg.tool_calls and any(t in msg.content for t in ["<tool_call>", "<function=", "```tool_call"]):
                # Silent retry — expected behavior on 9B, no need to alarm the user
                # Don't add this broken response to history, just retry at minimum temp
                try:
                    t0 = time.time()
                    response = client.chat.completions.create(
                        model=MLX_MODEL,
                        messages=messages,
                        tools=turn_tools,
                        tool_choice="auto",
                        max_tokens=2048,
                        temperature=0.1,
                    )
                    elapsed = time.time() - t0
                    choice = response.choices[0]
                    msg = choice.message
                except Exception:
                    pass  # fall through to normal handling

            # Track inference metrics
            usage = getattr(response, 'usage', None)
            prompt_tokens = getattr(usage, 'prompt_tokens', 0) if usage else 0
            completion_tokens = getattr(usage, 'completion_tokens', 0) if usage else 0
            gen_tps = round(completion_tokens / elapsed, 1) if elapsed > 0 and completion_tokens else 0
            _write_metrics(prompt_tokens, completion_tokens, elapsed, gen_tps, tool_rounds)

            # Check for tool calls
            if msg.tool_calls:
                # Add assistant message with tool calls
                messages.append({
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            }
                        }
                        for tc in msg.tool_calls
                    ]
                })

                # Execute each tool call
                for tc in msg.tool_calls:
                    name = tc.function.name
                    try:
                        args = json.loads(tc.function.arguments)
                    except (json.JSONDecodeError, TypeError):
                        args = {}

                    print_tool_call(name, args)
                    result = execute_tool(name, args)
                    print_tool_result(name, result)

                    # Add tool result
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    })

                tool_rounds += 1

                # Loop detection — same call 3x OR alternating pattern (A-B-A-B)
                for tc in msg.tool_calls:
                    call_sig = f"{tc.function.name}:{tc.function.arguments}"
                    recent_tool_calls.append(call_sig)
                if len(recent_tool_calls) >= 3:
                    last_3 = recent_tool_calls[-3:]
                    # Same call 3x
                    is_loop = last_3[0] == last_3[1] == last_3[2]
                    # Alternating: A-B-A
                    if not is_loop and last_3[0] == last_3[2] and last_3[0] != last_3[1]:
                        is_loop = True
                    if is_loop:
                        print(f"  {YELLOW}⚠ Tool loop detected — stopping{RESET}")
                        messages.append({"role": "assistant", "content": "I'm going in circles. What specifically would you like me to do?"})
                        break

                continue  # let model process tool results

            else:
                # No tool calls — print response and break
                content = msg.content or ""
                if content:
                    # Clean up any think tags from Qwen (full, partial, or bare)
                    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
                    content = re.sub(r'<think>.*$', '', content, flags=re.DOTALL)
                    content = re.sub(r'</?think>', '', content)
                    content = content.strip()
                    # Detect and truncate repetition degeneration
                    content = _truncate_repetition(content)

                if content:
                    print(f"\n{CYAN}{content}{RESET}\n")
                    if tool_rounds > 0:
                        memory.ingest("assistant", content)
                elif tool_rounds > 0:
                    # Model used tools but produced empty response — nudge it
                    print(f"  {DIM}(processing...){RESET}")
                    messages.append({"role": "assistant", "content": ""})
                    messages.append({"role": "user", "content": "Summarize what you found in 2-3 sentences."})
                    tool_rounds += 1
                    continue

                messages.append({"role": "assistant", "content": content or ""})
                break

        # Trim history if too long — respect tool call boundaries
        if len(messages) > MAX_HISTORY:
            messages = _trim_history(messages, MAX_HISTORY, client)

    # Shutdown — suppress noisy thread/resource cleanup
    threading.excepthook = lambda args: None

    print(f"  {DIM}Shutting down...{RESET}")
    memory.stop()
    browser.disconnect()
    if dashboard_proc:
        print(f"  {DIM}Dashboard still running at http://localhost:{DASHBOARD_PORT} (Ctrl+C to stop){RESET}")
    print(f"  {GREEN}Session saved. Goodbye.{RESET}")

    # Close stderr at the OS level — prevents resource_tracker child process
    # from printing semaphore warnings after we exit
    sys.stdout.flush()
    try:
        os.close(2)  # close fd 2 (stderr) so child processes inherit a dead fd
    except OSError:
        pass
    os._exit(0)


# ── Entry Point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Handle Ctrl+C gracefully
    signal.signal(signal.SIGINT, lambda *_: None)  # let input() handle it

    run_agent()
