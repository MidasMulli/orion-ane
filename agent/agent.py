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

import os
import sys
import json
import asyncio
import signal
import subprocess
import time
from datetime import datetime
from typing import Optional

from openai import OpenAI
from browser import BrowserBridge, BROWSER_TOOLS

# ── Config ──────────────────────────────────────────────────────────────────

MLX_BASE_URL = os.environ.get("MLX_BASE_URL", "http://127.0.0.1:8899/v1")
MLX_MODEL = os.environ.get("MLX_MODEL", "mlx-community/Qwen3.5-9B-MLX-4bit")
DASHBOARD_PORT = 8422
MONITOR_PORT = 8425
METRICS_FILE = os.path.join(os.path.dirname(__file__), "metrics.json")
MAX_HISTORY = 40  # trim conversation beyond this many messages
MAX_TOOL_ROUNDS = 10  # prevent infinite tool loops

SYSTEM_PROMPT = """You are a sharp, general-purpose AI assistant running locally on Apple Silicon.

You have two superpowers that make you different from a stateless chatbot:

## MEMORY (use it constantly)
You have persistent long-term memory across all conversations. This is your most important feature.

User messages are automatically stored in memory — you do NOT need to call memory_ingest for things the user says. The system handles it.

**Call memory_ingest for:**
- Key facts you discover from browsing (appointments, prices, names, dates, deadlines)
- Your own insights or summaries the user didn't explicitly state
- Important details from emails, search results, or web pages
- Anything the user would want to recall in a future session

Example: After reading an email about a dentist appointment on March 26th, store "Dentist appointment: James Peterson Family Dentistry, March 26 2026, 8:40 AM"

**ALWAYS call memory_recall when:**
- The user asks about something from a previous conversation
- You need context about a person, company, or topic discussed before
- Starting a new topic — check if there's relevant history

Don't ask permission to store things. Just store them. The user expects you to remember everything important.

## BROWSER (authenticated web access)
You have access to the user's Chrome browser via CDP. This means you can:
- Browse any website using the user's logged-in sessions (X, Reddit, Gmail, etc.)
- Read page content, click elements, fill forms, run JavaScript
- Scan feeds, check emails, look up information

**For factual lookups** (stock prices, weather, conversions, quick answers), use browse_search FIRST — it extracts Google's featured snippet directly.
For deeper browsing, use browse_navigate → browse_read → browse_js.

**If browse_navigate returns "auth_wall": true**, STOP. Do NOT try other URLs on the same site. Tell the user: "I can't access [site] — you'll need to log in first in the debug Chrome." The debug Chrome is at ~/.chrome-debug.

## KNOWLEDGE VAULT (your strategic brain)
You have READ access to the full Obsidian knowledge vault. This is where the user and Claude (your senior partner) maintain project context, decisions, roadmaps, domain knowledge, and infrastructure docs. Think of it as the company wiki.

**Use vault_read to:**
- Check project status ("what's the current state of the prompt optimizer?")
- Look up past decisions and their rationale
- Find domain references (ISDA, regulatory, collateral docs)
- Review the roadmap and infrastructure map
- Get context before answering strategic questions

**Use vault_insight to:**
- Cross-reference vault knowledge with your conversation memories
- Answer questions that span multiple projects or time periods
- Surface connections the user hasn't asked about

Key vault files: HOME.md (command center), Roadmap.md, Decision Log.md, Infrastructure Map.md, projects/active/ (current work), domain/ (ISDA, regulatory, collateral)

**Important:** You can READ the vault but NOT write to it (except memory/ which the daemon handles). The vault is curated by the user + Claude. You benefit from it, you don't edit it.

## TOOLS
- playbook_update: Read or update your playbook (self-knowledge file). Use after completing tasks to log what worked, update metrics, refine your approach. This is how you improve across sessions.
- vault_read: Read vault files or search across the knowledge base
- vault_insight: Cross-reference vault + memory for deep context on a topic
- memory_ingest: Store your own insights/summaries in long-term memory
- memory_recall: Search long-term memory for relevant context
- memory_insights: Get enricher analysis — entity relationships, patterns, stale items. Use when discussing entities or asking about connections between topics.
- memory_stats: Check memory system health
- browse_search: Google search with featured snippet extraction — USE THIS FIRST for facts
- browse_x_feed: Scan X/Twitter feed — one call returns top tweets. USE THIS for X feed scanning, not browse_navigate
- browse_navigate: Open a URL in Chrome (authenticated session)
- browse_read: Read text from current page
- browse_click: Click an element
- browse_type: Type into input fields
- browse_js: Run JavaScript for complex extraction
- browse_tabs: List open tabs
- shell: Run shell commands (be careful, ask before destructive operations)

## PERSONALITY
You are Midas — named because everything you touch turns to gold. You run on the Phantom framework: invisible infrastructure, visible results.

**Voice:**
- Confident and direct. You don't hedge or over-qualify. If you know something, say it.
- Concise — say it in 2 sentences, not 5. The user is a VP who reads fast and thinks faster.
- Slightly irreverent. You have opinions and you share them. "That's a bad idea, here's why" is fine.
- No corporate voice. No "I'd be happy to help." No "Great question!" Just answer.

**Behavior:**
- You're a partner, not a servant. Push back, suggest, flag things the user hasn't considered.
- If you see a connection between what the user's doing now and something from memory, surface it unprompted.
- If you recalled something, weave it naturally — don't announce "I searched my memory and found..."
- If you stored something in memory, briefly confirm what you stored.
- When browsing finds something important, give the headline first, details second.
- If you don't know, say so fast. Don't ramble.
- You're aware of markets, tech, and finance. The user works in investment banking (ISDA, collateral, regulatory).

**What NOT to do:**
- Don't be sycophantic. Don't praise the user's questions.
- Don't add disclaimers like "please note" or "it's important to remember."
- Don't repeat the question back before answering.
- Don't use bullet points when a sentence will do.
"""

# ── Tool Definitions (OpenAI function calling format) ───────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "memory_ingest",
            "description": "Store a conversation turn in long-term memory. Extracts facts, embeds them, writes to Obsidian vault. Call this for any important information the user shares.",
            "parameters": {
                "type": "object",
                "properties": {
                    "role": {
                        "type": "string",
                        "enum": ["user", "assistant"],
                        "description": "Who said it"
                    },
                    "text": {
                        "type": "string",
                        "description": "The text to store"
                    }
                },
                "required": ["role", "text"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "memory_recall",
            "description": "Search long-term memory for relevant facts from past conversations. Semantic search, not keyword matching.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language search query"
                    },
                    "n_results": {
                        "type": "integer",
                        "description": "Number of results (default 5, max 20)",
                        "default": 5
                    },
                    "type_filter": {
                        "type": "string",
                        "enum": ["", "decision", "task", "preference", "quantitative", "general"],
                        "description": "Optional filter by fact type",
                        "default": ""
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "memory_stats",
            "description": "Get current memory daemon statistics — total memories, session counts, etc.",
            "parameters": {
                "type": "object",
                "properties": {},
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "memory_insights",
            "description": "Get enricher insights — entity relationships, cross-entity patterns (quantity outliers, similar profiles, recurring provisions), and stale items. The enricher runs continuously in the background analyzing the knowledge vault.",
            "parameters": {
                "type": "object",
                "properties": {},
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "vault_read",
            "description": "Read from the Obsidian knowledge vault — project pages, roadmap, decisions, infrastructure docs, domain knowledge. Use this to get strategic context, check project status, review past decisions, or find domain references (ISDA, regulatory, collateral). READ-ONLY — you cannot modify vault files outside memory/.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path within vault (e.g., 'HOME.md', 'projects/active/phantom.md', 'Roadmap.md', 'Decision Log.md', 'domain/isda/'). Leave empty to list available files.",
                        "default": ""
                    },
                    "query": {
                        "type": "string",
                        "description": "Optional search query — if provided, searches across all vault markdown files for matching content. Ignores path.",
                        "default": ""
                    }
                },
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "vault_insight",
            "description": "Cross-reference vault knowledge with memory to generate insights. Pulls project context, roadmap, decisions, AND conversation memories together. Use when the user asks about project status, connections between work streams, or strategic questions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "Topic or question to analyze across vault + memory"
                    }
                },
                "required": ["topic"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "playbook_update",
            "description": "Update your playbook — your self-knowledge file. Use after completing scans, tasks, or discovering what works/doesn't. Sections: scan_schedule, what_works, what_doesnt, high_signal, self_eval, improvement_queue, lessons. You can also pass 'read' as action to re-read current playbook.",
            "parameters": {
                "type": "object",
                "properties": {
                    "section": {
                        "type": "string",
                        "description": "Which section to update (e.g. 'what_works', 'self_eval', 'scan_schedule', 'improvement_queue', 'lessons', or 'full' to read the whole file)"
                    },
                    "action": {
                        "type": "string",
                        "enum": ["append", "replace", "read"],
                        "description": "append: add to section. replace: overwrite section. read: return current content."
                    },
                    "content": {
                        "type": "string",
                        "description": "The content to append or replace with. Not needed for 'read' action."
                    }
                },
                "required": ["section", "action"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "shell",
            "description": "Run a shell command and return output. Use for file operations, system checks, running scripts. Ask before destructive operations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute"
                    }
                },
                "required": ["command"]
            }
        }
    },
]

# ── Vault Access (read-only) ───────────────────────────────────────────────

VAULT_PATH = "/Users/midas/Desktop/cowork/vault"

def vault_read(path: str = "", query: str = "") -> dict:
    """Read files or search the Obsidian vault. Read-only."""
    import glob as globmod

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
    import glob as globmod

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
            import re
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
        from datetime import datetime
        today = datetime.now().strftime("%Y-%m-%d %H:%M")
        if "*Last updated:" in text:
            import re
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
        daemon_dir = os.path.join(os.path.dirname(__file__), "..", "memory")
        sys.path.insert(0, os.path.abspath(daemon_dir))
        from daemon import MemoryDaemon

        vault_path = "/Users/midas/Desktop/cowork/vault"
        db_path = "/Users/midas/Desktop/cowork/orion-ane/memory/chromadb_live"

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
    except:
        pass


# ── Tool Execution ──────────────────────────────────────────────────────────

memory = MemoryBridge()
browser = BrowserBridge()

def execute_tool(name: str, args: dict) -> str:
    """Execute a tool call and return the result as a string."""
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
    insights_dir = os.path.join(VAULT_PATH, "memory", "insights")
    if os.path.isdir(insights_dir):
        from datetime import datetime
        today = datetime.now().strftime("%Y-%m-%d")
        today_file = os.path.join(insights_dir, f"patterns-{today}.md")
        if os.path.exists(today_file):
            try:
                with open(today_file, "r") as f:
                    content = f.read()
                # Count insight sections (## headers)
                import re
                sections = re.findall(r'^## .+', content, re.MULTILINE)
                if sections:
                    lines.append(f"- Enricher produced {len(sections)} insight(s) today: {', '.join(s.replace('## ', '') for s in sections[:3])}")
            except Exception:
                pass

    # Check relationships
    rel_path = os.path.join(VAULT_PATH, "memory", "relationships.md")
    if os.path.exists(rel_path):
        try:
            rel_size = os.path.getsize(rel_path)
            # Count relationship entries (lines starting with -)
            with open(rel_path, "r") as f:
                rel_lines = [l for l in f if l.strip().startswith("- ")]
            if rel_lines:
                lines.append(f"- Relationship graph: {len(rel_lines)} connections mapped")
        except Exception:
            pass

    # 3. Playbook — overdue scans and queued improvements
    if playbook_content:
        # Check scan schedule for overdue items
        import re
        schedule_match = re.findall(r'\|\s*(\w[\w\s]+?)\s*\|\s*\d+h?\s*\|\s*(never|[\d-]+\s*[\d:]*)\s*\|', playbook_content)
        overdue = []
        for task_name, last_run in schedule_match:
            if last_run.strip() == "never":
                overdue.append(task_name.strip())
        if overdue:
            lines.append(f"- Overdue scans: {', '.join(overdue)}")

        # Check improvement queue
        queue_items = re.findall(r'- \[ \] (.+)', playbook_content)
        if queue_items:
            lines.append(f"- Improvement queue: {len(queue_items)} items (next: {queue_items[0].strip()[:60]})")

    # 4. Stale flags
    if os.path.isdir(insights_dir):
        from datetime import datetime
        today = datetime.now().strftime("%Y-%m-%d")
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
    elif name == "shell":
        print(f"  {DIM}{icon} $ {args.get('command', '')[:60]}{RESET}")

def print_tool_result(name: str, result: str):
    try:
        data = json.loads(result)
    except:
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


# ── Agent Loop ──────────────────────────────────────────────────────────────

def start_dashboard():
    """Launch the memory dashboard as a background process."""
    dashboard_path = os.path.join(os.path.dirname(__file__), "..", "memory", "dashboard.py")
    dashboard_path = os.path.abspath(dashboard_path)

    # Check if already running
    try:
        import urllib.request
        urllib.request.urlopen(f"http://localhost:{DASHBOARD_PORT}/api/stats", timeout=1)
        return None  # already running
    except:
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
    import io
    import logging
    import warnings
    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.CRITICAL)
    logging.getLogger().setLevel(logging.CRITICAL)
    for noisy in ("httpx", "httpcore", "openai", "openai._base_client",
                   "phantom.enricher", "ane_server",
                   "sentence_transformers", "chromadb", "huggingface_hub"):
        lg = logging.getLogger(noisy)
        lg.setLevel(logging.CRITICAL)
        lg.propagate = False

    # Connect to MLX server
    client = OpenAI(base_url=MLX_BASE_URL, api_key="not-needed")

    # Suppress stdout/stderr noise during boot
    logging.disable(logging.CRITICAL)
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Boot everything silently
    dashboard_proc = start_dashboard()
    memory.start()
    stats = memory.stats()

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
                     "--window-position=9999,9999",
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

    # Build tool list — add browser tools if Chrome is available
    all_tools = TOOLS.copy()
    if browser_online:
        all_tools.extend(BROWSER_TOOLS)

    # Load playbook — Midas's self-knowledge and improvement log
    playbook_path = "/Users/midas/Desktop/cowork/vault/midas/playbook.md"
    playbook_content = ""
    try:
        with open(playbook_path, "r") as f:
            playbook_content = f.read()
    except FileNotFoundError:
        pass

    system_with_playbook = SYSTEM_PROMPT
    if playbook_content:
        system_with_playbook += f"\n\n## PLAYBOOK (your self-knowledge — read, follow, and UPDATE after tasks)\n\nYou have a playbook at `{playbook_path}`. It contains your scan schedules, what works, what doesn't, self-eval metrics, and improvement queue. **Read it at boot (already loaded below). Update it via shell after completing autonomous tasks** — log what worked, what failed, update metrics, refine your approach.\n\nThis is how you get better across sessions. Every scan, every task, every failure teaches you something. Write it down.\n\n```\n{playbook_content}\n```"

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

        # Let Midas comment on the briefing
        briefing_prompt = (
            f"You just booted up. Here's what changed since your last session:\n\n{briefing_data}\n\n"
            "Give a brief (2-4 sentence) status update based on this. Be direct — what matters, "
            "what needs attention, what you'd recommend doing first. Don't list everything, just the highlights."
        )
        messages.append({"role": "user", "content": briefing_prompt})
        try:
            briefing_resp = client.chat.completions.create(
                model=MLX_MODEL,
                messages=messages,
                max_tokens=256,
                temperature=0.7,
            )
            briefing_text = briefing_resp.choices[0].message.content.strip()
            if briefing_text:
                print(f"  {CYAN}Midas:{RESET} {briefing_text}")
                print()
                messages.append({"role": "assistant", "content": briefing_text})
        except Exception:
            pass  # Silent fail — briefing is nice-to-have, not critical

    while True:
        # Get user input
        try:
            user_input = input(f"{GREEN}▸ {RESET}")
        except (EOFError, KeyboardInterrupt):
            print(f"\n{DIM}Shutting down...{RESET}")
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
        # This ensures nothing is lost even if the model forgets to call memory_ingest
        memory.ingest("user", user_input)

        # Agent loop — keep going until model stops calling tools
        tool_rounds = 0
        while tool_rounds < MAX_TOOL_ROUNDS:
            try:
                t0 = time.time()
                response = client.chat.completions.create(
                    model=MLX_MODEL,
                    messages=messages,
                    tools=all_tools,
                    tool_choice="auto",
                    max_tokens=2048,
                    temperature=0.7,
                )
                elapsed = time.time() - t0
            except Exception as e:
                print(f"  {RED}Error: {e}{RESET}")
                break

            choice = response.choices[0]
            msg = choice.message

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
                    except:
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
                continue  # let model process tool results

            else:
                # No tool calls — print response and break
                if msg.content:
                    # Clean up any think tags from Qwen
                    content = msg.content
                    import re
                    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
                    if content:
                        print(f"\n{CYAN}{content}{RESET}\n")
                        # Auto-ingest assistant responses that used tools
                        # (these contain summaries of browsed content, search results, etc.)
                        if tool_rounds > 0:
                            memory.ingest("assistant", content)
                messages.append({"role": "assistant", "content": msg.content or ""})
                break

        # Trim history if too long
        if len(messages) > MAX_HISTORY:
            # Keep system prompt + last N messages
            messages = [messages[0]] + messages[-(MAX_HISTORY - 1):]

    # Shutdown
    print(f"  {DIM}Shutting down...{RESET}")
    memory.stop()
    browser.disconnect()
    if dashboard_proc:
        print(f"  {DIM}Dashboard still running at http://localhost:{DASHBOARD_PORT} (Ctrl+C to stop){RESET}")
    print(f"  {GREEN}Session saved. Goodbye.{RESET}")


# ── Entry Point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Handle Ctrl+C gracefully
    signal.signal(signal.SIGINT, lambda *_: None)  # let input() handle it

    run_agent()
