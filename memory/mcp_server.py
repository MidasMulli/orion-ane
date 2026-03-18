#!/usr/bin/env python3
"""
Memory Daemon MCP Server
=========================

Exposes the Memory Daemon as an MCP tool server over stdio.
Any MCP-compatible client (Hermes, Claude Code, etc.) can connect.

Tools:
  memory_ingest  — Store a conversation turn (extracts facts, embeds, writes to vault)
  memory_recall  — Semantic search across all stored memories
  memory_stats   — Current daemon statistics
  memory_entities — List all known entities in the knowledge graph
  memory_decisions — List all recorded decisions
  memory_tasks   — List all recorded tasks

Usage (standalone):
  python mcp_server.py

Hermes config (add to ~/.hermes/config.yaml):
  mcp_servers:
    memory:
      command: /Users/midas/.mlx-env/bin/python3
      args:
        - /Users/midas/Desktop/cowork/orion-ane/memory/mcp_server.py
      env:
        PATH: /opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin
        HOME: /Users/midas
"""

import os
import sys
import json
import logging

from mcp.server.fastmcp import FastMCP

# Import daemon components
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from daemon import MemoryDaemon

# ── Config ──
VAULT_PATH = os.environ.get("MEMORY_VAULT_PATH", "/Users/midas/Desktop/cowork/vault")
DB_PATH = os.environ.get("MEMORY_DB_PATH", "/Users/midas/Desktop/cowork/orion-ane/memory/chromadb_live")

# ── Logging (stderr only — stdout is MCP protocol) ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [memory-mcp] %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger("memory-mcp")

# ── Initialize daemon ──
log.info(f"Starting Memory Daemon (vault={VAULT_PATH}, db={DB_PATH})")
daemon = MemoryDaemon(vault_path=VAULT_PATH, db_path=DB_PATH)
daemon.start()
log.info(f"Daemon started (session={daemon.session_id}, existing_memories={daemon.stats['total_memories']})")

# ── MCP Server ──
mcp = FastMCP(
    "memory-daemon",
    instructions="Persistent memory system for local LLMs. Extracts facts from conversations, "
                 "embeds them in a vector store, and organizes them in an Obsidian vault. "
                 "Call memory_ingest after important conversation turns. "
                 "Call memory_recall to retrieve relevant context from past sessions.",
)


@mcp.tool()
def memory_ingest(role: str, text: str, source_agent: str = "hermes") -> str:
    """Store a conversation turn in long-term memory.

    Extracts atomic facts (decisions, tasks, preferences, quantities, entities),
    embeds them in ChromaDB, and writes organized notes to the Obsidian vault.

    Call this after conversation turns that contain important information worth
    remembering across sessions — decisions made, facts stated, preferences
    expressed, tasks assigned, or quantitative data mentioned.

    Do NOT call for greetings, filler, or conversational noise.

    Args:
        role: Who said it — "user" or "assistant"
        text: The full text of the conversation turn
        source_agent: Which agent is ingesting — "hermes", "claude", etc.

    Returns:
        Summary of what was extracted and stored.
    """
    daemon.ingest(role, text)

    # Wait briefly for background processing
    import time
    time.sleep(0.5)

    stats = daemon.stats
    return json.dumps({
        "status": "ingested",
        "source_agent": source_agent,
        "session": daemon.session_id,
        "total_extracted": stats["extracted"],
        "total_stored": stats["stored"],
        "total_deduped": stats["deduped"],
        "total_memories": stats["total_memories"],
    })


@mcp.tool()
def memory_recall(query: str, n_results: int = 5, type_filter: str = "",
                  include_superseded: bool = False) -> str:
    """Search long-term memory for relevant facts from past conversations.

    Uses semantic similarity (not keyword matching) to find the most relevant
    stored facts. Returns results ranked by a combined score of relevance
    and recency. Superseded facts (contradicted by newer information) are
    filtered out by default.

    Use this at the start of a conversation to load context, or mid-conversation
    when the user references something from a previous session.

    Args:
        query: Natural language search query (e.g. "cross-default threshold for Counterparty X")
        n_results: Number of results to return (default 5, max 20)
        type_filter: Optional — filter by fact type: "decision", "task", "preference", "quantitative", "general"
        include_superseded: If True, include old/contradicted facts (default False)

    Returns:
        Ranked list of matching memories with similarity scores and metadata.
    """
    n_results = min(max(1, n_results), 20)
    filter_val = type_filter if type_filter in ("decision", "task", "preference", "quantitative", "general") else None

    memories = daemon.store.recall(query, n_results=n_results, type_filter=filter_val,
                                   include_superseded=include_superseded)

    results = []
    for m in memories:
        meta = m["metadata"]
        entry = {
            "text": m["text"],
            "type": meta.get("type", "unknown"),
            "score": round(m["score"], 3),
            "similarity": round(m["similarity"], 3),
            "recency": m.get("recency", 0),
            "source_role": meta.get("source_role", "unknown"),
            "session": meta.get("session", "unknown"),
            "timestamp": meta.get("timestamp", ""),
            "entities": json.loads(meta.get("entities", "[]")),
        }
        if m.get("superseded"):
            entry["superseded"] = True
            entry["superseded_by"] = meta.get("superseded_by", "")
        results.append(entry)

    return json.dumps({
        "query": query,
        "n_results": len(results),
        "total_memories": daemon.store.count(),
        "results": results,
    }, indent=2)


@mcp.tool()
def memory_stats() -> str:
    """Get current memory daemon statistics.

    Returns counts of ingested turns, extracted facts, stored facts,
    deduplicated facts, and total memories in the vector store.
    """
    stats = daemon.stats
    return json.dumps({
        "session_id": daemon.session_id,
        "ingested_turns": stats["ingested"],
        "extracted_facts": stats["extracted"],
        "stored_facts": stats["stored"],
        "deduped_facts": stats["deduped"],
        "superseded_facts": stats.get("superseded", 0),
        "total_memories": stats["total_memories"],
        "vault_path": VAULT_PATH,
        "db_path": DB_PATH,
    }, indent=2)


@mcp.tool()
def memory_entities() -> str:
    """List all known entities in the knowledge graph.

    Returns the list of entity pages that exist in the Obsidian vault,
    showing what the memory system has learned about over time.
    """
    entities_dir = os.path.join(VAULT_PATH, "memory", "entities")
    if not os.path.exists(entities_dir):
        return json.dumps({"entities": [], "count": 0})

    entities = []
    for f in sorted(os.listdir(entities_dir)):
        if f.endswith(".md"):
            filepath = os.path.join(entities_dir, f)
            name = f.replace(".md", "").replace("-", " ")
            # Count facts on this entity page
            with open(filepath, "r") as fh:
                lines = [l for l in fh.readlines() if l.startswith("- [")]
            entities.append({
                "name": name,
                "file": f,
                "fact_count": len(lines),
            })

    return json.dumps({"entities": entities, "count": len(entities)}, indent=2)


@mcp.tool()
def memory_decisions() -> str:
    """List all recorded decisions from past conversations.

    Useful for checking what has been decided previously before making
    new recommendations or revisiting topics.
    """
    filepath = os.path.join(VAULT_PATH, "memory", "decisions", "decisions.md")
    if not os.path.exists(filepath):
        return json.dumps({"decisions": [], "count": 0})

    with open(filepath, "r") as fh:
        lines = [l.strip() for l in fh.readlines() if l.startswith("- [")]

    decisions = []
    for line in lines:
        # Parse "- [2026-03-17] text..."
        if "] " in line:
            date_part = line.split("] ")[0].replace("- [", "")
            text_part = "] ".join(line.split("] ")[1:])
            decisions.append({"date": date_part, "text": text_part})

    return json.dumps({"decisions": decisions, "count": len(decisions)}, indent=2)


@mcp.tool()
def memory_tasks() -> str:
    """List all recorded tasks and action items from past conversations.

    Useful for checking outstanding work items and deadlines.
    """
    filepath = os.path.join(VAULT_PATH, "memory", "tasks", "tasks.md")
    if not os.path.exists(filepath):
        return json.dumps({"tasks": [], "count": 0})

    with open(filepath, "r") as fh:
        lines = [l.strip() for l in fh.readlines() if l.startswith("- [")]

    tasks = []
    for line in lines:
        if "] " in line:
            date_part = line.split("] ")[0].replace("- [", "")
            text_part = "] ".join(line.split("] ")[1:])
            tasks.append({"date": date_part, "text": text_part})

    return json.dumps({"tasks": tasks, "count": len(tasks)}, indent=2)


# ── Cleanup on exit ──
import atexit

def _shutdown():
    log.info("Shutting down memory daemon...")
    daemon.stop()
    log.info("Session summary written. Goodbye.")

atexit.register(_shutdown)


# ── Run ──
if __name__ == "__main__":
    log.info("Memory MCP server starting on stdio...")
    mcp.run(transport="stdio")
