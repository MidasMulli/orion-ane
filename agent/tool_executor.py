"""
Layer 3: Tool Executor.

Takes (tool_name, args_dict) from the router and executes it.
No LLM calls. Pure function dispatch.

Returns a plain-text result string ready to hand to the synthesizer.
"""

import json
import os
import subprocess
import sys
from datetime import datetime

# ── Dependencies from existing agent infrastructure ──────────────────────────

VAULT_PATH = "/Users/midas/Desktop/cowork/vault"
PLAYBOOK_PATH = os.path.join(VAULT_PATH, "midas/playbook.md")
CLAUDE_INBOX = os.path.join(VAULT_PATH, "midas/claude-inbox.md")

# Lazy-init singletons (set by agent.py at boot)
_memory = None
_browser = None


def set_memory(bridge):
    global _memory
    _memory = bridge


def set_browser(bridge):
    global _browser
    _browser = bridge


# ── Vault (read-only) ───────────────────────────────────────────────────────

def _vault_read(path: str = "", query: str = "") -> dict:
    import glob as globmod
    if query:
        matches = []
        for md_file in globmod.glob(os.path.join(VAULT_PATH, "**/*.md"), recursive=True):
            rel = os.path.relpath(md_file, VAULT_PATH)
            try:
                with open(md_file, "r") as f:
                    content = f.read()
                if query.lower() in content.lower():
                    lines = content.split("\n")
                    snippets = []
                    for i, line in enumerate(lines):
                        if query.lower() in line.lower():
                            start = max(0, i - 1)
                            end = min(len(lines), i + 2)
                            snippets.append("\n".join(lines[start:end]))
                    matches.append({"file": rel, "snippets": snippets[:3]})
            except Exception:
                continue
        return {"query": query, "matches": matches[:10]}

    if not path:
        structure = {}
        for md_file in sorted(globmod.glob(os.path.join(VAULT_PATH, "**/*.md"), recursive=True)):
            rel = os.path.relpath(md_file, VAULT_PATH)
            if rel.startswith("memory/"):
                continue
            parts = rel.split("/")
            d = structure
            for p in parts[:-1]:
                d = d.setdefault(p, {})
            d[parts[-1]] = os.path.getsize(md_file)
        return {"vault_path": VAULT_PATH, "structure": structure}

    full_path = os.path.join(VAULT_PATH, path)
    if os.path.isdir(full_path):
        files = [f for f in sorted(os.listdir(full_path)) if f.endswith(".md")]
        return {"directory": path, "files": files}
    if not os.path.exists(full_path):
        return {"error": f"File not found: {path}"}
    try:
        with open(full_path, "r") as f:
            content = f.read()
        if len(content) > 8000:
            content = content[:8000] + "\n\n[... truncated ...]"
        return {"file": path, "content": content}
    except Exception as e:
        return {"error": str(e)}


def _vault_insight(topic: str) -> dict:
    import re
    result = {"topic": topic, "vault_context": [], "memory_context": []}

    key_files = ["HOME.md", "Roadmap.md", "Decision Log.md", "Infrastructure Map.md"]
    projects_dir = os.path.join(VAULT_PATH, "projects", "active")
    if os.path.isdir(projects_dir):
        for f in os.listdir(projects_dir):
            if f.endswith(".md"):
                key_files.append(f"projects/active/{f}")
    domain_dir = os.path.join(VAULT_PATH, "domain")
    if os.path.isdir(domain_dir):
        for root, dirs, files in os.walk(domain_dir):
            for f in files:
                if f.endswith(".md"):
                    key_files.append(os.path.relpath(os.path.join(root, f), VAULT_PATH))

    vault_hits = []
    for rel_path in key_files:
        full = os.path.join(VAULT_PATH, rel_path)
        if not os.path.exists(full):
            continue
        try:
            with open(full, "r") as f:
                content = f.read()
            if topic.lower() in content.lower() or any(
                w.lower() in content.lower() for w in topic.split() if len(w) > 3
            ):
                lines = content.split("\n")
                relevant = []
                for i, line in enumerate(lines):
                    if any(w.lower() in line.lower() for w in topic.split() if len(w) > 3):
                        start = max(0, i - 2)
                        end = min(len(lines), i + 5)
                        relevant.append("\n".join(lines[start:end]))
                if relevant:
                    vault_hits.append({"file": rel_path, "excerpts": relevant[:3]})
        except Exception:
            continue

    result["vault_context"] = vault_hits[:5]

    if _memory and _memory._started:
        memories = _memory.recall(topic, n_results=10)
        result["memory_context"] = memories.get("results", [])
        result["total_memories"] = memories.get("total_memories", 0)

    vault_entities = set()
    for hit in vault_hits:
        for excerpt in hit.get("excerpts", []):
            for match in re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', excerpt):
                if len(match) > 3:
                    vault_entities.add(match)
    memory_entities = set()
    for mem in result.get("memory_context", []):
        for ent in mem.get("entities", []):
            memory_entities.add(ent)
    overlap = vault_entities & memory_entities
    if overlap:
        result["cross_references"] = list(overlap)

    return result


# ── Scanner ──────────────────────────────────────────────────────────────────

def _scan_digest(mode: str = "latest", top_n: int = 10) -> dict:
    try:
        from scanner import Scanner
        scanner = Scanner()
    except ImportError:
        return {"error": "Scanner module not available"}

    if mode == "latest":
        items = scanner.get_latest_candidates(min(top_n, 5))
        lines = []
        for i, item in enumerate(items, 1):
            title = (item.get("title") or "")[:150]
            source = item.get("source", "")
            lines.append(f"{i}. {title} ({source})")
        return {"summary": f"{len(items)} candidates:\n" + "\n".join(lines)}
    elif mode == "unreviewed":
        return {"mode": "unreviewed", "scans": scanner.get_unreviewed(), "count": len(scanner.get_unreviewed())}
    elif mode == "clear":
        unreviewed = scanner.get_unreviewed()
        all_items, seen_ids = [], set()
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
        all_items.sort(key=lambda x: x.get("relevance", 0) + x.get("score", 0) / 1000, reverse=True)
        lines = []
        for i, item in enumerate(all_items[:min(top_n, 5)], 1):
            title = (item.get("title") or "")[:150]
            source = item.get("source", "")
            lines.append(f"{i}. {title} ({source})")
        return {"summary": f"Processed {len(unreviewed)} scans, {len(all_items)} unique items. Top {min(top_n, 5)}:\n" + "\n".join(lines)}
    elif mode == "stats":
        return {"mode": "stats", **scanner.get_calibration_stats()}
    else:
        return {"error": f"Unknown mode: {mode}"}


# ── Playbook ─────────────────────────────────────────────────────────────────

import re as _re

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

def _playbook(section: str, action: str, content: str = "") -> dict:
    if action == "read":
        try:
            with open(PLAYBOOK_PATH, "r") as f:
                text = f.read()
            if section == "full":
                return {"playbook": text}
            marker = _PLAYBOOK_SECTIONS.get(section)
            if not marker:
                return {"error": f"Unknown section: {section}. Valid: {list(_PLAYBOOK_SECTIONS.keys())}"}
            idx = text.find(marker)
            if idx == -1:
                return {"error": f"Section '{marker}' not found"}
            start = idx + len(marker)
            rest = text[start:]
            end = len(rest)
            for boundary in ["\n## ", "\n---"]:
                pos = rest.find(boundary)
                if pos != -1 and pos < end:
                    end = pos
            return {"section": section, "content": rest[:end].strip()}
        except FileNotFoundError:
            return {"error": "Playbook not found"}

    if action in ("append", "replace"):
        if not content:
            return {"error": "content required"}
        try:
            with open(PLAYBOOK_PATH, "r") as f:
                text = f.read()
        except FileNotFoundError:
            return {"error": "Playbook not found"}
        marker = _PLAYBOOK_SECTIONS.get(section)
        if not marker:
            return {"error": f"Unknown section: {section}"}
        idx = text.find(marker)
        if idx == -1:
            return {"error": f"Section '{marker}' not found"}
        start = idx + len(marker)
        rest = text[start:]
        end = len(rest)
        for boundary in ["\n## ", "\n---"]:
            pos = rest.find(boundary)
            if pos != -1 and pos < end:
                end = pos
        old_section = rest[:end]
        new_section = (old_section.rstrip() + "\n" + content + "\n") if action == "append" else ("\n" + content + "\n")
        text = text[:start] + new_section + rest[end:]
        today = datetime.now().strftime("%Y-%m-%d %H:%M")
        if "*Last updated:" in text:
            text = _re.sub(r'\*Last updated:.*\*', f'*Last updated: {today} (auto)*', text)
        with open(PLAYBOOK_PATH, "w") as f:
            f.write(text)
        return {"status": "updated", "section": section, "action": action}

    return {"error": f"Unknown action: {action}"}


# ── Claude Inbox ─────────────────────────────────────────────────────────────

def _message_claude(message: str, priority: str = "medium", context: str = "") -> dict:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    entry = f"\n## {timestamp}\n**Priority:** {priority}\n**From:** Midas\n\n{message}\n"
    if context:
        entry += f"\n**Context:**\n```\n{context[:2000]}\n```\n"
    entry += "\n---\n"
    os.makedirs(os.path.dirname(CLAUDE_INBOX), exist_ok=True)
    if not os.path.exists(CLAUDE_INBOX):
        with open(CLAUDE_INBOX, "w") as f:
            f.write("# Claude Inbox\n\nMessages from Midas for Claude to review.\n\n---\n")
    with open(CLAUDE_INBOX, "a") as f:
        f.write(entry)
    return {"status": "sent", "timestamp": timestamp, "priority": priority}


# ── Main dispatch ────────────────────────────────────────────────────────────

def execute(tool_name: str, args: dict) -> str:
    """Execute a tool and return plain-text result.

    Argument validation happens here — reject obviously bad calls
    before touching any backend.
    """
    # Validation
    if tool_name == "memory_recall" and not args.get("query", "").strip():
        return "Error: Empty recall query. Provide a specific search term."
    if tool_name == "browse_search" and not args.get("query", "").strip():
        return "Error: Empty search query. Provide a specific query."
    if tool_name == "browse_navigate" and not args.get("url", "").startswith("http"):
        return f"Error: Invalid URL: {args.get('url', '')}. Must start with http(s)."

    try:
        result = _dispatch(tool_name, args)
    except Exception as e:
        result = {"error": str(e)}

    # Flatten to plain text — 9B regurgitates JSON, so prefer text
    if isinstance(result, dict):
        if "summary" in result and len(result) <= 2:
            return result["summary"]
        if "error" in result:
            return f"Error: {result['error']}"
        return json.dumps(result, indent=2)
    return str(result)


def _dispatch(name: str, args: dict) -> dict:
    """Route tool name to handler. Returns dict."""
    # Memory
    if name == "memory_ingest":
        if not _memory or not _memory._started:
            return {"error": "memory daemon not started"}
        return _memory.ingest(args.get("role", "user"), args.get("text", ""))
    if name == "memory_recall":
        if not _memory or not _memory._started:
            return {"error": "memory daemon not started"}
        return _memory.recall(args.get("query", ""), args.get("n_results", 5), args.get("type_filter", ""))
    if name == "memory_stats":
        if not _memory or not _memory._started:
            return {"error": "memory daemon not started"}
        return _memory.stats()
    if name == "memory_insights":
        if not _memory or not _memory._started:
            return {"error": "memory daemon not started"}
        return _memory.get_insights()

    # Vault
    if name == "vault_read":
        return _vault_read(args.get("path", ""), args.get("query", ""))
    if name == "vault_insight":
        return _vault_insight(args.get("topic", ""))

    # Browser
    if name == "browse_navigate":
        return _browser.navigate(args.get("url", ""), args.get("wait", 2))
    if name == "browse_read":
        return _browser.read_page(args.get("selector", "body"), args.get("max_length", 5000))
    if name == "browse_click":
        return _browser.click(args.get("selector", ""))
    if name == "browse_type":
        return _browser.type_text(args.get("selector", ""), args.get("text", ""))
    if name == "browse_js":
        return _browser.run_js(args.get("expression", ""))
    if name == "browse_search":
        return _browser.search(args.get("query", ""), args.get("max_results", 5))
    if name == "browse_x_feed":
        return _browser.scan_x_feed(args.get("count", 5))
    if name == "browse_tabs":
        return {"tabs": _browser.get_tabs()}

    # Scanner
    if name == "scan_digest":
        return _scan_digest(args.get("mode", "latest"), args.get("top_n", 10))

    # Playbook
    if name == "playbook_update":
        return _playbook(args.get("section", "full"), args.get("action", "read"), args.get("content", ""))

    # Claude inbox
    if name == "message_claude":
        return _message_claude(args.get("message", ""), args.get("priority", "medium"), args.get("context", ""))

    # Shell
    if name == "shell":
        cmd = args.get("command", "")
        try:
            out = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, timeout=30,
                cwd="/Users/midas/Desktop/cowork"
            )
            return {"stdout": out.stdout[-2000:] if out.stdout else "", "stderr": out.stderr[-500:] if out.stderr else "", "returncode": out.returncode}
        except subprocess.TimeoutExpired:
            return {"error": "command timed out (30s)"}

    return {"error": f"unknown tool: {name}"}
