"""
Phase 2B: Structured briefing document assembly.

Replaces raw memory injection with a coherent briefing from multiple sources.
Regenerated every 5 turns (cached between regenerations).

Sources:
  1. Top-20 memories from Subconscious (via memory_cache)
  2. CLAUDE.md active section (first 50 lines)
  3. Current conversation domain
  4. Key measurements from memory
"""

import os
import time

VAULT_PATH = "/Users/midas/Desktop/cowork/vault"
CLAUDE_MD = os.path.join(VAULT_PATH, "CLAUDE.md")


def _load_claude_active():
    """Load first 50 lines of CLAUDE.md for current project state."""
    try:
        with open(CLAUDE_MD) as f:
            lines = f.readlines()[:50]
        # Extract just the active projects and infrastructure
        text = "".join(lines)
        # Trim to ~200 tokens worth
        if len(text) > 800:
            text = text[:800]
        return text.strip()
    except FileNotFoundError:
        return ""


def _extract_key_numbers(memories):
    """Pull key measurements from top memories."""
    numbers = []
    for m in memories[:10]:
        text = m.get("text", "")
        # Look for measurement patterns
        import re
        measurements = re.findall(
            r'(\d+\.?\d*\s*(?:tok/s|GB/s|ms|dispatches|%|MB|GB|facts))',
            text)
        if measurements:
            numbers.append(f"{measurements[0]} — {text[:60]}")
        if len(numbers) >= 5:
            break
    return numbers


def assemble_briefing(memories, domain="research", session_focus="",
                      turn_count=0, query=""):
    """Build structured briefing document for 70B context injection.

    Args:
        memories: list of memory dicts from retrieval
        domain: detected conversation domain
        session_focus: detected from recent turns
        turn_count: current turn number
        query: the user's current message — Main 24 Build 0 plumbs this
               through so the presentation layer can format memories
               appropriately for the query category. If empty, falls back
               to the legacy raw-memory dump.

    Returns:
        str: formatted briefing (~500 tokens max)
    """
    parts = []

    parts.append("BRIEFING:")
    parts.append("User: Nick, SVP. Research: Apple Silicon inference, ANE RE, "
                 "speculative decoding.")

    if session_focus:
        parts.append(f"Session focus: {session_focus}")

    # Key numbers from memories
    numbers = _extract_key_numbers(memories)
    if numbers:
        parts.append("Key numbers:")
        for n in numbers[:5]:
            parts.append(f"  - {n}")

    # Current state from CLAUDE.md
    active = _load_claude_active()
    if active:
        # Extract just the roadmap/active section
        lines = active.split("\n")
        active_lines = []
        in_active = False
        for line in lines:
            if "Active NOW" in line or "Production" in line:
                in_active = True
            if in_active:
                active_lines.append(line)
            if len(active_lines) > 8:
                break
        if active_lines:
            parts.append("Current state:")
            parts.append("\n".join(active_lines[:8]))

    # Memories — Main 24 Build 0 presentation layer.
    # If we have a query and the multi_path presentation layer is importable,
    # use category-aware formatting. Otherwise fall back to the legacy ranked
    # bullet dump (which is what shipped before Main 24).
    if memories:
        formatted = None
        if query:
            try:
                import sys as _sys
                _sys.path.insert(0, "/Users/midas/Desktop/cowork/vault/subconscious")
                from multi_path_retrieve import present
                formatted = present(memories, query, max_chars=1500)
            except Exception as _e:
                formatted = None
        if formatted:
            parts.append(formatted)
        else:
            # Legacy raw-memory dump
            parts.append(f"Relevant context ({len(memories)} memories):")
            for i, m in enumerate(memories[:15]):
                score = m.get("score", 0)
                text = m.get("text", "")[:120]
                parts.append(f"  {i+1}. [{score:.2f}] {text}")

    briefing = "\n".join(parts)

    # Cap at ~500 tokens (~2000 chars)
    if len(briefing) > 2000:
        briefing = briefing[:2000] + "\n  [... truncated]"

    return briefing
