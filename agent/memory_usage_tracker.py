"""
Phase 7A: Memory Usage Tracking.

After 70B generates a response, check which injected memories were
actually referenced. Used memories get relevance boost. Unused memories
get a small penalty. Over many conversations, genuinely useful memories
rise to the top.
"""

import re
import logging

log = logging.getLogger("memory_usage")


def track_memory_usage(response_text, injected_memories, signal_bus_update=None):
    """Compare response against injected memories.

    Args:
        response_text: the 70B's response
        injected_memories: list of memory dicts with 'text' and 'score'

    Returns:
        dict with used/ignored counts and lists
    """
    if not response_text or not injected_memories:
        return {"used": 0, "ignored": 0, "used_memories": [], "ignored_memories": []}

    resp_words = set(
        w.lower().strip('.,;:()[]"\'-/')
        for w in response_text.split()
        if len(w) > 4
    )

    used = []
    ignored = []

    for mem in injected_memories:
        text = mem if isinstance(mem, str) else mem.get("text", "")
        mem_words = set(
            w.lower().strip('.,;:()[]"\'-/')
            for w in text.split()
            if len(w) > 4
        )

        if not mem_words:
            ignored.append(text)
            continue

        overlap = len(resp_words & mem_words)
        usage_ratio = overlap / len(mem_words)

        if usage_ratio > 0.25:
            used.append(text)
        else:
            ignored.append(text)

    result = {
        "used": len(used),
        "ignored": len(ignored),
        "used_memories": used,
        "ignored_memories": ignored,
    }

    # Update signal bus
    if signal_bus_update:
        try:
            signal_bus_update("memories_used_by_70b", len(used))
            signal_bus_update("memories_ignored_by_70b", len(ignored))
        except Exception:
            pass

    return result
