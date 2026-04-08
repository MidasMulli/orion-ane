from __future__ import annotations
"""
Decision logging and implicit feedback detection for the deterministic router.

Every routing decision is logged. Corrections accumulate learning data.
The system gets more reliable through code changes informed by data.

Log format (JSONL):
  {"ts": "...", "msg": "...", "l1": "tool|null", "l2": "category|null",
   "final": "tool", "corrected": false, "correction": null}

Feedback triggers:
  - "no, I meant..." / "not that" / "wrong tool" → explicit correction
  - User immediately re-asks with different phrasing → implicit correction
  - User says "yes" / "perfect" / "exactly" after tool result → positive signal
"""

import json
import os
import re
from datetime import datetime

LOG_DIR = os.path.join(os.path.dirname(__file__), "routing_log")
LOG_FILE = os.path.join(LOG_DIR, "decisions.jsonl")
CORRECTIONS_FILE = os.path.join(LOG_DIR, "corrections.jsonl")
WEAKNESSES_FILE = os.path.join(LOG_DIR, "weaknesses.jsonl")
STATS_FILE = os.path.join(LOG_DIR, "stats.json")
LAST_STRESS_FILE = os.path.join(LOG_DIR, "last_stress_result.json")


def _ensure_dir():
    os.makedirs(LOG_DIR, exist_ok=True)


# ── Decision Logging ─────────────────────────────────────────────────────────

def log_decision(message: str, l1_result, l2_category: str | None, final_tool: str, final_args: dict):
    """Log every routing decision for analysis."""
    _ensure_dir()
    entry = {
        "ts": datetime.now().isoformat(),
        "msg": message[:200],
        "l1": l1_result[0] if l1_result else None,
        "l2": l2_category,
        "final": final_tool,
        "corrected": False,
    }
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ── Feedback Detection ───────────────────────────────────────────────────────

# Explicit correction patterns
_CORRECTION_PATTERNS = [
    r"^no[,.]?\s",
    r"^not that",
    r"^wrong tool",
    r"^i meant",
    r"^i didn'?t mean",
    r"^that'?s not what",
    r"^don'?t search",
    r"^don'?t use",
    r"^cancel that",
    r"^stop",
    r"^i wanted",
    r"^use (\w+) instead",
]

# Positive confirmation patterns
_POSITIVE_PATTERNS = [
    r"^(yes|yeah|yep|exactly|perfect|great|good|thanks|thank you|correct|right)",
    r"^that'?s (right|correct|what i wanted|perfect)",
    r"^nice",
]

# Implicit re-ask: same intent, different phrasing (detected by same tool target)
_REPHRASE_WINDOW = 2  # messages


def detect_feedback(user_msg: str, last_tool: str | None, last_msg: str | None) -> dict | None:
    """Detect explicit corrections, positive signals, or implicit re-asks.

    Returns:
        dict with 'type' (correction|positive|rephrase) and details, or None.
    """
    lower = user_msg.lower().strip()

    # Explicit correction
    for pattern in _CORRECTION_PATTERNS:
        if re.match(pattern, lower):
            # Try to extract what tool the user wanted
            wanted = None
            tool_mention = re.search(r'use (\w+)', lower)
            if tool_mention:
                wanted = tool_mention.group(1)
            return {
                "type": "correction",
                "original_tool": last_tool,
                "wanted_tool": wanted,
                "user_msg": user_msg,
                "ts": datetime.now().isoformat(),
            }

    # Positive confirmation
    for pattern in _POSITIVE_PATTERNS:
        if re.match(pattern, lower):
            return {
                "type": "positive",
                "confirmed_tool": last_tool,
                "user_msg": user_msg,
                "ts": datetime.now().isoformat(),
            }

    return None


def log_correction(feedback: dict):
    """Log a correction or positive signal for pattern analysis."""
    _ensure_dir()
    with open(CORRECTIONS_FILE, "a") as f:
        f.write(json.dumps(feedback) + "\n")

    # Update stats
    _update_stats(feedback)


def _update_stats(feedback):
    """Maintain running stats of routing accuracy."""
    stats = _load_stats()

    if feedback["type"] == "correction":
        tool = feedback.get("original_tool", "unknown")
        stats.setdefault("corrections", {})
        stats["corrections"].setdefault(tool, 0)
        stats["corrections"][tool] += 1
        stats["total_corrections"] = stats.get("total_corrections", 0) + 1
    elif feedback["type"] == "positive":
        tool = feedback.get("confirmed_tool", "unknown")
        stats.setdefault("confirmations", {})
        stats["confirmations"].setdefault(tool, 0)
        stats["confirmations"][tool] += 1
        stats["total_confirmations"] = stats.get("total_confirmations", 0) + 1

    stats["last_updated"] = datetime.now().isoformat()

    with open(STATS_FILE, "w") as f:
        json.dump(stats, f, indent=2)


def _load_stats() -> dict:
    try:
        with open(STATS_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"total_decisions": 0, "total_corrections": 0, "total_confirmations": 0}


# ── Analysis ─────────────────────────────────────────────────────────────────

def get_routing_stats() -> dict:
    """Get routing accuracy stats for the playbook / dashboard."""
    stats = _load_stats()

    # Count total decisions
    try:
        with open(LOG_FILE) as f:
            total = sum(1 for _ in f)
        stats["total_decisions"] = total
    except FileNotFoundError:
        stats["total_decisions"] = 0

    # Accuracy estimate
    total_d = stats["total_decisions"]
    total_c = stats.get("total_corrections", 0)
    if total_d > 0:
        stats["accuracy_pct"] = round((total_d - total_c) / total_d * 100, 1)
    else:
        stats["accuracy_pct"] = None

    # Most-corrected tools
    corrections = stats.get("corrections", {})
    if corrections:
        stats["most_corrected"] = sorted(corrections.items(), key=lambda x: x[1], reverse=True)[:3]

    return stats


def get_recent_corrections(n: int = 10) -> list:
    """Get the N most recent corrections for review."""
    try:
        with open(CORRECTIONS_FILE) as f:
            lines = f.readlines()
        entries = [json.loads(l) for l in lines[-n:]]
        return entries
    except FileNotFoundError:
        return []


# ── Self-identified Weakness Tracking ────────────────────────────────────────

def log_self_identified_weakness(test_id: str, category: str, input_msg: str,
                                  expected: str, actual: str, reason: str):
    """Log a weakness found by self-test. Same format as user corrections
    so router_improver.py can process both uniformly."""
    _ensure_dir()
    entry = {
        "ts": datetime.now().isoformat(),
        "source": "self_test",
        "test_id": test_id,
        "category": category,
        "input": input_msg[:200],
        "expected_route": expected,
        "actual_route": actual,
        "reason": reason,
    }
    with open(WEAKNESSES_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


def get_recent_weaknesses(n: int = 20) -> list:
    """Get the N most recent self-identified weaknesses."""
    try:
        with open(WEAKNESSES_FILE) as f:
            lines = f.readlines()
        return [json.loads(l) for l in lines[-n:]]
    except FileNotFoundError:
        return []


def save_stress_result(result: dict):
    """Save the last stress test result for comparison."""
    _ensure_dir()
    result["ts"] = datetime.now().isoformat()
    with open(LAST_STRESS_FILE, "w") as f:
        json.dump(result, f, indent=2)


def load_last_stress_result() -> dict | None:
    """Load the last stress test result for comparison."""
    try:
        with open(LAST_STRESS_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


# ── Session Stats & Last Decision ────────────────────────────────────────────

def get_last_decision() -> dict | None:
    """Get the most recent routing decision."""
    try:
        with open(LOG_FILE) as f:
            lines = f.readlines()
        if lines:
            return json.loads(lines[-1])
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    return None


def get_session_stats() -> dict:
    """Get detailed session-level routing stats."""
    stats = get_routing_stats()

    # Count L1 vs L2 vs conversation from decisions log
    l1_count = 0
    l2_count = 0
    conv_count = 0
    route_times = []

    try:
        with open(LOG_FILE) as f:
            for line in f:
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if d.get("l1"):
                    l1_count += 1
                elif d.get("l2"):
                    l2_count += 1
                elif d.get("final") == "conversation":
                    conv_count += 1
                if "route_ms" in d:
                    route_times.append(d["route_ms"])
    except FileNotFoundError:
        pass

    stats["l1_count"] = l1_count
    stats["l2_count"] = l2_count
    stats["conv_count"] = conv_count
    stats["avg_route_ms"] = round(sum(route_times) / len(route_times), 2) if route_times else None

    # Last stress test
    last_stress = load_last_stress_result()
    if last_stress:
        stats["last_stress_result"] = {
            "total": last_stress.get("total", 0),
            "pass": last_stress.get("pass", 0),
            "warn": last_stress.get("warn", 0),
            "fail": last_stress.get("fail", 0),
            "ts": last_stress.get("ts", ""),
        }

    return stats
