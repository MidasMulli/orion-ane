"""
Phase 4A: Signal Bus — the nervous system.

Shared state dictionary. Every component reads and writes its own section.
No component writes to another's section. This is the coordination layer
that enables adaptive behavior across the cognitive architecture.

Usage:
    from signal_bus import bus, update, read

    # Writer (only the owning component):
    update("70b_tok_s", 12.5)

    # Reader (any component):
    tps = read("70b_tok_s")
    breadth = read("retrieval_breadth")
"""

import time
import threading

_lock = threading.Lock()

_state = {
    # 1B adaptation state
    "1b_avg_rank": 0,
    "1b_confidence": 0.0,
    "1b_domain": "unknown",
    "1b_adaptation_magnitude": 0.0,

    # Subconscious state
    "memory_count": 0,
    "last_retrieval_relevance": 0.0,
    "domain_detected": "unknown",
    "memories_used_by_70b": 0,
    "memories_ignored_by_70b": 0,
    "retrieval_ms": 0.0,

    # 70B state
    "70b_tok_s": 0.0,
    "70b_tokens_generated": 0,
    "70b_finish_reason": "",
    "generation_active": False,

    # System state
    "thermal_pressure": "nominal",
    "memory_pressure_mb": 0,
    "user_idle_seconds": 0,
    "conversation_turn": 0,
    "session_start_time": time.time(),

    # Orchestration (computed from signals)
    "retrieval_breadth": 10,
    "retrieval_threshold": 0.35,
    "adaptation_lr": 0.1,
    "reasoning_mode": "single",
    "system_mode": "active",

    # Query classification
    "query_type": "casual",
    "query_mode": "",
}


def update(key, value):
    """Update a signal. Thread-safe."""
    with _lock:
        if key in _state:
            _state[key] = value
        else:
            _state[key] = value  # Allow new keys


def read(key, default=None):
    """Read a signal. Thread-safe."""
    with _lock:
        return _state.get(key, default)


def snapshot():
    """Return a copy of the full bus state."""
    with _lock:
        return dict(_state)


def update_batch(updates):
    """Update multiple signals atomically."""
    with _lock:
        _state.update(updates)


# Module-level access for convenience
bus = _state
