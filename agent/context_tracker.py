"""Main 35 +1 Task 4 — ContextTracker.

Detects what topics the user is currently working on and produces
per-topic weight multipliers for retrieval. The retrieval scorer in
multi_path_retrieve.py uses these to gently boost active-topic
memories without burying dormant-topic memories.

Design constraints (per directive):
- Topic detection v1 = keyword heuristic (no model call, instant).
  Embedding fallback for unmatched messages is a TODO.
- Boost is multiplicative and conservative: max +30% on active topic.
  Dormant topics are NOT penalized — they just don't get the boost.
- Weights decay gently per message (× 0.95) so context drifts
  smoothly rather than flipping hard.
- Topic clusters are sourced from data/session_analysis.json (the
  Main 35 lean topic analysis), not hardcoded here. The TOPIC_KEYWORDS
  dict below is the v1 default; can be overridden via constructor.
"""
from __future__ import annotations
import json
import re
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional


# v1 keyword clusters — derived from Main 35 build_session_analysis.py
DEFAULT_TOPIC_KEYWORDS = {
    "hardware_characterization": [
        "slc", "amcc", "macc", "dcs", "ane", "nax", "gpu", "amx",
        "bandwidth", "register", "kext", "ioreport", "tflops",
        "fabric", "silicon", "m5 pro", "m4", "dispatch", "mmio",
    ],
    "midas_infrastructure": [
        "midas", "midas_ui", "agent", "router", "delegation", "server",
        "endpoint", "/api/", "spec decode", "n-gram", "drafter",
        "verifier", "prompt cache", "production stack",
    ],
    "subconscious_memory": [
        "subconscious", "memory", "extraction", "fact extract", "recall",
        "retrieval", "memorystore", "maintenance loop", "enricher",
        "supersession", "canonical state", "vault sync",
    ],
    "ml_models_training": [
        "llama", "qwen", "gpt", "8b", "72b", "3b", "1b", "tok/s",
        "fine-tune", "lora", "distill", "inference", "training",
        "hugging face", "model",
    ],
    "ane_compiler": [
        "ane-compiler", "ane-dispatch", "coreml", "hwx", "compile",
        "fusion", "conv2d", "gelu", "ffn", "espresso", "mlpackage",
        # Main 35 +3 T2B refinements: ANE opcode / dispatch terminology
        "opcode", "0x9341", "0x9141", "0x9241", "dispatch table",
        "kernel", "isa", "macroop", "instruction",
    ],
    "paper_writing": [
        "paper", "draft", "abstract", "every cycle", "arxiv",
        "reviewer", "methodology", "citation", "thesis", "m1", "m2",
        "m5", "m6", "locomo", "gold set",
    ],
    "cen_derivatives": [
        "cen", "isda", "csa", "collateral", "derivative", "swap",
        "var", "risk", "trade lifecycle", "haircut", "margin",
        "repo", "libor", "sofr",
    ],
    "strategic_planning": [
        "plan", "strateg", "priorit", "roadmap", "next session",
        "decision", "directive", "main ", "kill test", "commit",
    ],
    "agent_systems": [
        "agent", "mcp", "tool use", "claude code", "cli",
        "claude desktop", "hook", "skill", "subagent",
    ],
    "ai_news_research": [
        "karpathy", "magnitude", "earthquake", "industry",
        "memori", "benchmark", "arxiv",
    ],
    "personal_misc": [
        "macbook", "mac air", "tinnitus", "board game", "switch",
        "food", "travel", "diet", "peptide", "motrin",
    ],
}

DEFAULT_BOOST_MULTIPLIER = 0.30   # max +30% to dominant-topic memory scores
DEFAULT_SECONDARY_BOOST_MULTIPLIER = 0.15  # secondary topic gets half the boost
DEFAULT_DECAY_PER_MESSAGE = 0.92  # Main 35 +4: 0.95 → 0.92 — gentler hold on dominant
DEFAULT_ACTIVE_BOOST_PER_HIT = 0.25   # 0.20 → 0.25 — faster topic adoption
DEFAULT_MIN_RELEVANT_WEIGHT = 0.15    # 0.10 → 0.15 — stricter "active" threshold
DEFAULT_BRIEF_QUERY_CHARS = 100       # messages shorter than this skip state update if off-topic
DEFAULT_MAX_ACTIVE_TOPICS = 2         # cap concurrent active topics at 2 (avoid signal dilution)


class ContextTracker:
    """Tracks the user's active topic distribution from a stream of messages.

    Use:
        tracker = ContextTracker()
        tracker.on_message("what is the SLC way count on M5 Pro?")
        boost = tracker.get_retrieval_boost()  # {topic: multiplier}
        # Pass boost into multi_path_recall via the new context_boost arg.
    """

    def __init__(
        self,
        topic_keywords: Optional[dict[str, list[str]]] = None,
        boost_multiplier: float = DEFAULT_BOOST_MULTIPLIER,
        secondary_boost_multiplier: float = DEFAULT_SECONDARY_BOOST_MULTIPLIER,
        decay_per_message: float = DEFAULT_DECAY_PER_MESSAGE,
        active_boost_per_hit: float = DEFAULT_ACTIVE_BOOST_PER_HIT,
        min_relevant_weight: float = DEFAULT_MIN_RELEVANT_WEIGHT,
        brief_query_chars: int = DEFAULT_BRIEF_QUERY_CHARS,
        max_active_topics: int = DEFAULT_MAX_ACTIVE_TOPICS,
    ):
        self.topic_keywords = topic_keywords or DEFAULT_TOPIC_KEYWORDS
        self.boost_multiplier = boost_multiplier
        self.secondary_boost_multiplier = secondary_boost_multiplier
        self.decay_per_message = decay_per_message
        self.active_boost_per_hit = active_boost_per_hit
        self.min_relevant_weight = min_relevant_weight
        self.brief_query_chars = brief_query_chars
        self.max_active_topics = max_active_topics

        # Compile keyword patterns once
        self._patterns = {
            topic: re.compile(
                r"\b(" + "|".join(re.escape(k) for k in keywords) + r")\b",
                re.IGNORECASE,
            )
            for topic, keywords in self.topic_keywords.items()
        }

        self.topic_weights: dict[str, float] = defaultdict(float)
        # Main 35 +4: replace single active_topic with ranked list of (topic, weight)
        # tuples. The dominant topic is active_topics[0]. Up to max_active_topics
        # entries are returned by get_retrieval_boost.
        self.active_topics: list[tuple[str, float]] = []
        self.topic_history: list[dict] = []
        self.session_start_ts: Optional[float] = None
        self.message_count: int = 0
        self.brief_queries_skipped: int = 0  # for diagnostics

    # ── classification ──
    def _classify_topic(self, text: str) -> Optional[str]:
        """Return the topic with the most keyword hits in `text`. None if no hits."""
        if not text:
            return None
        scores: dict[str, int] = {}
        for topic, patt in self._patterns.items():
            hits = len(patt.findall(text))
            if hits:
                scores[topic] = hits
        if not scores:
            return None
        return max(scores, key=scores.get)

    # ── lifecycle hooks ──
    @property
    def active_topic(self) -> Optional[str]:
        """Backward-compat: dominant topic name (or None)."""
        return self.active_topics[0][0] if self.active_topics else None

    def _refresh_active_topics(self):
        """Sort topic_weights into the ranked active_topics list.

        Filters by min_relevant_weight and caps at max_active_topics.
        """
        sorted_topics = sorted(self.topic_weights.items(),
                                key=lambda kv: -kv[1])
        self.active_topics = [
            (t, w) for (t, w) in sorted_topics
            if w >= self.min_relevant_weight
        ][:self.max_active_topics]

    def on_session_start(self) -> None:
        """Called once when a new conversation session begins. Resets weights."""
        self.session_start_ts = time.time()
        self.topic_weights = defaultdict(float)
        self.topic_history = []
        self.active_topics = []
        self.message_count = 0
        self.brief_queries_skipped = 0

    def warm_from_recent_events(self, recent_topics: list[tuple[str, float]]) -> None:
        """Optional: seed topic weights from recent events log entries.

        recent_topics: list of (topic, recency_seconds_ago) pairs.
        Weights are scaled by an exponential recency decay.
        """
        for topic, age_s in recent_topics:
            recency_weight = max(0.1, min(1.0, 3600.0 / max(age_s, 1.0)))
            self.topic_weights[topic] = max(self.topic_weights[topic], recency_weight)
        self._refresh_active_topics()

    def on_message(self, text: str, role: str = "human") -> Optional[str]:
        """Called per user message. Updates weights and returns the detected topic.

        Main 35 +4 multi-topic: maintains a vector of topic weights, not just
        a single active topic. Brief off-topic queries (< brief_query_chars)
        are answered but do not shift the topic state — see corpus pattern of
        16 quick-Q&A sessions at 3.3% of messages.
        """
        self.message_count += 1
        detected = self._classify_topic(text)

        # Brief-query guard: short messages that DON'T match any current
        # active topic skip the weight update entirely. The retrieval still
        # works via embedding similarity; only the state-tracking is
        # suppressed so a "what time is it" doesn't decay a 3-week hardware
        # deep dive.
        #
        # Three cases for a short message (< brief_query_chars):
        #   (a) detected is None and we have active topics
        #       → off-topic personal Q&A — skip update
        #   (b) detected is some topic NOT in active_topics
        #       → brief detour to a totally new topic — skip update
        #   (c) detected is in active_topics OR no active state yet
        #       → on-topic or first-message — process normally
        if len(text) < self.brief_query_chars and self.active_topics:
            active_set = {t for t, _ in self.active_topics}
            if detected is None or detected not in active_set:
                self.brief_queries_skipped += 1
                return detected

        # Decay all topics gently
        for t in list(self.topic_weights.keys()):
            self.topic_weights[t] *= self.decay_per_message
            if self.topic_weights[t] < 0.01:
                del self.topic_weights[t]

        if detected:
            previous_dominant = self.active_topic
            self.topic_weights[detected] = min(
                1.0, self.topic_weights[detected] + self.active_boost_per_hit
            )
            self._refresh_active_topics()
            new_dominant = self.active_topic
            if new_dominant != previous_dominant:
                self.topic_history.append({
                    "from": previous_dominant,
                    "to": new_dominant,
                    "ts": datetime.utcnow().isoformat(),
                    "msg_idx": self.message_count,
                })
        else:
            self._refresh_active_topics()
        return detected

    # ── retrieval-side API ──
    def get_retrieval_boost(self) -> dict[str, float]:
        """Return per-topic boost values for the top-N active topics.

        Main 35 +4 multi-topic: the dominant topic gets boost_multiplier × weight,
        secondary topics get secondary_boost_multiplier × weight (half by default).
        Dormant topics are excluded (not in active_topics list).

        The retrieval scorer applies: score *= 1 + topic_boost_for_memory_topic
        """
        out: dict[str, float] = {}
        for i, (topic, weight) in enumerate(self.active_topics):
            mult = self.boost_multiplier if i == 0 else self.secondary_boost_multiplier
            out[topic] = weight * mult
        return out

    def state(self) -> dict:
        """Return a JSON-serializable snapshot of the tracker state."""
        return {
            "active_topic": self.active_topic,
            "active_topics": [
                {"topic": t, "weight": round(w, 3)} for t, w in self.active_topics
            ],
            "message_count": self.message_count,
            "brief_queries_skipped": self.brief_queries_skipped,
            "topic_weights": {k: round(v, 3) for k, v in self.topic_weights.items()},
            "topic_history": self.topic_history,
            "session_start_ts": self.session_start_ts,
        }


# ── Helper to apply boost to a score (used by multi_path_retrieve patch) ──
def apply_context_boost(
    base_score: float,
    memory_topic: str,
    context_boost: dict[str, float],
    boost_multiplier: float = DEFAULT_BOOST_MULTIPLIER,
) -> float:
    """Apply context boost to a base score. Active-topic memories get up to
    `boost_multiplier` × topic_weight uplift; dormant topics are unchanged.
    """
    if not context_boost or not memory_topic:
        return base_score
    weight = context_boost.get(memory_topic, 0.0)
    return base_score * (1.0 + boost_multiplier * weight)


if __name__ == "__main__":
    # Main 35 +4 smoke test: cross-topic + brief-query guard
    t = ContextTracker()
    t.on_session_start()
    print("=== ContextTracker multi-topic smoke test ===\n")
    msgs = [
        ("Let's analyze the SLC way count measurements from Main 27", "long hardware"),
        ("How does that affect the memory system design?", "long hardware+subc"),
        ("Tell me more about the AMCC bandwidth ceiling and how subconscious memory uses it for retrieval scoring", "long cross-topic"),
        ("what time is it", "BRIEF off-topic — should not shift state"),
        ("what's the weather", "BRIEF off-topic — should not shift state"),
        ("Back to the AMCC — what's the bandwidth ceiling we measured in Main 30?", "long hardware return"),
    ]
    for m, lbl in msgs:
        topic = t.on_message(m)
        print(f"  [{lbl}]")
        print(f"    text: '{m[:70]}{'...' if len(m)>70 else ''}'")
        print(f"    detected: {topic}")
        print(f"    active: {[(tp, round(w,2)) for tp,w in t.active_topics]}")
        print()
    print("--- final state ---")
    print(f"active_topic (dominant): {t.active_topic}")
    print(f"active_topics: {[(tp, round(w,3)) for tp,w in t.active_topics]}")
    print(f"brief_queries_skipped: {t.brief_queries_skipped}")
    print()
    print("get_retrieval_boost():")
    print(json.dumps(t.get_retrieval_boost(), indent=2))
