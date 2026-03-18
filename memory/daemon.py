"""
Three-Tier Memory Daemon for Local LLMs on Apple Silicon
=========================================================

Tier 1 — CPU: Extract facts + embed into ChromaDB (1,721/sec, real-time)
Tier 2 — ANE: Analysis + summarization via 1.7B CoreML (async, 57 tok/s, 2W background)
Tier 3 — GPU: Conversation + reasoning (25 tok/s, interactive)

All three tiers run concurrently. Near-zero contention measured (~3.8% interference, within noise).

Usage:
    daemon = MemoryDaemon(vault_path="/path/to/vault")
    daemon.start()

    # Feed conversation turns as they happen
    daemon.ingest("user", "What is the cross-default threshold for Counterparty X?")
    daemon.ingest("assistant", "The cross-default threshold is $50M including affiliates...")

    # At next session start, retrieve relevant context
    context = daemon.recall("cross-default threshold Counterparty X")
    # → Returns relevant facts from previous sessions
"""

import os
import re
import time
import json
import queue
import hashlib
import threading
from datetime import datetime
from typing import Optional

import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer


# ── Fact Extraction (CPU, heuristic, zero model cost) ────────────

class FactExtractor:
    """Extracts atomic facts from conversation text using heuristics.

    v2: Rewrote entity extraction to use domain-specific patterns instead
    of naive multi-word capitalized phrase matching. Added noise filtering,
    better sentence splitting, and content-hash deduplication.
    """

    # ─── Quantities ───
    QUANTITY_PATTERN = re.compile(
        r'(?:USD|GBP|EUR|\$|£|€)\s*[\d,]+(?:\.\d+)?(?:\s*(?:million|billion|M|B|K))?\b'
        r'|[\d,]+(?:\.\d+)?%'
        r'|\b\d+\s+(?:days?|months?|years?|business days?)\b',
        re.IGNORECASE
    )

    # ─── Entities (domain-specific, curated patterns) ───
    ENTITY_PATTERNS = [
        # Finance/legal document types
        re.compile(r'\b(ISDA(?:\s+Master\s+Agreement)?|CSA|Credit\s+Support\s+Annex|Schedule)\b', re.IGNORECASE),
        # Parties / clients (formal and informal naming)
        re.compile(r'\b((?:Party|Counterparty|[Cc]lient)\s+[A-Z]\w*)\b'),
        # Standalone proper names used as identifiers (Alpha, Beta, Gamma, etc.)
        re.compile(r'\b((?:Alpha|Beta|Gamma|Delta|Epsilon|Zeta|Omega|Sigma|Theta|Lambda)\b)'),
        # Sections/clauses
        re.compile(r'\b(Section\s+\d+(?:\.\d+)*(?:\([a-z]\))?)\b', re.IGNORECASE),
        # Named provisions/concepts (multi-word finance terms)
        re.compile(r'\b(cross[- ]default(?:\s+threshold)?|close[- ]out\s+netting|automatic\s+early\s+termination'
                   r'|credit\s+event\s+upon\s+merger|additional\s+termination\s+event'
                   r'|eligible\s+collateral|independent\s+amount|minimum\s+transfer\s+amount'
                   r'|threshold\s+amount|rating[- ]dependent\s+threshold'
                   r'|force\s+majeure|flawed\s+assets?|netting\s+provisions?'
                   r'|variation\s+margin|initial\s+margin|haircut'
                   r'|NAV\s+trigger|credit\s+support|valuation\s+date)\b', re.IGNORECASE),
        # Credit ratings
        re.compile(r'\b((?:AAA|AA|A|BBB|BB|B|CCC|CC|C)[+-]?)\b'),
        # Currencies
        re.compile(r'\b(USD|EUR|GBP|JPY|CHF)\b'),
        # Securities types
        re.compile(r'\b(US\s+Treasur(?:y|ies)|Agency\s+securities|government\s+securities'
                   r'|G7\s+government\s+securities)\b', re.IGNORECASE),
        # Organizations (explicitly named, not just any capitalized words)
        re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Bank|Corp|Inc|LLC|LP|Fund|Capital|Partners'
                   r'|Securities|Financial|Asset\s+Management|Group|Holdings))\b'),
        # General proper nouns — capitalized words not in common-word stoplist
        # This is the catch-all for real company/person names (Apple, Goldman, JPMorgan, etc.)
        re.compile(r'\b([A-Z][a-zA-Z]{2,})\b'),
    ]

    # Words that match the proper noun pattern but aren't entities
    ENTITY_STOPWORDS = {
        # Common sentence starters / English words that get capitalized
        "The", "This", "That", "These", "Those", "There", "They", "Then",
        "What", "When", "Where", "Which", "While", "Who", "Why", "How",
        "Here", "Have", "Has", "Had", "His", "Her", "Its",
        "Are", "And", "But", "For", "Not", "All", "Any", "Can",
        "May", "Our", "Out", "Own", "Too", "Was", "Will", "Yes",
        "Also", "Some", "Each", "From", "Just", "Into", "Over",
        "Such", "Very", "Been", "Both", "Does", "Done", "Down",
        "Even", "Gets", "Goes", "Gone", "Good", "Got", "Great",
        "Keep", "Kept", "Know", "Last", "Left", "Let", "Like",
        "Long", "Look", "Made", "Make", "Many", "More", "Most",
        "Much", "Must", "Need", "Next", "Note", "Now", "Only",
        "Part", "Plus", "Put", "Said", "Same", "See", "Set",
        "Should", "Show", "Side", "Since", "Still", "Sure", "Take",
        "Tell", "Than", "Them", "Think", "Time", "Under", "Upon",
        "Used", "Using", "Want", "Way", "Well", "Were", "With",
        "Would", "Year", "Your",
        # Common adjectives/nouns that start sentences
        "Important", "Key", "Main", "New", "Old", "Other",
        "First", "Second", "Third", "Based", "Given", "Known",
        "Noted", "Stored", "Done", "Got", "Updated", "Added",
        # Common nouns that aren't entities
        "Meeting", "Client", "Clients", "Company", "Bank", "Counterparty", "Party",
        "Threshold", "Amount", "Agreement", "Schedule", "Rating",
        "Group", "Inc", "Corp", "Fund", "Capital", "Partners", "Holdings",
        "Securities", "Financial", "Management",
        "Today", "Tomorrow", "Monday", "Tuesday", "Wednesday",
        "Thursday", "Friday", "Saturday", "Sunday",
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
    }

    # ─── Type markers ───
    DECISION_MARKERS = [
        "decided", "chose", "selected", "will use", "switched to",
        "going with", "agreed", "confirmed", "settled on", "set at",
        "set to", "specified", "established",
        "we'll go with", "final answer", "approved",
    ]

    TASK_MARKERS = [
        "need to", "todo", "next step", "will do",
        "plan to", "going to", "must", "remember to",
        "deadline", "due by", "by friday", "by monday",
        "by end of", "action item",
        "meeting with", "scheduled", "should be meeting",
        "follow up", "check in", "reach out", "send to",
    ]

    PREFERENCE_MARKERS = [
        "prefer", "we like", "we want", "always use",
        "never use", "our standard", "our policy", "we require",
        "we typically", "we usually", "our approach",
    ]

    # ─── Noise filters ───
    FILLER_PATTERNS = re.compile(
        r'^(?:(?:sure|ok|okay|yes|no|yeah|yep|nope|thanks|thank you|hello|hi|hey'
        r'|got it|sounds good|makes sense|right|exactly|absolutely|understood'
        r'|perfect|great|good point|fair enough|interesting|I see|hmm|let me think'
        r'|can you help|what do you think|how about|what about)[.!?,\s]*)+$',
        re.IGNORECASE
    )

    # Skip assistant preamble/filler
    ASSISTANT_FILLER = re.compile(
        r'^(?:I can help with that|I\'d be happy to|Let me|Here\'s|Sure,? (?:I\'ll|let me)|'
        r'That\'s a (?:good|great) (?:point|question)|I\'ll note)',
        re.IGNORECASE
    )

    def __init__(self):
        self._seen_hashes = set()  # Content-hash dedup within session

    def extract(self, text: str, role: str = "user") -> list[dict]:
        """Extract atomic facts from a text block."""
        facts = []
        sentences = self._split_sentences(text)

        for sentence in sentences:
            s = sentence.strip()

            # ── Length filter ──
            if len(s) < 12 or len(s) > 500:
                continue

            # ── Noise filter ──
            if self.FILLER_PATTERNS.match(s):
                continue

            # ── Content-hash dedup ──
            content_hash = hashlib.md5(s.lower().encode()).hexdigest()[:12]
            if content_hash in self._seen_hashes:
                continue

            # ── Extract components ──
            entities = self._extract_entities(s)
            quantities = self._extract_quantities(s)
            fact_type = self._classify_type(s)

            # ── Substance gate ──
            # Must have at least one: named entity, quantity, or non-general type
            if not entities and not quantities and fact_type == "general":
                continue

            # ── Strip assistant preamble if it's the only substance ──
            if role == "assistant" and self.ASSISTANT_FILLER.match(s):
                # Only skip if there's nothing else of value
                if not entities and not quantities:
                    continue

            self._seen_hashes.add(content_hash)

            facts.append({
                "text": s,
                "source_role": role,
                "timestamp": datetime.now().isoformat(),
                "type": fact_type,
                "entities": entities,
                "quantities": quantities,
            })

        return facts

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences, handling currency and abbreviations."""
        # Protect common abbreviations and currency from splitting
        protected = text
        # Protect $50M. or $50M-$100M from splitting
        protected = re.sub(r'(\$[\d,.]+[MBK]?)([.-])', r'\1⌀\2', protected)
        # Protect e.g., i.e., etc.
        protected = re.sub(r'\b(e\.g|i\.e|etc|vs|approx|incl)\.\s', r'\1⌁ ', protected)
        # Protect A-, BBB+, etc. from splitting on trailing punctuation
        protected = re.sub(r'\b([A-Z]{1,3}[+-])[.,]\s', r'\1⌂ ', protected)

        # Split on real sentence boundaries
        parts = re.split(r'(?<=[.!?])\s+|\n+', protected)

        # Restore protected characters
        result = []
        for p in parts:
            p = p.replace('⌀', '').replace('⌁', '.').replace('⌂', ',')
            p = p.strip()
            if p:
                result.append(p)
        return result

    @classmethod
    def classify_type(cls, sentence: str) -> str:
        """Classify a sentence into a fact type. Classmethod so enricher can call it."""
        s_lower = sentence.lower()
        if any(m in s_lower for m in cls.DECISION_MARKERS):
            return "decision"
        if any(m in s_lower for m in cls.TASK_MARKERS):
            return "task"
        if any(m in s_lower for m in cls.PREFERENCE_MARKERS):
            return "preference"
        if cls.QUANTITY_PATTERN.search(sentence):
            return "quantitative"
        return "general"

    # Keep backward compat
    def _classify_type(self, sentence: str) -> str:
        return self.classify_type(sentence)

    def _extract_entities(self, sentence: str) -> list[str]:
        """Extract named entities using domain-specific patterns."""
        entities = set()
        for pattern in self.ENTITY_PATTERNS:
            for match in pattern.finditer(sentence):
                entity = match.group(1).strip()
                # Skip single-char or very short matches
                if len(entity) < 2:
                    continue
                # Skip standalone credit ratings that are just letters (A, B, etc.)
                if len(entity) <= 2 and not entity.endswith(('+', '-')):
                    continue
                # Skip common English words that aren't entities
                if entity in self.ENTITY_STOPWORDS:
                    continue
                entities.add(entity)
        return sorted(entities)

    def _extract_quantities(self, sentence: str) -> list[str]:
        return self.QUANTITY_PATTERN.findall(sentence)


# ── Embedding + Storage (CPU, sentence-transformers + ChromaDB) ──

class MemoryStore:
    """Embeds and stores facts in ChromaDB for semantic retrieval.

    v3: Temporal decay, contradiction detection + supersession, dedup.

    When a new fact about the same entities contradicts an existing one
    (high similarity but different quantities/values), the old fact is
    marked as superseded. Recall filters superseded facts automatically.
    """

    DEDUP_THRESHOLD = 0.95    # Skip if >95% similar to existing fact
    CONTRADICT_THRESHOLD = 0.70  # Check for contradiction if >70% similar
    CONTRADICT_CEILING = 0.94   # But below dedup threshold

    def __init__(self, db_path: str, collection_name: str = "conversation_memory",
                 embedding_model: str = "all-MiniLM-L6-v2"):
        self.emb_model = SentenceTransformer(embedding_model, device="cpu")
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        self._counter = self.collection.count()

    def store(self, fact: dict) -> Optional[str]:
        """Embed and store a single fact. Returns fact ID or None if duplicate.

        If the fact contradicts an existing one (same topic, different values),
        the old fact is marked as superseded.
        """
        embedding = self.emb_model.encode(
            [fact["text"]],
            normalize_embeddings=True,
            show_progress_bar=False
        )[0]

        # Dedup check: skip if near-duplicate exists
        if self._counter > 0 and self._is_duplicate(embedding):
            return None

        # Contradiction check: supersede old facts about the same topic
        superseded_ids = []
        if self._counter > 0:
            superseded_ids = self._check_contradictions(embedding, fact)

        self._counter += 1
        fact_id = f"fact_{self._counter}_{int(time.time())}"

        metadata = self._make_metadata(fact)
        if superseded_ids:
            metadata["supersedes"] = json.dumps(superseded_ids)

        self.collection.upsert(
            ids=[fact_id],
            embeddings=[embedding.tolist()],
            documents=[fact["text"]],
            metadatas=[metadata],
        )

        return fact_id

    def store_batch(self, facts: list[dict]) -> list[str]:
        """Embed and store multiple facts efficiently with true batch upsert."""
        if not facts:
            return []

        texts = [f["text"] for f in facts]
        embeddings = self.emb_model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=32,
        )

        # Filter duplicates
        ids, docs, embs, metas = [], [], [], []
        for fact, emb in zip(facts, embeddings):
            if self._counter > 0 and self._is_duplicate(emb):
                continue

            self._counter += 1
            fact_id = f"fact_{self._counter}_{int(time.time())}"
            ids.append(fact_id)
            docs.append(fact["text"])
            embs.append(emb.tolist())
            metas.append(self._make_metadata(fact))

        if not ids:
            return []

        # True batch upsert — single ChromaDB call
        self.collection.upsert(
            ids=ids,
            embeddings=embs,
            documents=docs,
            metadatas=metas,
        )

        return ids

    def recall(self, query: str, n_results: int = 5, type_filter: str = None,
               recency_weight: float = 0.15, include_superseded: bool = False) -> list[dict]:
        """Retrieve relevant facts via semantic search with temporal decay.

        v3: Superseded facts are filtered out by default. Temporal decay is
        more aggressive — 7-day half-life instead of 30-day linear decay.
        Recency weight increased from 0.1 to 0.15.
        """
        q_emb = self.emb_model.encode(
            [query], normalize_embeddings=True, show_progress_bar=False
        )[0]

        where_filter = {"type": type_filter} if type_filter else None

        # Fetch extra results to allow re-ranking + superseded filtering
        fetch_n = min(n_results * 5, max(n_results * 3, self.collection.count()))

        results = self.collection.query(
            query_embeddings=[q_emb.tolist()],
            n_results=fetch_n,
            where=where_filter,
        )

        recalled = []
        now = time.time()
        for i in range(len(results["documents"][0])):
            similarity = 1 - results["distances"][0][i]  # cosine distance → similarity
            meta = results["metadatas"][0][i]

            # ── Filter superseded facts ──
            if not include_superseded and meta.get("superseded_by"):
                continue

            # ── Temporal decay: exponential with 7-day half-life ──
            try:
                fact_time = datetime.fromisoformat(meta.get("timestamp", "")).timestamp()
                age_days = (now - fact_time) / 86400
                # Exponential decay: score halves every 7 days
                # Recent facts (~0 days) → 1.0, 7 days → 0.5, 14 days → 0.25, 30 days → 0.06
                recency_score = 2 ** (-age_days / 7)
            except (ValueError, TypeError):
                recency_score = 0.3  # Unknown age = low confidence

            combined_score = similarity * (1 - recency_weight) + recency_score * recency_weight

            recalled.append({
                "text": results["documents"][0][i],
                "similarity": similarity,
                "recency": round(recency_score, 4),
                "score": combined_score,
                "metadata": meta,
                "superseded": bool(meta.get("superseded_by")),
            })

        # Sort by combined score and return top N
        recalled.sort(key=lambda x: x["score"], reverse=True)
        return recalled[:n_results]

    def count(self) -> int:
        return self.collection.count()

    def get_by_type(self, fact_type: str, limit: int = 100, offset: int = 0) -> dict:
        """Get facts filtered by type. Used by enricher for batch processing."""
        return self.collection.get(
            where={"type": fact_type}, limit=limit, offset=offset,
            include=["documents", "metadatas", "embeddings"],
        )

    def get_all(self, limit: int = 500, offset: int = 0) -> dict:
        """Get all facts. Used by enricher for sweeps."""
        return self.collection.get(
            limit=limit, offset=offset,
            include=["documents", "metadatas"],
        )

    def _is_duplicate(self, embedding) -> bool:
        """Check if a near-duplicate exists in the store."""
        try:
            results = self.collection.query(
                query_embeddings=[embedding.tolist()],
                n_results=1,
            )
            if results["distances"][0]:
                similarity = 1 - results["distances"][0][0]
                return similarity >= self.DEDUP_THRESHOLD
        except Exception:
            pass
        return False

    def _check_contradictions(self, embedding, new_fact: dict) -> list[str]:
        """Detect and supersede contradicted facts.

        A contradiction is when:
        1. An existing fact is semantically similar (same topic, 70-94%)
        2. Both facts mention the same entities
        3. But they have DIFFERENT quantities (the value changed)

        Example: "threshold is $50M" superseded by "threshold increased to $75M"
        """
        try:
            results = self.collection.query(
                query_embeddings=[embedding.tolist()],
                n_results=5,
            )
        except Exception:
            return []

        if not results["distances"][0]:
            return []

        new_entities = set(new_fact.get("entities", []))
        new_quantities = set(new_fact.get("quantities", []))
        new_text = new_fact.get("text", "").lower()

        # Contradiction signals: quantity change, update language, or entity value change
        UPDATE_SIGNALS = [
            "increased", "decreased", "changed", "updated", "revised",
            "expanded", "reduced", "tightened", "downgraded", "upgraded",
            "renegotiated", "amended", "modified", "raised", "lowered",
        ]
        has_update_signal = any(s in new_text for s in UPDATE_SIGNALS)

        superseded = []
        for i in range(len(results["distances"][0])):
            similarity = 1 - results["distances"][0][i]

            # Must be in contradiction range (similar topic, but not exact duplicate)
            if similarity < self.CONTRADICT_THRESHOLD or similarity >= self.CONTRADICT_CEILING:
                continue

            meta = results["metadatas"][0][i]

            # Already superseded — skip
            if meta.get("superseded_by"):
                continue

            # Must share at least one entity
            try:
                old_entities = set(json.loads(meta.get("entities", "[]")))
            except (json.JSONDecodeError, TypeError):
                old_entities = set()

            if not (new_entities & old_entities):
                continue

            # Contradiction check
            try:
                old_quantities = set(json.loads(meta.get("quantities", "[]")))
            except (json.JSONDecodeError, TypeError):
                old_quantities = set()

            old_text = results["documents"][0][i].lower()
            is_contradiction = False

            # Case 1: Both have quantities and they differ
            if old_quantities and new_quantities and old_quantities != new_quantities:
                is_contradiction = True
            # Case 2: New fact has quantities, old doesn't (new info replaces vague old)
            elif new_quantities and not old_quantities and has_update_signal:
                is_contradiction = True
            # Case 3: Update language + shared entities (e.g., rating change, status change)
            elif has_update_signal and (new_entities & old_entities):
                is_contradiction = True
            # Case 4: High similarity (>85%) + shared entities + different text
            # Catches value replacements without explicit update language
            # e.g., "Valuation date: every Wednesday" → "Valuation date is every Thursday"
            elif similarity > 0.85 and (new_entities & old_entities) and new_text != old_text:
                is_contradiction = True

            if is_contradiction:
                # This is a contradiction — supersede the old fact
                old_id = results["ids"][0][i]
                superseded.append(old_id)

                # Mark old fact as superseded in ChromaDB
                meta["superseded_by"] = new_fact["text"][:200]
                meta["superseded_at"] = datetime.now().isoformat()
                self.collection.update(
                    ids=[old_id],
                    metadatas=[meta],
                )

        return superseded

    def _make_metadata(self, fact: dict) -> dict:
        return {
            "type": fact.get("type", "general"),
            "source_role": fact.get("source_role", "unknown"),
            "timestamp": fact.get("timestamp", datetime.now().isoformat()),
            "entities": json.dumps(fact.get("entities", [])),
            "quantities": json.dumps(fact.get("quantities", [])),
            "session": fact.get("session", "unknown"),
        }


# ── Vault Writer (Obsidian markdown, organized by type) ──────────

class VaultWriter:
    """Writes organized facts to Obsidian vault as structured markdown.

    v3: Clean entity filenames, wikilinks, AND contradiction tracking.
    When a fact is superseded, the vault file gets updated with a
    strikethrough on the old entry and a note pointing to the new one.
    """

    def __init__(self, vault_path: str):
        self.vault_path = vault_path
        self._ensure_structure()

    def _ensure_structure(self):
        """Create vault folder structure."""
        folders = [
            "memory/entities",
            "memory/facts",
            "memory/decisions",
            "memory/preferences",
            "memory/tasks",
            "memory/sessions",
        ]
        for folder in folders:
            os.makedirs(os.path.join(self.vault_path, folder), exist_ok=True)

    def write_fact(self, fact: dict, category: str = None):
        """Write a fact to the appropriate vault location."""
        fact_type = category or fact.get("type", "general")
        entities = fact.get("entities", [])
        text = fact["text"]
        timestamp = fact.get("timestamp", datetime.now().isoformat())

        # Add wikilinks to other entities in the text
        display_text = self._add_wikilinks(text, entities)

        # Route to appropriate file
        if fact_type == "decision":
            self._append_to_file("memory/decisions/decisions.md", display_text, timestamp)
        elif fact_type == "preference":
            self._append_to_file("memory/preferences/preferences.md", display_text, timestamp)
        elif fact_type == "task":
            self._append_to_file("memory/tasks/tasks.md", display_text, timestamp)
        else:
            self._append_to_file("memory/facts/general.md", display_text, timestamp)

        # Always write to entity pages too (regardless of type)
        if entities:
            for entity in entities:
                safe_name = self._entity_filename(entity)
                if safe_name:
                    self._append_to_file(
                        f"memory/entities/{safe_name}.md",
                        display_text, timestamp, entity_name=entity
                    )

    def write_session_summary(self, session_id: str, facts: list[dict]):
        """Write a session summary note."""
        filepath = os.path.join(self.vault_path, f"memory/sessions/{session_id}.md")

        lines = [
            f"# Session: {session_id}",
            f"*Generated: {datetime.now().isoformat()}*",
            "",
            f"## Facts Extracted ({len(facts)})",
            "",
        ]

        by_type = {}
        for f in facts:
            t = f.get("type", "general")
            by_type.setdefault(t, []).append(f)

        for fact_type, type_facts in sorted(by_type.items()):
            lines.append(f"### {fact_type.title()} ({len(type_facts)})")
            for f in type_facts:
                entities = f.get("entities", [])
                links = " ".join(f"[[{self._entity_filename(e)}|{e}]]" for e in entities
                                if self._entity_filename(e))
                lines.append(f"- {f['text']} {links}")
            lines.append("")

        with open(filepath, "w") as fh:
            fh.write("\n".join(lines))

    def _entity_filename(self, entity: str) -> str:
        """Convert entity name to a clean, readable filename.

        'cross-default threshold' → 'cross-default-threshold'
        'ISDA Master Agreement' → 'ISDA-Master-Agreement'
        'Counterparty X'        → 'Counterparty-X'
        'US Treasuries'         → 'US-Treasuries'
        'Section 5(a)'          → 'Section-5a'
        """
        # Remove parentheses content but keep the alphanumeric part
        name = re.sub(r'\(([a-z0-9]+)\)', r'\1', entity)
        # Replace spaces and slashes with hyphens
        name = re.sub(r'[\s/]+', '-', name)
        # Remove anything that isn't alphanumeric, hyphen, or plus/minus
        name = re.sub(r'[^\w+-]', '', name)
        # Collapse multiple hyphens
        name = re.sub(r'-{2,}', '-', name)
        # Strip leading/trailing hyphens
        name = name.strip('-')
        # Skip if too short or too long
        if len(name) < 2 or len(name) > 60:
            return ""
        return name

    def _add_wikilinks(self, text: str, entities: list[str]) -> str:
        """Add Obsidian [[wikilinks]] to entity mentions in text."""
        result = text
        for entity in entities:
            filename = self._entity_filename(entity)
            if filename and entity in result:
                result = result.replace(entity, f"[[{filename}|{entity}]]", 1)
        return result

    def supersede_in_vault(self, old_text: str, new_text: str, timestamp: str):
        """Mark an old fact as superseded across all vault files.

        Finds lines containing the old text snippet and adds strikethrough
        + a pointer to the replacement fact.
        """
        memory_dir = os.path.join(self.vault_path, "memory")
        # Normalize for matching — strip wikilinks from old_text
        old_clean = re.sub(r'\[\[[^\]|]+\|([^\]]+)\]\]', r'\1', old_text)
        # Use first 60 chars for matching (enough to be unique, handles truncation)
        match_prefix = old_clean[:60]

        for root, dirs, files in os.walk(memory_dir):
            for fname in files:
                if not fname.endswith(".md"):
                    continue
                filepath = os.path.join(root, fname)
                try:
                    with open(filepath, "r") as fh:
                        lines = fh.readlines()
                except Exception:
                    continue

                modified = False
                new_lines = []
                for line in lines:
                    # Check if this line contains the old fact
                    line_clean = re.sub(r'\[\[[^\]|]+\|([^\]]+)\]\]', r'\1', line)
                    if match_prefix in line_clean and not line.startswith("- ~~["):
                        # Strikethrough the old entry
                        stripped = line.rstrip('\n')
                        # Convert "- [date] text" → "- ~~[date] text~~ *(superseded)*"
                        new_lines.append(f"- ~~{stripped[2:]}~~ *(superseded {timestamp[:10]})*\n")
                        modified = True
                    else:
                        new_lines.append(line)

                if modified:
                    with open(filepath, "w") as fh:
                        fh.writelines(new_lines)

    def _append_to_file(self, rel_path: str, text: str, timestamp: str, entity_name: str = None):
        filepath = os.path.join(self.vault_path, rel_path)

        if not os.path.exists(filepath):
            title = entity_name or rel_path.split('/')[-1].replace('.md', '').replace('-', ' ').title()
            header = f"# {title}\n\n"
            with open(filepath, "w") as fh:
                fh.write(header)

        with open(filepath, "a") as fh:
            fh.write(f"- [{timestamp[:10]}] {text}\n")


# ── Memory Daemon (orchestrates all three tiers) ─────────────────

class MemoryDaemon:
    """
    Three-tier memory system for local LLMs.

    Tier 1 (CPU): Extract facts + embed into ChromaDB
    Tier 2 (ANE): Classify + organize into Obsidian (future: ANE model)
    Tier 3 (GPU): Conversation (external, not managed by daemon)
    """

    def __init__(self, vault_path: str, db_path: str = None, session_id: str = None,
                 embedding_model: str = "all-MiniLM-L6-v2",
                 enable_enricher: bool = False, enricher_interval: int = 300):
        self.session_id = session_id or datetime.now().strftime("%Y-%m-%d-%H%M")

        # Tier 1: CPU extraction + embedding
        self.extractor = FactExtractor()
        if db_path is None:
            db_path = os.path.join(os.path.dirname(vault_path), "memory", "chromadb")
        os.makedirs(db_path, exist_ok=True)
        self.store = MemoryStore(db_path, embedding_model=embedding_model)

        # Tier 2: Vault organization
        self.vault = VaultWriter(vault_path)

        # Tier 3: Enricher (optional, always-on background intelligence)
        self._enable_enricher = enable_enricher
        self._enricher_interval = enricher_interval
        self.enricher = None

        # Processing queue and thread
        self._queue = queue.Queue()
        self._running = False
        self._thread = None
        self._session_facts = []
        self._stats = {"ingested": 0, "extracted": 0, "stored": 0, "deduped": 0}

    def start(self):
        """Start the background memory processing daemon."""
        self._running = True
        self._thread = threading.Thread(target=self._process_loop, daemon=True)
        self._thread.start()

        # Start ANE server + enricher if enabled
        if self._enable_enricher:
            from phantom_memory.enricher import PhantomEnricher

            # Auto-launch ANE server if CoreML model exists
            classifier = None
            self._ane_process = None
            try:
                from phantom_memory.ane_server import ANEClient, SOCKET_PATH
                if not ANEClient.is_running():
                    self._ane_process = self._launch_ane_server()
                if ANEClient.is_running():
                    from phantom_memory.enricher import ANEClassifier
                    classifier = ANEClassifier()
                    print("[MemoryDaemon] ✓ ANE server connected — 1.7B on Neural Engine")
                else:
                    print("[MemoryDaemon] ANE server unavailable — using regex classifier")
            except Exception as e:
                print(f"[MemoryDaemon] ANE setup skipped: {e}")

            self.enricher = PhantomEnricher(
                store=self.store, vault=self.vault,
                interval=self._enricher_interval,
                classifier=classifier,
            )
            self.enricher.start()

    def _launch_ane_server(self):
        """Auto-launch the ANE server as a subprocess using the ANEMLL Python 3.9 venv.

        The ANE server needs Python 3.9 + CoreML (ANEMLL venv), while the daemon
        may run on Python 3.11 (mlx-env). This bridges the gap by spawning the
        server as a separate process and connecting via Unix socket.
        """
        import subprocess

        ANEMLL_PYTHON = os.path.expanduser("~/Desktop/cowork/anemll/env-anemll/bin/python3")
        ANE_SERVER = os.path.join(os.path.dirname(__file__), "ane_server.py")
        META_PATH = os.path.expanduser(
            "~/Desktop/cowork/anemll/models/qwen3-1.7b-coreml/meta.yaml"
        )

        if not os.path.exists(META_PATH):
            print("[MemoryDaemon] No CoreML model found — skipping ANE server")
            return None

        if not os.path.exists(ANEMLL_PYTHON):
            print("[MemoryDaemon] ANEMLL venv not found — skipping ANE server")
            return None

        print("[MemoryDaemon] Launching ANE server (Qwen3-1.7B on Neural Engine)...")
        proc = subprocess.Popen(
            [ANEMLL_PYTHON, ANE_SERVER, "--meta", META_PATH],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,  # Won't die when parent terminal closes
        )

        # Wait for server to be ready (model load + warmup)
        from phantom_memory.ane_server import ANEClient
        for i in range(30):  # 30s timeout
            time.sleep(1)
            if ANEClient.is_running():
                print(f"[MemoryDaemon] ANE server ready (PID {proc.pid}, {i+1}s)")
                return proc
            if proc.poll() is not None:
                print(f"[MemoryDaemon] ANE server exited with code {proc.returncode}")
                return None

        print("[MemoryDaemon] ANE server timed out after 30s")
        proc.kill()
        return None

    def stop(self):
        """Stop the daemon, enricher, and ANE server."""
        self._running = False
        self._queue.put(None)  # Sentinel to unblock
        if self._thread:
            self._thread.join(timeout=5)

        # Stop enricher
        if self.enricher:
            self.enricher.stop()

        # Stop ANE server if we launched it
        if hasattr(self, '_ane_process') and self._ane_process:
            self._ane_process.terminate()
            self._ane_process.wait(timeout=5)
            print("[MemoryDaemon] ANE server stopped")

        # Write session summary
        if self._session_facts:
            self.vault.write_session_summary(self.session_id, self._session_facts)

    def ingest(self, role: str, text: str):
        """Feed a conversation turn to the daemon. Non-blocking."""
        self._queue.put({"role": role, "text": text})
        self._stats["ingested"] += 1

    def recall(self, query: str, n_results: int = 5) -> list[dict]:
        """Retrieve relevant memories for context injection."""
        return self.store.recall(query, n_results=n_results)

    def recall_formatted(self, query: str, n_results: int = 5) -> str:
        """Retrieve and format memories for LLM context injection."""
        memories = self.recall(query, n_results)
        if not memories:
            return ""

        lines = ["## Relevant Memories from Previous Sessions\n"]
        for m in memories:
            meta = m["metadata"]
            lines.append(f"- [{meta.get('type', '?')}] {m['text']} (score={m['score']:.2f})")

        return "\n".join(lines)

    @property
    def stats(self):
        return {
            **self._stats,
            "total_memories": self.store.count(),
            "superseded": self._stats.get("superseded", 0),
        }

    def _process_loop(self):
        """Background processing loop with contradiction detection."""
        while self._running:
            try:
                item = self._queue.get(timeout=1)
            except queue.Empty:
                continue

            if item is None:
                break

            # Tier 1: Extract facts
            facts = self.extractor.extract(item["text"], role=item["role"])

            for fact in facts:
                fact["session"] = self.session_id

                # Tier 1: Embed and store in ChromaDB (with dedup + contradiction)
                fact_id = self.store.store(fact)
                if fact_id:
                    self._stats["stored"] += 1

                    # Check if this fact superseded anything
                    # (store() already marked old facts in ChromaDB)
                    # Now update the vault files too
                    try:
                        meta = self.store.collection.get(ids=[fact_id])
                        if meta and meta["metadatas"]:
                            supersedes_json = meta["metadatas"][0].get("supersedes", "[]")
                            superseded_ids = json.loads(supersedes_json)
                            if superseded_ids:
                                # Get the old fact texts to mark them in vault
                                old_facts = self.store.collection.get(ids=superseded_ids)
                                if old_facts and old_facts["documents"]:
                                    for old_text in old_facts["documents"]:
                                        self.vault.supersede_in_vault(
                                            old_text, fact["text"],
                                            fact.get("timestamp", datetime.now().isoformat())
                                        )
                                self._stats.setdefault("superseded", 0)
                                self._stats["superseded"] += len(superseded_ids)
                    except Exception:
                        pass  # Non-critical — vault update is best-effort

                    # Tier 2: Write to Obsidian vault
                    self.vault.write_fact(fact)
                    self._session_facts.append(fact)
                else:
                    self._stats["deduped"] += 1

            self._stats["extracted"] += len(facts)


# ── CLI Demo ─────────────────────────────────────────────────────

def demo():
    """Run a demo showing the full memory pipeline."""
    import shutil

    vault_path = "/Users/midas/Desktop/cowork/vault"
    db_path = "/Users/midas/Desktop/cowork/orion-ane/memory/chromadb_demo_v2"

    # Clean previous demo data
    if os.path.exists(db_path):
        shutil.rmtree(db_path)

    # Clean previous vault memory files
    vault_memory = os.path.join(vault_path, "memory")
    if os.path.exists(vault_memory):
        shutil.rmtree(vault_memory)

    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  MEMORY DAEMON v2 — REFINED                                   ║")
    print("║  Domain entities • Noise filtering • Dedup • Clean vault      ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print()

    # Start daemon
    daemon = MemoryDaemon(vault_path=vault_path, db_path=db_path)
    daemon.start()
    print(f"  Daemon started (session: {daemon.session_id})")
    print()

    # Simulate a conversation
    conversation = [
        ("user", "I'm reviewing the ISDA Master Agreement for Counterparty X. They're a BBB+ rated hedge fund based in London."),
        ("assistant", "I can help with that review. A BBB+ rated hedge fund will have specific considerations around credit support provisions and rating-dependent thresholds. What sections would you like to focus on?"),
        ("user", "The cross-default threshold is set at $50 million and includes affiliate indebtedness. I think that's too low given their AUM of $2 billion."),
        ("assistant", "I agree — a $50M threshold against $2B AUM represents only 2.5% of assets, which is below the typical 3-5% range for BBB-rated counterparties. I'd recommend negotiating to at least $75M-$100M. The affiliate inclusion is also aggressive; consider limiting it to material subsidiaries only."),
        ("user", "The CSA specifies zero threshold below A-, with eligible collateral limited to cash only. We prefer to include US Treasuries with a 2% haircut."),
        ("assistant", "Zero threshold below A- is market standard for hedge fund counterparties. The cash-only collateral restriction is overly conservative. Standard practice for a BBB+ counterparty would include G7 government securities. I'd push for: cash in USD/EUR/GBP, plus US Treasuries at 2% haircut and Agency securities at 4% haircut."),
        ("user", "We decided to go with the $75M cross-default threshold and add Treasuries. I need to draft the counter-proposal by Friday."),
        ("assistant", "I'll note those decisions. Key counter-proposal points: (1) Cross-default threshold increased from $50M to $75M, (2) Eligible collateral expanded to include US Treasuries at 2% haircut, (3) Consider also proposing the affiliate indebtedness limitation to material subsidiaries."),
    ]

    # Also test noise — these should NOT produce facts
    noise = [
        ("user", "Hello, can you help me?"),
        ("user", "Sure, sounds good"),
        ("user", "Ok thanks"),
        ("assistant", "Let me help you with that. What would you like to know?"),
    ]

    print("Ingesting conversation (8 turns)...")
    for role, text in conversation:
        daemon.ingest(role, text)
        time.sleep(0.1)

    print("Ingesting noise (4 turns — should be filtered)...")
    for role, text in noise:
        daemon.ingest(role, text)
        time.sleep(0.1)

    # Wait for processing
    time.sleep(2)

    stats = daemon.stats
    print(f"\n  Stats: {stats}")
    print(f"  → {stats['extracted']} facts extracted from {stats['ingested']} turns")
    print(f"  → {stats['stored']} stored, {stats['deduped']} deduped")
    print()

    # Test recall
    print("═" * 70)
    print("RECALL TEST: Retrieving memories")
    print("═" * 70)
    print()

    queries = [
        "What is the cross-default threshold for Counterparty X?",
        "What collateral is eligible?",
        "What decisions were made?",
        "What is the deadline?",
        "What is the counterparty's credit rating?",
    ]

    for query in queries:
        print(f"  Q: {query}")
        memories = daemon.recall(query, n_results=2)
        for m in memories:
            print(f"    → [{m['metadata']['type']}] {m['text'][:80]}  (score={m['score']:.3f})")
        print()

    # Test formatted context injection
    print("═" * 70)
    print("CONTEXT INJECTION: What would be injected into next session")
    print("═" * 70)
    print()

    context = daemon.recall_formatted("ISDA agreement Counterparty X thresholds collateral")
    print(context)
    print()

    # Stop daemon and write session summary
    daemon.stop()

    # Show vault files created
    print("═" * 70)
    print("VAULT FILES CREATED")
    print("═" * 70)
    for root, dirs, files in os.walk(os.path.join(vault_path, "memory")):
        for f in sorted(files):
            if f.endswith(".md"):
                filepath = os.path.join(root, f)
                rel = os.path.relpath(filepath, vault_path)
                size = os.path.getsize(filepath)
                print(f"  {rel} ({size} bytes)")

    print()
    print("Done. v2 memory daemon: cleaner entities, noise filtered, deduped.")


if __name__ == "__main__":
    demo()
