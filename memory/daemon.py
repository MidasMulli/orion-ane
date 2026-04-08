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
    daemon.ingest("user", "What's the 8B tok/s on ANE in production?")
    daemon.ingest("assistant", "Llama-3.1-8B Q8 runs at 7.9 tok/s with 72 CoreML dispatches.")

    # At next session start, retrieve relevant context
    context = daemon.recall("8B tok/s on ANE")
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

    # ─── Entities (Apple Silicon ML domain) ───
    # Main 26 housekeeping: replaced original domain-specific (banking) patterns
    # with Track 1 patterns covering models, hardware, dispatch counts, and the
    # Subconscious memory architecture vocabulary the project actually uses.
    ENTITY_PATTERNS = [
        # Model families
        re.compile(r'\b(Llama[- ]?3(?:\.\d)?[- ]?(?:1|3|8|70)B(?:[- ]Instruct)?|GPT[- ]?2|Qwen[- ]?\d(?:\.\d)?[- ]?\d{1,3}(?:\.\d)?B?(?:[- ]Instruct)?|MiniLM[- ]?L\d+[- ]v\d+|Neuron(?:[- ]?\d+M)?)\b', re.IGNORECASE),
        # Quantization levels
        re.compile(r'\b(Q[3-8]|FP16|BF16|INT[48])\b'),
        # Hardware accelerators / silicon
        re.compile(r'\b(ANE|AMX|SME|GPU|CPU|SLC|DMA|MCC|SRAM|DRAM|NAX)\b'),
        re.compile(r'\b(M[345][- ]?(?:Pro|Max|Ultra)?)\b'),
        # Dispatch / kernel terminology
        re.compile(r'\b(\d+\s*dispatch(?:es)?|\d+[- ]?layer|head[- ]dim\s*\d+)\b', re.IGNORECASE),
        # Performance metrics (units only — numerics are caught by QUANTITY_PATTERN)
        re.compile(r'\b(tok/s|GB/s|ms/tok|TFLOPS|GFLOPS|GFLOPs?|us|µs)\b', re.IGNORECASE),
        # Subconscious / agent components
        re.compile(r'\b(Subconscious|LocalMemoryStore|CoreML|MIL\s*IR|espresso|mlpackage|hwx|ane[- ]compiler|ane[- ]dispatch|ane[- ]toolkit|midas[- ]ui|memory[_ ]bridge|canonical[_ ]inject|meta[_ ]memory|multi[_ ]path|vault[_ ]sweep)\b', re.IGNORECASE),
        # Standard ports/services (project conventions)
        re.compile(r'\b(:8\d{3}|port\s*8\d{3})\b', re.IGNORECASE),
        # Spec decode terminology
        re.compile(r'\b(spec[- ]decode|n[- ]gram|drafter|verifier|prompt[- ]?cache|EAGLE[- ]?\d?|Medusa|MTP|PARD)\b', re.IGNORECASE),
        # General proper nouns — capitalized words not in common-word stoplist
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
        "Meeting", "Company", "Group", "System", "Project",
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
        """Split text into claim-level segments using Subconscious splitter.

        Enhanced with claim_splitter for better decomposition of data-heavy
        text (comma-separated lists, colon-data patterns, etc.).
        Falls back to regex sentence splitting if claim_splitter not available.
        """
        try:
            # Use Subconscious claim_splitter for better decomposition
            extractor_dir = os.path.join(
                os.path.dirname(__file__), "..", "..", "vault", "subconscious", "extractor")
            if os.path.exists(extractor_dir):
                import importlib.util
                spec = importlib.util.spec_from_file_location(
                    "claim_splitter",
                    os.path.join(extractor_dir, "claim_splitter.py"))
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                claims = mod.split_claims(text)
                return [c['text'] for c in claims if len(c['text']) > 10]
        except Exception:
            pass

        # Fallback: original sentence splitting
        protected = text
        protected = re.sub(r'(\$[\d,.]+[MBK]?)([.-])', r'\1⌀\2', protected)
        protected = re.sub(r'\b(e\.g|i\.e|etc|vs|approx|incl)\.\s', r'\1⌁ ', protected)
        protected = re.sub(r'\b([A-Z]{1,3}[+-])[.,]\s', r'\1⌂ ', protected)

        parts = re.split(r'(?<=[.!?])\s+|\n+', protected)

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

    DEDUP_THRESHOLD = 0.85    # Skip if >85% similar to existing fact
    CONTRADICT_THRESHOLD = 0.70  # Check for contradiction if >70% similar
    CONTRADICT_CEILING = 0.94   # But below dedup threshold

    EMB_DIM = 384
    MAX_CACHE = 50000

    def __init__(self, db_path: str, collection_name: str = "conversation_memory",
                 embedding_model: str = "all-MiniLM-L6-v2"):
        self.emb_model = SentenceTransformer(embedding_model, device="cpu")
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        self._counter = self.collection.count()

        # Numpy embedding cache for sub-ms dedup via AMX BLAS
        self._emb_cache = np.zeros((self.MAX_CACHE, self.EMB_DIM), dtype=np.float32)
        self._cache_count = 0
        self._load_embedding_cache()

    def _load_embedding_cache(self):
        """Load all embeddings into numpy cache for fast AMX dedup."""
        try:
            all_data = self.collection.get(include=["embeddings"])
            n = min(len(all_data["ids"]), self.MAX_CACHE)
            for i in range(n):
                emb = all_data["embeddings"][i]
                if emb is not None and len(emb) == self.EMB_DIM:
                    self._emb_cache[self._cache_count] = np.array(emb, dtype=np.float32)
                    self._cache_count += 1
        except Exception:
            pass

    def _fast_dedup_check(self, embedding: np.ndarray) -> bool:
        """Sub-ms dedup via numpy dot product (AMX-accelerated BLAS).

        Compares against full embedding cache. At 3,800 × 384 = 5.8MB,
        fits in L2 cache. AMX at 900 GB/s does this in microseconds.
        """
        if self._cache_count == 0:
            return False
        # Normalized embeddings: dot product = cosine similarity
        sims = embedding @ self._emb_cache[:self._cache_count].T
        return float(sims.max()) >= self.DEDUP_THRESHOLD

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

        # Fast dedup via numpy cache (sub-ms AMX path)
        if self._cache_count > 0 and self._fast_dedup_check(embedding):
            self._stats_deduped = getattr(self, '_stats_deduped', 0) + 1
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

        # Add to numpy cache for future dedup checks
        if self._cache_count < self.MAX_CACHE:
            self._emb_cache[self._cache_count] = embedding.astype(np.float32)
            self._cache_count += 1

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

        # Filter duplicates and detect contradictions
        ids, docs, embs, metas = [], [], [], []
        for fact, emb in zip(facts, embeddings):
            if self._counter > 0 and self._is_duplicate(emb):
                continue

            # Check for contradictions (same logic as single store())
            superseded_ids = []
            if self._counter > 0:
                superseded_ids = self._check_contradictions(emb, fact)

            self._counter += 1
            fact_id = f"fact_{self._counter}_{int(time.time())}"
            ids.append(fact_id)
            docs.append(fact["text"])
            embs.append(emb.tolist())
            meta = self._make_metadata(fact)
            if superseded_ids:
                meta["supersedes"] = json.dumps(superseded_ids)
            metas.append(meta)

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
               recency_weight: float = 0.25, include_superseded: bool = False) -> list[dict]:
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
        fetch_n = min(n_results * 5, max(n_results * 3, self._counter))

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

            # Build 3 (Main 22): canonical-state boost.
            # Memories tagged source_role=canonical describe the current
            # production ground truth (Production Critical Path entries
            # injected from CLAUDE.md). They should outrank noise memories
            # on the same topic. The 1.30x multiplier is enough to push a
            # 0.50-similarity canonical above a 0.60-similarity noise memory
            # while still requiring the canonical to be topically relevant.
            if meta.get("source_role") == "canonical":
                combined_score *= 1.30

            recalled.append({
                "text": results["documents"][0][i],
                "similarity": similarity,
                "recency": round(recency_score, 4),
                "score": combined_score,
                "metadata": meta,
                "superseded": bool(meta.get("superseded_by")),
            })

        # Sort by combined score
        recalled.sort(key=lambda x: x["score"], reverse=True)

        # ── Query expansion: if top results are weak, do keyword fallback ──
        top_score = recalled[0]["score"] if recalled else 0
        if top_score < 0.45 and len(query.split()) >= 3:
            # Extract keywords (>3 chars, not stopwords)
            stopwords = {"what", "when", "where", "which", "how", "does", "did",
                         "the", "about", "with", "from", "that", "this", "have",
                         "been", "were", "was", "are", "our", "should", "would",
                         "could", "make", "tell", "give", "know"}
            keywords = [w.lower() for w in query.split()
                       if len(w) > 2 and w.lower().rstrip("?.,!") not in stopwords]

            if keywords:
                # Search by keyword in document text
                try:
                    all_data = self.collection.get(
                        include=["documents", "metadatas"], limit=500)
                    keyword_hits = []
                    for idx in range(len(all_data["ids"])):
                        meta = all_data["metadatas"][idx]
                        if not include_superseded and meta.get("superseded_by"):
                            continue
                        doc = all_data["documents"][idx].lower()
                        hits = sum(1 for kw in keywords if kw in doc)
                        if hits >= 2:  # At least 2 keyword matches
                            # Check not already in recalled
                            fid = all_data["ids"][idx]
                            if not any(r.get("metadata", {}).get("id") == fid for r in recalled):
                                keyword_hits.append({
                                    "text": all_data["documents"][idx],
                                    "similarity": 0.3 + (hits * 0.05),
                                    "recency": 0.5,
                                    "score": 0.35 + (hits * 0.05),
                                    "metadata": meta,
                                    "superseded": False,
                                    "source": "keyword_expansion",
                                })
                    # Add keyword hits, re-sort
                    recalled.extend(keyword_hits[:5])
                    recalled.sort(key=lambda x: x["score"], reverse=True)
                except Exception:
                    pass

        return recalled[:n_results]

    def count(self) -> int:
        return self._counter

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

        'Llama 3.1 8B'   → 'Llama-3.1-8B'
        'Qwen2.5-72B'    → 'Qwen2.5-72B'
        'M5 Pro'         → 'M5-Pro'
        'tok/s'          → 'tok-s'
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
        # Main 24 Build 0 (revised): use LocalMemoryStore (SQLite + numpy)
        # instead of the chromadb-backed MemoryStore. Same db_path, but the
        # local store creates a `memory_local.db` SQLite file inside it. Drop-in
        # API; existing call sites use `self.store.collection.*` via the shim.
        try:
            from local_store import LocalMemoryStore
        except ImportError:
            from phantom_memory.local_store import LocalMemoryStore
        self.store = LocalMemoryStore(db_path, embedding_model=embedding_model)

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
            try:
                from phantom_memory.enricher import PhantomEnricher
            except ImportError:
                from enricher import PhantomEnricher

            # Auto-launch ANE server if CoreML model exists
            classifier = None
            self._ane_process = None
            try:
                try:
                    from phantom_memory.ane_server import ANEClient, SOCKET_PATH
                except ImportError:
                    from ane_server import ANEClient, SOCKET_PATH
                if not ANEClient.is_running():
                    self._ane_process = self._launch_ane_server()
                if ANEClient.is_running():
                    try:
                        from phantom_memory.enricher import ANEClassifier
                    except ImportError:
                        from enricher import ANEClassifier
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
        try:
            from phantom_memory.ane_server import ANEClient
        except ImportError:
            from ane_server import ANEClient
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
            "ane_queue_depth": getattr(self, '_ane_queue_depth', 0),
        }

    def _extract_via_ane(self, text: str, role: str) -> Optional[list]:
        """Extract facts via 8B Q8 on ANE (HTTP to extraction server).

        Calls the 8B extraction server running as a separate process.
        Falls back to None if server is down.
        """
        import urllib.request

        # Use the 8B extraction prompt with typed output
        prompt = (
            "Extract every important fact from this text. Include:\n"
            "- Measurements and numbers (exactly as stated)\n"
            "- Decisions (\"we decided...\", \"killed because...\", \"parked...\")\n"
            "- Architectural insights (\"X is actually Y\", \"the real role is...\")\n"
            "- Relationships (\"X feeds Y\", \"X replaces Y\")\n"
            "- Preferences (\"always do X\", \"never do Y\")\n\n"
            "Each fact is one complete sentence. Classify each as:\n"
            "  quantitative, decision, preference, relationship, conceptual, or fact.\n\n"
            "Format each line as: [TYPE] fact sentence\n\n"
            f"TEXT:\n---\n{text[:2000]}\n---\n\n"
            "Extracted facts:\n- "
        )

        try:
            data = json.dumps({"prompt": prompt, "max_tokens": 400}).encode()
            req = urllib.request.Request(
                "http://localhost:8891/analyze", data=data,
                headers={"Content-Type": "application/json"})
            resp = urllib.request.urlopen(req, timeout=120)  # 8B needs ~80s
            result = json.loads(resp.read())
            raw = result.get("result", "").strip()

            import re
            facts = []
            for line in raw.split("\n"):
                line = line.strip()
                # Strip bullet prefix
                for pfx in ["- ", "* ", "• ", "· "]:
                    if line.startswith(pfx):
                        line = line[len(pfx):]
                        break
                else:
                    m = re.match(r'^\d+[\.\)]\s+', line)
                    if m:
                        line = line[m.end():]
                    elif line.startswith(("•", "-", "*")):
                        line = line[1:].strip()
                    else:
                        if len(line) < 20:
                            continue

                # Extract [TYPE] prefix
                fact_type = "general"
                type_match = re.match(r'^\[(\w+)\]\s*', line)
                if type_match:
                    t = type_match.group(1).lower()
                    if t in ("quantitative", "decision", "preference",
                             "relationship", "conceptual", "fact"):
                        fact_type = t
                    line = line[type_match.end():]

                content = line.strip().rstrip(".")
                if len(content) >= 15:
                    facts.append({
                        "text": content,
                        "source_role": role,
                        "timestamp": datetime.now().isoformat(),
                        "type": fact_type,
                        "entities": [],
                        "quantities": [],
                        "extraction_source": "ane_8b",
                    })

            return facts if facts else None

        except Exception:
            return None

    def _ane_extract_worker(self, text, role):
        """Background worker for 8B ANE extraction. Non-blocking."""
        try:
            ane_facts = self._extract_via_ane(text, role)
            if ane_facts:
                for fact in ane_facts:
                    fact["session"] = self.session_id
                    fact_id = self.store.store(fact)
                    if fact_id:
                        self._stats["stored"] += 1
                        self._stats["extracted"] += 1
        except Exception:
            pass
        finally:
            self._ane_queue_depth = max(0, getattr(self, '_ane_queue_depth', 0) - 1)

    def _process_loop(self):
        """Background processing loop — CPU extraction immediate, 8B async."""
        self._ane_queue_depth = 0

        while self._running:
            try:
                item = self._queue.get(timeout=1)
            except queue.Empty:
                continue

            if item is None:
                break

            # IMMEDIATE: CPU FactExtractor (ms, never blocks)
            facts = self.extractor.extract(item["text"], role=item["role"])

            # Filter: skip questions and volatile metrics
            facts = [f for f in facts
                     if not f["text"].strip().endswith("?")
                     and not any(v in f["text"].lower() for v in
                                ["currently running", "server is up", "uptime",
                                 "as of right now", "at this moment"])]

            # ASYNC: 8B ANE extraction in separate thread (doesn't block queue)
            if self._ane_queue_depth < 10:  # Backpressure: cap at 10 pending
                self._ane_queue_depth += 1
                t = threading.Thread(
                    target=self._ane_extract_worker,
                    args=(item["text"], item["role"]),
                    daemon=True)
                t.start()

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

    # Simulate a conversation (Apple Silicon ML research domain)
    conversation = [
        ("user", "Started benchmarking Llama-3.1-8B Q8 on the ANE today. I want to see how it compares to GPT-2 on dispatch count."),
        ("assistant", "Good baseline target. GPT-2 sits at 229 tok/s with 25 dispatches via the ane-compiler MIL IR path. Llama-8B is much heavier — 32 transformer layers means at least 32 attention dispatches plus 32 FFN, plus token embed and lm_head. Expect 50-80 dispatches and a much lower tok/s."),
        ("user", "Confirmed. 72 dispatches end-to-end, 7.9 tok/s warm. Used FP32 residual accumulation between layers. Q8 weights, 7.2 GB on disk."),
        ("assistant", "FP32 residual is the right call at dim 4096 — FP16 accumulation produces garbage past 16 layers. 72 dispatches breaks down as 32 pre + 32 post + 8 lm_head shards. The 7.9 tok/s is dispatch-floor-bound, not compute-bound."),
        ("user", "We decided to ship 8B Q8 as the production extraction model on port 8891. The Subconscious will use it for typed fact extraction."),
        ("assistant", "Recording the decision. 8B Q8 on ANE @ 7.9 tok/s for typed extraction, port 8891. CPU FactExtractor stays as the fallback merge path — 83% combined recall vs 76% solo per the Tier 1/2 boundary measurement."),
    ]

    # Also test noise — these should NOT produce facts
    noise = [
        ("user", "Hello, can you help me?"),
        ("user", "Sure, sounds good"),
        ("user", "Ok thanks"),
        ("assistant", "Let me help you with that. What would you like to know?"),
    ]

    print("Ingesting conversation (6 turns)...")
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
        "What is the 8B tok/s on ANE?",
        "How many dispatches does Llama-8B use?",
        "What decisions were made?",
        "Why FP32 residual accumulation?",
        "What is the production extraction model?",
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

    context = daemon.recall_formatted("Llama-8B Q8 ANE dispatches tok/s")
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
