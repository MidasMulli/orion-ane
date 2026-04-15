"""
LocalMemoryStore — SQLite + numpy replacement for the chromadb-backed MemoryStore.

Main 24 Build 0 (revised): chromadb's HNSW/segment/lock layer keeps wedging on
maintenance writes (Track 2 purge corrupted internal indices, Rust binding
deadlocks on `get_collection`). This file replaces the entire backend with
something we control end-to-end:

  • SQLite (WAL mode, set by us) for atom fields, metadata, and embedding bytes
  • Numpy float32 matrix in memory for cosine similarity (loaded once on init)
  • Drop-in API matching the original `daemon.MemoryStore`
  • A `_CollectionShim` that exposes the chromadb collection methods used by
    enricher.py / idle_queue.py / cli.py / eval_tiers.py so existing call
    sites work unchanged.

384-dim MiniLM-L6-v2 embeddings, same as before. ~7.5 MB matrix for 4,879
memories. Cosine via single matmul = sub-millisecond.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from datetime import datetime
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer


EMB_DIM = 384

SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id                  TEXT PRIMARY KEY,
    text                TEXT NOT NULL,
    embedding           BLOB NOT NULL,
    type                TEXT,
    source_role         TEXT,
    timestamp           TEXT,
    session             TEXT,
    entities_json       TEXT,
    quantities_json     TEXT,
    atom_type           TEXT,
    atom_entities_json  TEXT,
    atom_impacts_json   TEXT,
    atom_tense          TEXT,
    atom_confidence     REAL,
    atom_core           TEXT,
    atom_schema_version INTEGER,
    atom_migrated_at    TEXT,
    atom_replaces       TEXT,
    superseded_by       TEXT,
    superseded_at       TEXT,
    supersedes          TEXT,
    file                TEXT,
    source              TEXT,
    relevance_score     REAL,
    topic               TEXT,
    extra_json          TEXT
);
CREATE INDEX IF NOT EXISTS idx_type        ON memories(type);
CREATE INDEX IF NOT EXISTS idx_atom_type   ON memories(atom_type);
CREATE INDEX IF NOT EXISTS idx_source_role ON memories(source_role);
CREATE INDEX IF NOT EXISTS idx_superseded  ON memories(superseded_by);
CREATE INDEX IF NOT EXISTS idx_topic       ON memories(topic);
"""

# Columns we lift to dedicated SQL fields. Anything else from the input
# metadata dict is JSON-serialized into extra_json.
KNOWN_META_FIELDS = {
    "type", "source_role", "timestamp", "session",
    "entities", "quantities",
    "atom_type", "atom_entities", "atom_impacts", "atom_tense",
    "atom_confidence", "atom_core", "atom_schema_version",
    "atom_migrated_at", "atom_replaces",
    "superseded_by", "superseded_at", "supersedes",
    "file", "source", "relevance_score", "topic",
}

# Reserved at-rest column → metadata-key mapping for SELECT * → meta dict.
COL_TO_META_KEY = {
    "entities_json":      "entities",
    "quantities_json":    "quantities",
    "atom_entities_json": "atom_entities",
    "atom_impacts_json":  "atom_impacts",
}


# ─────────────────────────────────────────────────────────────────────────────
# Connection helper (one connection per call — sqlite3 is fast enough that
# this beats fighting threading.local + WAL across writers)
# ─────────────────────────────────────────────────────────────────────────────
def _connect(path: str) -> sqlite3.Connection:
    c = sqlite3.connect(path, timeout=30, isolation_level=None,
                        check_same_thread=False)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("PRAGMA synchronous=NORMAL")
    c.execute("PRAGMA busy_timeout=30000")
    return c


# ─────────────────────────────────────────────────────────────────────────────
# Collection shim — exposes the subset of chromadb's collection API that
# existing code paths in this repo actually call (enricher, idle_queue, cli,
# eval_tiers, daemon's vault-supersede block).
# ─────────────────────────────────────────────────────────────────────────────
class _CollectionShim:
    def __init__(self, store: "LocalMemoryStore"):
        self._store = store

    # ── reads ──
    def count(self) -> int:
        return self._store.count()

    def get(self, ids=None, where=None, limit=None, offset=0, include=None):
        s = self._store
        with s._lock:
            c = _connect(s.db_path)
            try:
                sql = "SELECT * FROM memories"
                params: list = []
                clauses = []
                if ids:
                    placeholders = ",".join("?" * len(ids))
                    clauses.append(f"id IN ({placeholders})")
                    params.extend(ids)
                if where:
                    for k, v in where.items():
                        clauses.append(f"{k} = ?")
                        params.append(v)
                if clauses:
                    sql += " WHERE " + " AND ".join(clauses)
                if limit is not None:
                    sql += " LIMIT ? OFFSET ?"
                    params.extend([limit, offset])
                rows = c.execute(sql, params).fetchall()
            finally:
                c.close()

        # Preserve `ids` order if requested explicitly
        if ids:
            row_by_id = {r["id"]: r for r in rows}
            rows = [row_by_id[i] for i in ids if i in row_by_id]

        out_ids, docs, metas, embs = [], [], [], []
        for r in rows:
            out_ids.append(r["id"])
            docs.append(r["text"])
            metas.append(s._row_to_meta(r))
            if include and "embeddings" in include:
                embs.append(np.frombuffer(r["embedding"], dtype=np.float32).tolist())
        result = {"ids": out_ids, "documents": docs, "metadatas": metas}
        if include and "embeddings" in include:
            result["embeddings"] = embs
        return result

    def query(self, query_embeddings, n_results=5, where=None):
        # Single-query batch (chromadb's wire shape: lists-of-lists)
        s = self._store
        if not s._ids:
            return {"ids": [[]], "distances": [[]], "documents": [[]], "metadatas": [[]]}
        q = np.asarray(query_embeddings[0], dtype=np.float32)
        # Normalize defensively
        norm = np.linalg.norm(q)
        if norm > 0:
            q = q / norm
        with s._lock:
            mat = s._emb_matrix
            sims = mat @ q  # cosine since both are unit-normalized
        n = min(n_results, len(s._ids))
        if n == 0:
            return {"ids": [[]], "distances": [[]], "documents": [[]], "metadatas": [[]]}
        # Top-n by similarity
        top = np.argpartition(-sims, n - 1)[:n]
        top = top[np.argsort(-sims[top])]

        ids = [s._ids[i] for i in top]
        with s._lock:
            c = _connect(s.db_path)
            try:
                placeholders = ",".join("?" * len(ids))
                rows = {r["id"]: r for r in c.execute(
                    f"SELECT * FROM memories WHERE id IN ({placeholders})", ids
                ).fetchall()}
            finally:
                c.close()

        # Apply where filter post-fetch (cheap; n is small)
        out_ids, out_docs, out_metas, out_dists = [], [], [], []
        for i, fid in zip(top, ids):
            r = rows.get(fid)
            if r is None:
                continue
            meta = s._row_to_meta(r)
            if where:
                if not all(meta.get(k) == v for k, v in where.items()):
                    continue
            out_ids.append(fid)
            out_docs.append(r["text"])
            out_metas.append(meta)
            out_dists.append(float(1.0 - sims[i]))  # cosine distance
        return {
            "ids": [out_ids],
            "distances": [out_dists],
            "documents": [out_docs],
            "metadatas": [out_metas],
        }

    # ── writes ──
    def upsert(self, ids, embeddings, documents, metadatas):
        self._store._upsert_batch(ids, embeddings, documents, metadatas)

    def add(self, ids, embeddings, documents, metadatas):
        self.upsert(ids, embeddings, documents, metadatas)

    def update(self, ids, metadatas=None, documents=None, embeddings=None):
        self._store._update_batch(ids, metadatas, documents, embeddings)

    def delete(self, ids=None, where=None):
        self._store._delete(ids=ids, where=where)


# ─────────────────────────────────────────────────────────────────────────────
# LocalMemoryStore — drop-in for daemon.MemoryStore
# ─────────────────────────────────────────────────────────────────────────────
class LocalMemoryStore:
    DEDUP_THRESHOLD = 0.85
    CONTRADICT_THRESHOLD = 0.70
    CONTRADICT_CEILING = 0.94
    EMB_DIM = EMB_DIM

    def __init__(self, db_path: str, collection_name: str = "conversation_memory",
                 embedding_model: str = "all-MiniLM-L6-v2"):
        # `db_path` may be a directory (legacy chromadb usage) or a file path.
        # If a directory, we put memory_local.db inside it.
        if os.path.isdir(db_path) or db_path.endswith("/"):
            os.makedirs(db_path, exist_ok=True)
            self.db_path = os.path.join(db_path, "memory_local.db")
        else:
            os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
            self.db_path = db_path

        self._lock = threading.RLock()
        # Main 24 (post-Build 1): prefer the precompiled CoreML MiniLM
        # routed through ANE (CPU_AND_NE compute unit). Falls back to CPU
        # sentence-transformers if the artifact is missing or the env flag
        # `MIDAS_DISABLE_COREML_EMBED=1` is set. Measured 0.84 ms/embed on
        # ANE vs 2.68 ms on CPU; cosine 0.999985 match.
        try:
            from coreml_embedder import maybe_load_coreml_embedder
        except ImportError:
            from phantom_memory.coreml_embedder import maybe_load_coreml_embedder
        coreml = maybe_load_coreml_embedder()
        if coreml is not None:
            self.emb_model = coreml
            import sys as _sys
            print("[LocalMemoryStore] embedder: CoreML MiniLM (ANE)",
                  file=_sys.stderr)
        else:
            self.emb_model = SentenceTransformer(embedding_model, device="cpu")
            import sys as _sys
            print("[LocalMemoryStore] embedder: CPU SentenceTransformer (fallback)",
                  file=_sys.stderr)

        # Initialize schema
        c = _connect(self.db_path)
        try:
            c.executescript(SCHEMA)
            # Main 43: migrate existing databases — add topic column if missing
            cols = {row[1] for row in c.execute("PRAGMA table_info(memories)").fetchall()}
            if "topic" not in cols:
                c.execute("ALTER TABLE memories ADD COLUMN topic TEXT")
                c.execute("CREATE INDEX IF NOT EXISTS idx_topic ON memories(topic)")
        finally:
            c.close()

        # Load embedding matrix into memory
        self._ids: list[str] = []
        self._id_to_idx: dict[str, int] = {}
        self._emb_matrix = np.zeros((0, EMB_DIM), dtype=np.float32)
        self._load_index()

        # chromadb-compat shim
        self.collection = _CollectionShim(self)

    # ── index management ──
    def _load_index(self):
        with self._lock:
            c = _connect(self.db_path)
            try:
                rows = c.execute(
                    "SELECT id, embedding FROM memories WHERE superseded_by IS NULL"
                ).fetchall()
            finally:
                c.close()
            self._ids = [r["id"] for r in rows]
            self._id_to_idx = {fid: i for i, fid in enumerate(self._ids)}
            if rows:
                self._emb_matrix = np.stack([
                    np.frombuffer(r["embedding"], dtype=np.float32) for r in rows
                ])
            else:
                self._emb_matrix = np.zeros((0, EMB_DIM), dtype=np.float32)

    def _row_to_meta(self, row: sqlite3.Row) -> dict:
        meta: dict = {}
        keys = row.keys()
        for k in keys:
            if k in ("id", "text", "embedding", "extra_json"):
                continue
            v = row[k]
            if v is None:
                continue
            mk = COL_TO_META_KEY.get(k, k)
            meta[mk] = v
        if row["extra_json"]:
            try:
                meta.update(json.loads(row["extra_json"]))
            except Exception:
                pass
        # Normalize entities/quantities to JSON strings (the rest of the
        # codebase expects them as JSON strings, then json.loads()es them).
        for jf in ("entities", "quantities", "atom_entities", "atom_impacts"):
            if jf in meta and meta[jf] is None:
                meta[jf] = "[]"
        return meta

    # ── public API ──
    def count(self) -> int:
        return len(self._ids)

    # M54 Phase 4: vocabulary-gap query expansion. Known term mismatches
    # where the index uses one spelling and the user's vocabulary uses
    # another (Q03 enclave/exclave). When a trigger appears in the query,
    # we compute cosine against the original + expansions and take the
    # per-memory max. Keep this list tight — every entry is an extra
    # embed + matmul. Two-way so either side of the gap works.
    _QUERY_EXPANSIONS = {
        "enclave": ["exclave"],
        "exclave": ["enclave"],
    }

    @classmethod
    def _expand_query(cls, query: str) -> list[str]:
        low = query.lower()
        variants: list[str] = []
        for trigger, alts in cls._QUERY_EXPANSIONS.items():
            if trigger in low:
                for alt in alts:
                    variant = query.replace(trigger, alt).replace(
                        trigger.capitalize(), alt.capitalize()
                    )
                    if variant != query and variant not in variants:
                        variants.append(variant)
        return variants

    def recall(self, query: str, n_results: int = 5, type_filter: str = None,
               recency_weight: float = 0.25, include_superseded: bool = False,
               possessive_intent: bool = False) -> list[dict]:
        if not self._ids:
            return []
        # M54 Phase 4: compute cosine against original query + any
        # vocabulary-gap expansions, then take per-memory max.
        queries = [query] + self._expand_query(query)
        q_embs = self.emb_model.encode(
            queries, normalize_embeddings=True, show_progress_bar=False
        ).astype(np.float32)

        with self._lock:
            if q_embs.shape[0] == 1:
                sims = self._emb_matrix @ q_embs[0]  # (N,)
            else:
                # (N, D) @ (D, K) -> (N, K) -> max over K -> (N,)
                sims = (self._emb_matrix @ q_embs.T).max(axis=1)

        # Pull a wide candidate pool to allow type/superseded filtering + rerank
        fetch_n = min(max(n_results * 5, 30), len(self._ids))
        top = np.argpartition(-sims, fetch_n - 1)[:fetch_n]
        top = top[np.argsort(-sims[top])]

        ids = [self._ids[i] for i in top]
        with self._lock:
            c = _connect(self.db_path)
            try:
                placeholders = ",".join("?" * len(ids))
                row_map = {r["id"]: r for r in c.execute(
                    f"SELECT * FROM memories WHERE id IN ({placeholders})", ids
                ).fetchall()}
            finally:
                c.close()

        recalled = []
        now = time.time()
        for i in top:
            fid = self._ids[i]
            r = row_map.get(fid)
            if r is None:
                continue
            if not include_superseded and r["superseded_by"]:
                continue
            if type_filter and r["type"] != type_filter:
                continue
            similarity = float(sims[i])
            try:
                ft = datetime.fromisoformat(r["timestamp"]).timestamp()
                age_days = (now - ft) / 86400
                rec = 2 ** (-age_days / 7)
            except Exception:
                rec = 0.3
            score = similarity * (1 - recency_weight) + rec * recency_weight
            meta = self._row_to_meta(r)
            if meta.get("source_role") == "canonical":
                score *= 1.30
            # Provenance authority: model-generated content scores lower
            # than human-provided or vault-sourced content. Prevents
            # self-reinforcing hallucination loops where the system
            # retrieves its own prior fabrications as authoritative.
            sr = r["source_role"] or ""
            if sr == "assistant":
                score *= 0.50
            # M53 P4 / M54 Phase 2.2: possessive-intent filter.
            # When query asks "our X" / "we have Y", research-sourced
            # memories describing external projects score lower so they
            # don't dominate recall (Orion → "our LoRA pipeline").
            # M54: tightened from 0.30 to 0.05 because 109 Orion memories
            # at 0.30 still outranked the canonical denial. With a 20x
            # downweight, only the highest-cosine research memory has any
            # chance of surfacing in top-K when canonical denials exist.
            if possessive_intent and sr == "research":
                score *= 0.05
            # M54 Phase 4: when the query asks "what have we researched",
            # prior user questions on the same topic crowd out actual
            # answers (Q03 enclave — top 4 recalls were the user's own
            # past questions at sim 0.747, buried the research content).
            # Downweight user-role memories on possessive knowledge
            # queries; they are question-shaped, not answer-shaped.
            if possessive_intent and sr == "user":
                score *= 0.30
            recalled.append({
                "text": r["text"],
                "similarity": similarity,
                "recency": round(rec, 4),
                "score": score,
                "metadata": meta,
                "superseded": bool(r["superseded_by"]),
            })

        recalled.sort(key=lambda x: x["score"], reverse=True)
        return recalled[:n_results]

    # M54 Phase 4: ingest-time noise filter. The 8B extractor sometimes
    # returns "no facts to extract from this text" or similar meta
    # commentary instead of actual facts. These got stored as memories
    # with high recall similarity to question-shaped queries (Q18 bug —
    # 44 noise entries dominated recall on "what was the t14-t15 finding").
    _INGEST_NOISE_PATTERNS = (
        "no facts to extract",
        "no extracted facts",
        "appears to be a question",
        "appears to be a request",
        "starting point for research",
        "however, if we consider",
        "fact: [type] fact sentence",
    )

    @classmethod
    def _is_extraction_noise(cls, text: str) -> bool:
        low = text.lower().strip()
        return any(p in low for p in cls._INGEST_NOISE_PATTERNS)

    def store(self, fact: dict) -> Optional[str]:
        text = fact.get("text", "")
        if not text:
            return None
        # M54 Phase 4: drop extraction noise before embedding/storage
        if self._is_extraction_noise(text):
            return None
        emb = self.emb_model.encode(
            [text], normalize_embeddings=True, show_progress_bar=False
        )[0].astype(np.float32)
        if self._fast_dedup_check(emb):
            return None
        fid = f"fact_{len(self._ids) + 1}_{int(time.time())}"
        self._upsert_batch([fid], [emb.tolist()], [text],
                           [self._fact_to_metadata(fact)])
        return fid

    def store_batch(self, facts: list[dict]) -> list[str]:
        if not facts:
            return []
        # M54 Phase 4: drop extraction noise before any embedding work
        facts = [f for f in facts if not self._is_extraction_noise(f.get("text", ""))]
        if not facts:
            return []
        texts = [f["text"] for f in facts]
        embs = self.emb_model.encode(
            texts, normalize_embeddings=True,
            show_progress_bar=False, batch_size=32
        ).astype(np.float32)

        ids, kept_embs, kept_docs, kept_metas = [], [], [], []
        base = len(self._ids)
        for i, (fact, emb) in enumerate(zip(facts, embs)):
            if self._fast_dedup_check(emb):
                continue
            fid = f"fact_{base + i + 1}_{int(time.time())}"
            ids.append(fid)
            kept_embs.append(emb.tolist())
            kept_docs.append(fact["text"])
            kept_metas.append(self._fact_to_metadata(fact))
        if ids:
            self._upsert_batch(ids, kept_embs, kept_docs, kept_metas)
        return ids

    def get_by_type(self, fact_type: str, limit: int = 100, offset: int = 0) -> dict:
        return self.collection.get(
            where={"type": fact_type}, limit=limit, offset=offset,
            include=["documents", "metadatas", "embeddings"],
        )

    def get_all(self, limit: int = 500, offset: int = 0) -> dict:
        return self.collection.get(
            limit=limit, offset=offset,
            include=["documents", "metadatas"],
        )

    def get_recent_by_source(self, source_role: str, n: int = 20) -> list[dict]:
        """Return the N most-recent active memories with the given source_role.

        Used by multi_path_recall to guarantee that meta-memory bullets reach
        the fusion stage on activity-shaped queries even when raw cosine
        misses them. Returns the same dict shape as recall() so the caller
        can merge transparently.
        """
        with self._lock:
            c = _connect(self.db_path)
            try:
                rows = c.execute(
                    "SELECT * FROM memories "
                    "WHERE source_role = ? AND superseded_by IS NULL "
                    "ORDER BY timestamp DESC LIMIT ?",
                    (source_role, n)
                ).fetchall()
            finally:
                c.close()

        out = []
        now = time.time()
        for r in rows:
            try:
                ft = datetime.fromisoformat(r["timestamp"]).timestamp()
                age_days = (now - ft) / 86400
                rec = 2 ** (-age_days / 7)
            except Exception:
                rec = 0.3
            out.append({
                "text": r["text"],
                "similarity": 0.0,           # no query yet — multi_path will fill
                "recency": round(rec, 4),
                "score": rec * 0.5,
                "metadata": self._row_to_meta(r),
                "superseded": False,
            })
        return out

    # ── topic classifier (Main 43 Phase 1) ──
    _topic_patterns = None

    @classmethod
    def _classify_topic(cls, text: str) -> str | None:
        """Classify text into a topic using context_tracker keyword clusters.
        Returns the best-matching topic or None."""
        if cls._topic_patterns is None:
            import re as _re
            try:
                from context_tracker import DEFAULT_TOPIC_KEYWORDS
            except ImportError:
                import sys as _sys
                _sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                                  "..", "agent"))
                from context_tracker import DEFAULT_TOPIC_KEYWORDS
            cls._topic_patterns = {
                t: _re.compile(r"\b(" + "|".join(_re.escape(k) for k in kws) + r")\b",
                               _re.IGNORECASE)
                for t, kws in DEFAULT_TOPIC_KEYWORDS.items()
            }
        scores = {}
        for topic, patt in cls._topic_patterns.items():
            hits = len(patt.findall(text))
            if hits:
                scores[topic] = hits
        return max(scores, key=scores.get) if scores else None

    # ── internal write paths ──
    def _fact_to_metadata(self, fact: dict) -> dict:
        meta = {
            "type": fact.get("type", "general"),
            "source_role": fact.get("source_role", "unknown"),
            "timestamp": fact.get("timestamp", datetime.now().isoformat()),
            "entities": json.dumps(fact.get("entities", [])),
            "quantities": json.dumps(fact.get("quantities", [])),
            "session": fact.get("session", "unknown"),
        }
        # Main 43 Phase 1: auto-classify topic if not provided
        if "topic" not in fact:
            text = fact.get("text", "")
            topic = self._classify_topic(text)
            if topic:
                fact["topic"] = topic
        for k in ("atom_type", "atom_tense", "atom_core", "atom_confidence",
                  "atom_schema_version", "atom_migrated_at", "atom_replaces",
                  "file", "source", "topic"):
            if k in fact:
                meta[k] = fact[k]
        if "atom_entities" in fact:
            meta["atom_entities"] = (
                fact["atom_entities"] if isinstance(fact["atom_entities"], str)
                else json.dumps(fact["atom_entities"])
            )
        if "atom_impacts" in fact:
            meta["atom_impacts"] = (
                fact["atom_impacts"] if isinstance(fact["atom_impacts"], str)
                else json.dumps(fact["atom_impacts"])
            )
        return meta

    def _split_meta(self, meta: dict) -> tuple[dict, dict]:
        """Split meta dict into known SQL columns + extras for extra_json."""
        cols: dict = {}
        extras: dict = {}
        for k, v in meta.items():
            if k == "entities":
                cols["entities_json"] = v if isinstance(v, str) else json.dumps(v)
            elif k == "quantities":
                cols["quantities_json"] = v if isinstance(v, str) else json.dumps(v)
            elif k == "atom_entities":
                cols["atom_entities_json"] = v if isinstance(v, str) else json.dumps(v)
            elif k == "atom_impacts":
                cols["atom_impacts_json"] = v if isinstance(v, str) else json.dumps(v)
            elif k in KNOWN_META_FIELDS:
                cols[k] = v
            else:
                extras[k] = v
        return cols, extras

    def _upsert_batch(self, ids, embeddings, documents, metadatas):
        with self._lock:
            c = _connect(self.db_path)
            try:
                c.execute("BEGIN")
                for fid, emb, doc, meta in zip(ids, embeddings, documents, metadatas):
                    cols, extras = self._split_meta(meta or {})
                    emb_arr = np.asarray(emb, dtype=np.float32)
                    if emb_arr.shape != (EMB_DIM,):
                        raise ValueError(f"embedding shape {emb_arr.shape} != ({EMB_DIM},)")
                    blob = emb_arr.tobytes()
                    fields = {
                        "id": fid,
                        "text": doc,
                        "embedding": blob,
                        "extra_json": json.dumps(extras) if extras else None,
                        **cols,
                    }
                    keys = list(fields.keys())
                    vals = [fields[k] for k in keys]
                    placeholders = ",".join("?" * len(keys))
                    col_list = ",".join(keys)
                    update_set = ",".join(f"{k}=excluded.{k}" for k in keys if k != "id")
                    c.execute(
                        f"INSERT INTO memories ({col_list}) VALUES ({placeholders}) "
                        f"ON CONFLICT(id) DO UPDATE SET {update_set}",
                        vals
                    )
                c.execute("COMMIT")
            finally:
                c.close()
            # Refresh in-memory index for new ids (skip ones already present)
            # Cheapest: just append new rows; reload if any existed already.
            need_reload = any(fid in self._id_to_idx for fid in ids)
            if need_reload:
                self._load_index()
            else:
                new_embs = []
                for fid, emb in zip(ids, embeddings):
                    self._id_to_idx[fid] = len(self._ids)
                    self._ids.append(fid)
                    new_embs.append(np.asarray(emb, dtype=np.float32))
                self._emb_matrix = np.vstack([self._emb_matrix, np.stack(new_embs)])

    def _update_batch(self, ids, metadatas=None, documents=None, embeddings=None):
        with self._lock:
            c = _connect(self.db_path)
            try:
                c.execute("BEGIN")
                for i, fid in enumerate(ids):
                    if metadatas:
                        cols, extras = self._split_meta(metadatas[i] or {})
                        sets = []
                        vals = []
                        for k, v in cols.items():
                            sets.append(f"{k}=?")
                            vals.append(v)
                        sets.append("extra_json=?")
                        vals.append(json.dumps(extras) if extras else None)
                        vals.append(fid)
                        c.execute(f"UPDATE memories SET {','.join(sets)} WHERE id=?", vals)
                    if documents:
                        c.execute("UPDATE memories SET text=? WHERE id=?",
                                  (documents[i], fid))
                    if embeddings:
                        emb_arr = np.asarray(embeddings[i], dtype=np.float32)
                        c.execute("UPDATE memories SET embedding=? WHERE id=?",
                                  (emb_arr.tobytes(), fid))
                c.execute("COMMIT")
            finally:
                c.close()
            # If any update marked rows superseded, reload index to drop them.
            if metadatas and any(m.get("superseded_by") for m in metadatas):
                self._load_index()
            elif embeddings:
                self._load_index()

    def _delete(self, ids=None, where=None):
        with self._lock:
            c = _connect(self.db_path)
            try:
                if ids:
                    placeholders = ",".join("?" * len(ids))
                    c.execute(f"DELETE FROM memories WHERE id IN ({placeholders})", ids)
                elif where:
                    clauses = []
                    vals = []
                    for k, v in where.items():
                        clauses.append(f"{k}=?")
                        vals.append(v)
                    c.execute(f"DELETE FROM memories WHERE {' AND '.join(clauses)}", vals)
            finally:
                c.close()
            self._load_index()

    def _fast_dedup_check(self, embedding: np.ndarray) -> bool:
        if len(self._ids) == 0:
            return False
        sims = embedding @ self._emb_matrix.T
        return float(sims.max()) >= self.DEDUP_THRESHOLD

    # ─────────────────────────────────────────────────────────────────
    # Main 61: reactive framing-stale triggers
    #
    # flag_framing_stale: event-driven supersession for assistant/user
    # echoes whose numeric framing no longer matches the registry /
    # canonical truth. NEVER mutates canonical rows — those get logged
    # to maintenance_log.jsonl with action=flag_for_review for human
    # review (per vault/CLAUDE.md "DO NOT auto-supersede canonical").
    #
    # update_canonical_in_place: the narrow escape hatch for obvious
    # two-number rewrites derivable from the registry. Preserves id,
    # changes text + timestamp, logs to maintenance_log.jsonl.
    # ─────────────────────────────────────────────────────────────────
    _MAINTENANCE_LOG_PATH = "/Users/midas/Desktop/cowork/data/maintenance_log.jsonl"

    def _append_maintenance_log(self, entry: dict) -> None:
        try:
            os.makedirs(os.path.dirname(self._MAINTENANCE_LOG_PATH), exist_ok=True)
            entry.setdefault("ts", datetime.utcnow().isoformat())
            with open(self._MAINTENANCE_LOG_PATH, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError:
            pass  # non-critical, never break the hot path on log write

    def flag_framing_stale(self, memory_id: str, reason: str,
                           triggered_by: str) -> None:
        """Event-driven flag for framing-stale memories.

        If the row is role=canonical: DO NOT mutate, only log
        action=flag_for_review. Otherwise mark superseded_by=triggered_by
        and superseded_at=utcnow(). Idempotent: a row already marked
        superseded is a no-op except for the log line.
        """
        with self._lock:
            c = _connect(self.db_path)
            try:
                row = c.execute(
                    "SELECT id, source_role, superseded_by FROM memories WHERE id=?",
                    (memory_id,),
                ).fetchone()
                if row is None:
                    self._append_maintenance_log({
                        "memory_id": memory_id, "action": "flag_missing",
                        "reason": reason, "triggered_by": triggered_by,
                        "role": None,
                    })
                    return
                role = row["source_role"] or ""
                already_flagged = bool(row["superseded_by"])
                if role == "canonical":
                    # Log-only path; canonical memories are human-review only.
                    self._append_maintenance_log({
                        "memory_id": memory_id, "action": "flag_for_review",
                        "reason": reason, "triggered_by": triggered_by,
                        "role": role,
                    })
                    return
                if already_flagged:
                    # Idempotent: re-log but don't mutate
                    self._append_maintenance_log({
                        "memory_id": memory_id, "action": "auto_supersede_dup",
                        "reason": reason, "triggered_by": triggered_by,
                        "role": role,
                    })
                    return
                ts = datetime.utcnow().isoformat()
                c.execute(
                    "UPDATE memories SET superseded_by=?, superseded_at=? "
                    "WHERE id=?",
                    (triggered_by, ts, memory_id),
                )
            finally:
                c.close()
            # Rebuild in-memory index so superseded row drops out of recall
            self._load_index()
        self._append_maintenance_log({
            "memory_id": memory_id, "action": "auto_supersede",
            "reason": reason, "triggered_by": triggered_by,
            "role": role,
        })

    def update_canonical_in_place(self, memory_id: str, new_text: str,
                                  reason: str) -> bool:
        """Narrow-safe two-number rewrite for canonical rows.

        Verifies source_role='canonical'. Updates text + timestamp only,
        preserves id and embedding (caller must re-embed separately if
        semantic drift is large — intended for numeric value swaps).
        Returns True on success. Idempotent: no-op if text already matches.
        """
        with self._lock:
            c = _connect(self.db_path)
            try:
                row = c.execute(
                    "SELECT id, source_role, text FROM memories WHERE id=?",
                    (memory_id,),
                ).fetchone()
                if row is None:
                    return False
                if (row["source_role"] or "") != "canonical":
                    return False
                old_text = row["text"] or ""
                if old_text == new_text:
                    return True  # idempotent no-op
                ts = datetime.utcnow().isoformat()
                c.execute(
                    "UPDATE memories SET text=?, timestamp=? WHERE id=?",
                    (new_text, ts, memory_id),
                )
            finally:
                c.close()
        self._append_maintenance_log({
            "memory_id": memory_id, "action": "canonical_in_place_update",
            "reason": reason,
            "old_text_prefix": old_text[:120],
            "new_text_prefix": new_text[:120],
            "triggered_by": "update_canonical_in_place",
            "role": "canonical",
        })
        return True
