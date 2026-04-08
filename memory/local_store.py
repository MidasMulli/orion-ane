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
    extra_json          TEXT
);
CREATE INDEX IF NOT EXISTS idx_type        ON memories(type);
CREATE INDEX IF NOT EXISTS idx_atom_type   ON memories(atom_type);
CREATE INDEX IF NOT EXISTS idx_source_role ON memories(source_role);
CREATE INDEX IF NOT EXISTS idx_superseded  ON memories(superseded_by);
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
    "file", "source", "relevance_score",
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

    def recall(self, query: str, n_results: int = 5, type_filter: str = None,
               recency_weight: float = 0.25, include_superseded: bool = False) -> list[dict]:
        if not self._ids:
            return []
        q_emb = self.emb_model.encode(
            [query], normalize_embeddings=True, show_progress_bar=False
        )[0].astype(np.float32)

        with self._lock:
            sims = self._emb_matrix @ q_emb  # (N,)

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

    def store(self, fact: dict) -> Optional[str]:
        text = fact.get("text", "")
        if not text:
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
        for k in ("atom_type", "atom_tense", "atom_core", "atom_confidence",
                  "atom_schema_version", "atom_migrated_at", "atom_replaces",
                  "file", "source"):
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
