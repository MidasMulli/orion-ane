"""
In-memory embedding index for sub-millisecond retrieval.

Loads all embeddings from ChromaDB into a numpy array on startup.
Retrieval is a single BLAS call (AMX-accelerated cblas_sgemv).
ChromaDB remains the write path and metadata store.

Architecture:
  warm set: ALL embeddings in DRAM numpy array (3,500 @ 384-dim = 5.1MB)
  hot set:  top-100 most relevant for current conversation (~150KB, L2-resident)
  ChromaDB: persistence + metadata only (writes go to both)
"""

import os
import time
import json
import logging
import numpy as np

import chromadb
from sentence_transformers import SentenceTransformer

log = logging.getLogger("memory_cache")

DB_PATH = os.path.expanduser("~/Desktop/cowork/orion-ane/memory/chromadb_live")
COLLECTION = "conversation_memory"
MAX_MEMORIES = 50_000
EMB_DIM = 384
HOT_SIZE = 100


class MemoryCache:
    """In-memory embedding cache with hot set for sub-ms retrieval."""

    def __init__(self, db_path=DB_PATH, collection_name=COLLECTION):
        self.db_path = db_path
        self.collection_name = collection_name

        # Warm set: all embeddings in contiguous numpy array
        self.embeddings = np.zeros((MAX_MEMORIES, EMB_DIM), dtype=np.float32)
        self.contents = []
        self.metadata = []
        self.ids = []
        self.count = 0

        # Hot set: indices into warm set for current conversation
        self.hot_indices = np.zeros(HOT_SIZE, dtype=np.int32)
        self.hot_count = 0

        # Embedding model (shared with daemon)
        self.emb_model = None

        # ChromaDB connection (for writes + metadata)
        self.client = None
        self.collection = None

    def load(self):
        """Load all embeddings from ChromaDB into DRAM."""
        t0 = time.time()

        self.client = chromadb.PersistentClient(path=self.db_path)
        self.collection = self.client.get_collection(self.collection_name)

        # Load embedding model
        self.emb_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

        # Fetch all data
        total = self.collection.count()
        batch_size = 5000
        offset = 0

        while offset < total:
            batch = self.collection.get(
                include=["embeddings", "documents", "metadatas"],
                limit=batch_size, offset=offset)

            for i in range(len(batch["ids"])):
                if self.count >= MAX_MEMORIES:
                    break

                meta = batch["metadatas"][i]
                # Skip superseded
                if meta.get("superseded_by"):
                    continue

                emb = batch["embeddings"][i]
                self.embeddings[self.count] = np.array(emb, dtype=np.float32)
                self.contents.append(batch["documents"][i])
                self.metadata.append(meta)
                self.ids.append(batch["ids"][i])
                self.count += 1

            offset += batch_size

        elapsed = time.time() - t0
        size_mb = self.count * EMB_DIM * 4 / 1e6
        log.info(f"MemoryCache loaded: {self.count} memories, {size_mb:.1f}MB, {elapsed:.2f}s")

    def embed(self, text):
        """Embed a text string. Returns [EMB_DIM] float32."""
        return self.emb_model.encode(
            [text], normalize_embeddings=True, show_progress_bar=False
        )[0].astype(np.float32)

    def retrieve(self, query_text, k=20, threshold=0.35):
        """Retrieve top-k memories by cosine similarity.

        Uses hot set first (L2-speed), falls back to warm set (DRAM-speed).
        Filters by threshold. Returns list of dicts.
        """
        query_emb = self.embed(query_text)

        # Try hot set first
        if self.hot_count > 0:
            hot_emb = self.embeddings[self.hot_indices[:self.hot_count]]
            hot_scores = query_emb @ hot_emb.T
            if hot_scores.max() > 0.5:
                top_k = hot_scores.argsort()[-k:][::-1]
                results = []
                for idx in top_k:
                    score = float(hot_scores[idx])
                    if score < threshold:
                        continue
                    real_idx = self.hot_indices[idx]
                    results.append({
                        "text": self.contents[real_idx],
                        "score": score,
                        "type": self.metadata[real_idx].get("type", "unknown"),
                        "entities": json.loads(self.metadata[real_idx].get("entities", "[]")),
                        "timestamp": self.metadata[real_idx].get("timestamp", ""),
                    })
                if len(results) >= 3:
                    return results

        # Full warm set search
        scores = query_emb @ self.embeddings[:self.count].T
        top_k_indices = scores.argsort()[-k:][::-1]

        results = []
        for idx in top_k_indices:
            score = float(scores[idx])
            if score < threshold:
                continue
            # Filter vault file content and questions
            text = self.contents[idx]
            if text.startswith("[") and ".md]" in text[:50]:
                continue
            if text.strip().endswith("?"):
                continue
            results.append({
                "text": text,
                "score": score,
                "type": self.metadata[idx].get("type", "unknown"),
                "entities": json.loads(self.metadata[idx].get("entities", "[]")),
                "timestamp": self.metadata[idx].get("timestamp", ""),
            })

        return results

    def retrieve_raw(self, query_emb, k=20):
        """Retrieve using pre-computed embedding. No filtering. Fastest path."""
        scores = query_emb @ self.embeddings[:self.count].T
        top_k = scores.argsort()[-k:][::-1]
        return [(int(i), float(scores[i])) for i in top_k]

    def warm_hot_set(self, domain_keywords=None, recent_queries=None):
        """Build hot set from domain keywords or recent query history.

        Hot set = top-100 memories most relevant to current conversation.
        ~150KB, designed to stay in L2 cache.
        """
        if recent_queries:
            # Average the query embeddings to get conversation centroid
            query_embs = [self.embed(q) for q in recent_queries[-5:]]
            centroid = np.mean(query_embs, axis=0)
        elif domain_keywords:
            centroid = self.embed(" ".join(domain_keywords))
        else:
            # Default: most recent memories
            self.hot_indices[:min(HOT_SIZE, self.count)] = np.arange(
                max(0, self.count - HOT_SIZE), self.count)
            self.hot_count = min(HOT_SIZE, self.count)
            return

        scores = centroid @ self.embeddings[:self.count].T
        top_indices = scores.argsort()[-HOT_SIZE:][::-1]
        n = min(HOT_SIZE, len(top_indices))
        self.hot_indices[:n] = top_indices[:n]
        self.hot_count = n

    def add_memory(self, text, metadata, embedding=None):
        """Add a new memory to both cache and ChromaDB."""
        if embedding is None:
            embedding = self.embed(text)

        # Add to warm set
        if self.count < MAX_MEMORIES:
            self.embeddings[self.count] = embedding
            self.contents.append(text)
            self.metadata.append(metadata)
            fact_id = f"fact_{self.count}_{int(time.time())}"
            self.ids.append(fact_id)
            self.count += 1

        # Persist to ChromaDB
        if self.collection:
            self.collection.upsert(
                ids=[fact_id],
                embeddings=[embedding.tolist()],
                documents=[text],
                metadatas=[metadata],
            )

        return fact_id
