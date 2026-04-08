#!/usr/bin/env python3
"""
Migrate the chromadb snapshot (taken by Main 24 Track 2 purge agent) into
the new LocalMemoryStore. Skips the 401 IDs the purge marked as deletes.
Applies the 124 redactions to text content. Re-embeds via MiniLM-L6-v2.

Inputs:
  /tmp/main24_track2_purge_snapshot.json   — full pre-purge snapshot
  /tmp/main24_track2_purge_log.json        — purge actions

Output:
  <db_dir>/memory_local.db                  — new SQLite store
"""
import json
import os
import sys
import time

import numpy as np
from sentence_transformers import SentenceTransformer

sys.path.insert(0, os.path.dirname(__file__))
from local_store import LocalMemoryStore, _connect, EMB_DIM, SCHEMA

SNAPSHOT = "/tmp/main24_track2_purge_snapshot.json"
PURGE_LOG = "/tmp/main24_track2_purge_log.json"
DB_DIR = "/Users/midas/Desktop/cowork/orion-ane/memory/chromadb_live"


def main():
    print(f"[migrate] loading snapshot from {SNAPSHOT}")
    with open(SNAPSHOT) as f:
        snapshot = json.load(f)
    print(f"[migrate]   {len(snapshot)} rows in snapshot")

    print(f"[migrate] loading purge log from {PURGE_LOG}")
    with open(PURGE_LOG) as f:
        log = json.load(f)
    delete_ids = set()
    redactions = {}
    for a in log["actions"]:
        if a["action"] == "delete":
            delete_ids.add(a["id"])
        elif a["action"] == "redact":
            redactions[a["id"]] = a["redacted_text"]
    print(f"[migrate]   {len(delete_ids)} deletes, {len(redactions)} redactions")

    # Filter and prepare rows
    surviving = []
    for row in snapshot:
        eid = row["embedding_id"]
        if eid in delete_ids:
            continue
        text = row["document"]
        if eid in redactions:
            text = redactions[eid]
        if not text or not text.strip():
            continue
        surviving.append({
            "id": eid,
            "text": text,
            "metadata": row.get("metadata") or {},
        })
    print(f"[migrate] {len(surviving)} surviving rows after filter")

    # Initialize the store (creates schema, no data yet)
    print(f"[migrate] initializing LocalMemoryStore at {DB_DIR}/memory_local.db")
    store = LocalMemoryStore(DB_DIR)
    # Wipe any existing data so this is idempotent
    c = _connect(store.db_path)
    try:
        c.execute("DELETE FROM memories")
    finally:
        c.close()
    store._load_index()
    print(f"[migrate]   store initialized, {store.count()} rows present")

    # Re-embed in batches of 64 — faster than per-row, model is CPU MiniLM
    BATCH = 64
    t0 = time.time()
    inserted = 0
    for batch_start in range(0, len(surviving), BATCH):
        batch = surviving[batch_start:batch_start + BATCH]
        texts = [r["text"] for r in batch]
        embs = store.emb_model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False, batch_size=BATCH
        ).astype(np.float32)

        ids = [r["id"] for r in batch]
        metas = [r["metadata"] for r in batch]

        # Use the upsert path directly so we keep the original IDs
        store._upsert_batch(ids, [e.tolist() for e in embs], texts, metas)
        inserted += len(batch)

        if inserted % 256 == 0 or inserted == len(surviving):
            elapsed = time.time() - t0
            rate = inserted / elapsed if elapsed > 0 else 0
            eta = (len(surviving) - inserted) / rate if rate > 0 else 0
            print(f"[migrate]   {inserted}/{len(surviving)} "
                  f"({rate:.1f} rows/s, eta {eta:.0f}s)")

    elapsed = time.time() - t0
    print(f"[migrate] DONE: {inserted} rows in {elapsed:.1f}s "
          f"({inserted/elapsed:.1f} rows/s)")
    print(f"[migrate] final store.count() = {store.count()}")
    print(f"[migrate] db file: {store.db_path}")
    print(f"[migrate] db size: "
          f"{os.path.getsize(store.db_path)/1e6:.1f} MB")

    # Sanity recall
    print("[migrate] sanity recall: 'what is the 8B tok/s on ANE'")
    results = store.recall("what is the 8B tok/s on ANE", n_results=3)
    for i, r in enumerate(results, 1):
        print(f"  {i}. score={r['score']:.3f} sim={r['similarity']:.3f}  "
              f"{r['text'][:120]}")


if __name__ == "__main__":
    main()
