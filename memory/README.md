# Phantom Memory

Zero-cost persistent memory for local LLMs — extraction, embedding, and a continuous enrichment loop that organizes your knowledge while you sleep.

Mem0 stores facts. **Phantom thinks about them.**

## The Pitch

Every local LLM forgets. Phantom remembers. But it doesn't just store facts — it thinks about them. A continuous enrichment loop reclassifies, builds relationships, detects staleness, finds cross-entity patterns, and consolidates profiles. Your vault gets smarter while you sleep.

Three processors. Three loops. Zero contention.

```
┌─ GPU ────────────────────────────────────┐
│  Your LLM — conversation + reasoning     │
└──────────────────────────────────────────┘
┌─ CPU ────────────────────────────────────┐
│  Memory Daemon — extract, embed, store   │
│  1,721 emb/sec, zero GPU impact          │
└──────────────────────────────────────────┘
┌─ ANE (optional) ────────────────────────┐
│  Enricher — classify, relate, analyze    │
│  Always-on, ~2W, zero GPU impact         │
└──────────────────────────────────────────┘
```

## Dashboard

![Phantom Memory Dashboard](dashboard_screenshot.png)

Real-time visualization of the knowledge graph, activity feed, memory stats, fact type distribution, and recent decisions. Dark cyberpunk UI at `http://localhost:8422`.

## Quick Start

```python
from daemon import MemoryDaemon

# Basic — memory only
daemon = MemoryDaemon(vault_path="~/vault")
daemon.start()

# Full stack — memory + enricher (recommended)
daemon = MemoryDaemon(vault_path="~/vault", enable_enricher=True)
daemon.start()

# Custom enricher interval (default 300s)
daemon = MemoryDaemon(
    vault_path="~/vault",
    enable_enricher=True,
    enricher_interval=60,  # seconds
)
daemon.start()

# Ingest conversation turns
daemon.ingest("user", "We agreed to set the cross-default threshold at $75M for Counterparty Alpha")
daemon.ingest("assistant", "Noted. That's higher than the $50M standard for BBB+ entities.")

# Semantic recall
results = daemon.recall("cross-default thresholds")

# One-shot enrichment (run all sweeps once, then exit)
daemon.enrich_once()
```

## Continuous Enrichment

This is the headline feature. Five sweeps run continuously in the background, analyzing and improving your vault:

| Sweep | What it does | Status |
|-------|-------------|--------|
| **RECLASSIFY** | Fixes facts filed as "general" that should be decision/task/preference/quantitative | Working |
| **RELATE** | Builds entity relationship graph + `graph.json` | Working — 40+ relationships found |
| **STALE** | Flags outdated time-sensitive facts (deadlines, "by Friday", temporal refs) | Working |
| **PATTERNS** | Cross-entity analysis — outliers, similar profiles, recurring provisions | Working — 35 insights generated |
| **CONSOLIDATE** | Auto-generates entity profile summaries pinned to top of entity pages | Working |

### What Gets Created in the Vault

```
vault/memory/
├── relationships.md          # Entity graph with wikilinks
├── graph.json                # Machine-readable graph (for dashboard/agent consumption)
├── insights/
│   ├── patterns-2026-03-18.md  # Cross-entity analysis
│   └── stale-2026-03-18.md     # Staleness flags
├── entities/
│   ├── counterparty-alpha.md   # Auto-generated profile at top
│   ├── jpmorgan.md
│   └── ...
├── facts/
├── decisions/
├── preferences/
└── tasks/
```

### Sample Insight Note

What Phantom produces overnight while your LLM is idle:

```markdown
# ◈ Phantom Insights — 2026-03-18

## Threshold Inconsistencies
- Counterparty Alpha: $75M cross-default (BBB+)
- Counterparty Beta: $50M cross-default (BBB+)
- Same rating, 50% threshold difference. Intentional?

## Stale Items
- "Draft counter-proposal by Friday" (from March 14) — likely past

## Entity Summaries Updated
- Counterparty Alpha: 47 facts → 5-line profile (updated overnight)
```

### ANE-Ready Architecture

`Classifier` and `Embedder` are protocols — swap in a CoreML model with zero changes to sweep logic:

```python
class Classifier(Protocol):
    def classify(self, text: str) -> str: ...

class Embedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray: ...
```

Currently runs on CPU with heuristics. On Apple Silicon with a CoreML model loaded, classification runs on the Neural Engine at ~2W with zero GPU impact.

## How Memory Works

### Extraction

Every conversation turn passes through `FactExtractor`:
- Regex-based entity detection (organizations, people, amounts, dates)
- Fact type classification (decision, task, preference, quantitative, general)
- Quantity parsing ($75M, 50 basis points, etc.)
- Deduplication against existing memories via cosine similarity

### Embedding & Storage

- **sentence-transformers** (`all-MiniLM-L6-v2`, 22M params) on CPU
- 1,721 embeddings/sec — fast enough that ANE isn't needed for this tier
- **ChromaDB** for persistent vector storage with cosine similarity search
- Semantic recall: query in natural language, get relevant facts ranked by relevance

### Vault Writing

Extracted facts are written as structured markdown to an Obsidian-compatible vault:
- One file per entity, fact, decision, task
- Wikilinks between related entities
- Machine-readable frontmatter (type, date, entities, confidence)

## Benchmarks

### Three-Tier Eval Suite (22/22)

```
CPU (extraction + recall) ....... 12/12 (100%)
ANE (1.7B analysis) ............  5/5  (100%)
GPU (9B reasoning) .............  5/5  (100%)
TOTAL .......................... 22/22 (100%)
Completed in 45.1s
```

**CPU tier tests:** Fact extraction (all 4 types from ISDA paragraph), entity detection (10 entities including Counterparty Alpha), semantic recall accuracy, type filtering, noise rejection.

**ANE tier tests:** Entity summarization (JPMorgan profile), relationship extraction (4 entities), risk identification (low cross-default threshold), quantity extraction (4 dollar amounts from CSA text). All under 2 seconds.

**GPU tier tests:** Domain explanation (5/6 terms), risk comparison, JSON generation, tool call formatting.

Run it yourself:
```bash
~/.mlx-env/bin/python3 eval_tiers.py
```

### Concurrent Performance

GPU inference runs at full speed while the memory daemon processes in background:

| Scenario | GPU tok/s | Impact |
|----------|-----------|--------|
| GPU alone | 24.4-25.0 | baseline |
| GPU + daemon (20 embeddings) | 25.4 | -4.2% (noise) |
| GPU + daemon (100 embeddings) | 25.9 | -3.6% (noise) |

Near-zero contention measured (~3.8% average). 18 facts extracted and stored while the GPU generated at full speed.

### Embedding Throughput

| Model | Params | Speed | Hardware |
|-------|--------|-------|----------|
| all-MiniLM-L6-v2 | 22M | 1,721 emb/sec | CPU |

## Agent Self-Knowledge (Playbook)

When integrated with the Phantom agent framework, the agent maintains a self-assessment document (`playbook.md`) it reads at boot and updates after every task:

- **Scan schedule** with timestamps (last run, next due)
- **What works / what doesn't** — learned from experience, not hardcoded
- **High-signal sources** — builds over time as the agent discovers useful feeds
- **Self-eval metrics** — tracks accuracy, tool chain success rates, noise ratio
- **Improvement queue** — next things to try
- **Lessons learned** — persistent behavioral memory

The playbook lives in the vault (visible in Obsidian) and survives restarts. The agent reads it at boot, follows it during tasks, and writes back what it learned. This is how the agent gets better across sessions.

## API Reference

### MemoryDaemon

```python
daemon = MemoryDaemon(
    vault_path: str,              # Path to Obsidian vault
    db_path: str = None,          # ChromaDB path (default: ./chromadb_live)
    enable_enricher: bool = False, # Start continuous enrichment loop
    enricher_interval: int = 300,  # Seconds between enrichment sweeps
)

daemon.start()                    # Boot daemon + enricher (if enabled)
daemon.stop()                     # Shutdown gracefully

daemon.ingest(role, text)         # Extract facts from text, embed, store, write to vault
daemon.recall(query, n=5, type_filter="")  # Semantic search over memories
daemon.stats()                    # {extracted, stored, deduped, superseded, total_memories}
daemon.enrich_once()              # Run all 5 enrichment sweeps once
```

### PhantomEnricher

```python
from enricher import PhantomEnricher

enricher = PhantomEnricher(
    store=memory_store,           # MemoryStore instance (ChromaDB)
    vault=vault_writer,           # VaultWriter instance
    interval=300,                 # Seconds between sweeps
)

enricher.start()                  # Background thread, runs forever
enricher.stop()
enricher.run_once()               # One-shot: all 5 sweeps, then return
```

### SweepEngine

```python
from enricher import SweepEngine

engine = SweepEngine(store=memory_store, vault=vault_writer)
engine.sweep_reclassify()         # Fix mistyped facts
engine.sweep_relate()             # Build relationship graph
engine.sweep_stale()              # Flag outdated facts
engine.sweep_patterns()           # Cross-entity anomaly detection
engine.sweep_consolidate()        # Generate entity profiles
```

## Apple Silicon Bonus

On Macs with Apple Silicon, the three tiers map to physically separate processors:

- **GPU** (Metal) — your main LLM, fully utilized during conversation
- **CPU** (Efficiency cores) — embedding + extraction, negligible power draw
- **ANE** (Neural Engine) — enricher classification, ~2W, completely independent bus

This isn't software concurrency — it's hardware parallelism. The ANE has its own data path to unified memory. The enricher literally cannot slow down your LLM.

## Requirements

- Python 3.11+
- macOS (tested on M4/M5 Apple Silicon, works on Intel with CPU-only)
- ChromaDB, sentence-transformers, numpy

```bash
pip install chromadb sentence-transformers numpy
```

Optional for ANE tier:
- CoreML model converted via [ANEMLL](https://github.com/Anemll/Anemll)
- `ane_server.py` running with persistent CoreML model

## License

MIT

---

*Built by a human + Claude, one weekend at a time.*
