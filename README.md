# orion-ane

Cognitive architecture for local LLM agents on Apple Silicon. Five model tiers running concurrently on a single MacBook Pro M5 Pro, coordinated by a self-correcting memory system that the agent never sees but the user benefits from on every turn.

Zero cloud. Sub-millisecond memory recall. The agent that uses this stack is **Midas** — a research partner that remembers what was shipped, what was killed, and what's still open across sessions.

---

## Stack

| Tier | Model | Hardware | Role | Speed |
|---|---|---|---|---|
| **Verifier** | Gemma 4 31B Q4 | GPU (MLX-Metal, 20 cores) | conversation + reasoning | 17.5 tok/s pure AR |
| **Extractor** | Llama 3.1-8B-Instruct Q8 | ANE (CoreML, 72 dispatches) | typed fact extraction | 7.9 tok/s |
| **Embedder** | MiniLM-L6-v2 | ANE (CoreML, CPU_AND_NE) | retrieval embedding | 0.84 ms/embed |
| **Classifier** | Neuron 80M (FFN-only) | ANE SRAM | domain routing | 905 μs |
| **Drafter** | N-gram (truncate-on-miss) | software | speculative decode | disabled (phrase loops) |

The verifier's prompt cache integration (Main 25) snapshots the system + briefing KV state on first turn and reuses it on every subsequent turn. **~30 seconds of TTFT eliminated per turn** at production briefing scale.

The cognitive architecture is model-agnostic: four verifier swaps (Llama 70B → Qwen 72B → Qwen 27B → Gemma 4 31B) required zero pipeline changes. Retrieval, extraction, maintenance, and scrub all continued to operate unchanged across swaps.

---

## Subconscious — the cognitive memory layer

The agent's primary product. The user never sees it; they just notice the agent remembers what they shipped.

- **~7,400 memories** in `LocalMemoryStore` (SQLite WAL + in-memory float32 numpy matrix). Sub-ms cosine via single matmul. Replaced ChromaDB in Main 24.
- **Multi-path 5-signal retrieval** — `embedding (0.35) + entity (0.25) + type (0.15) + impact (0.15) + recency (0.10)` with `1.30x` canonical-state boost and activity-query category override.
- **Structured atom storage** — every memory has `atom_type`, `atom_entities`, `atom_impacts`, `atom_tense`, `atom_confidence`, `atom_core`, `source_role`. Retrieval uses all of them.
- **10 maintenance feedback loops** — 9 hourly loops running via launchd plus reactive triggers shipped Main 61:
  1. `decay_scores` — exponential decay on relevance
  2. `consolidate_duplicates` — merge near-duplicate facts
  3. `resolve_contradictions` — verifier contradiction resolver during idle time
  4. `vault_sync` — supersede memories that conflict with the canonical knowledge files
  5. `production_state_sync` — sync the live infrastructure state into memory
  6. `semantic_supersession` — 3-signal paraphrase-aware supersession (cosine + tense + restate-vs-contradict)
  7. `canonical_state_inject` — parse `CLAUDE.md` tables into first-class canonical memories
  8. `meta_memory_inject` — parse session-log bullets into first-class activity memories ("what did we ship today")
  9. `vault_sweep` — surface deliverables on disk that no knowledge file references (closes the "completed work, never wired" recurring failure mode)
  10. **Reactive maintenance** — event-driven framing-stale detection (60.9 ms from data change to flag), registry write propagation, and immediate vault-sync on canonical state changes. Closes the correction window from hours to milliseconds.
- **100% cross-session continuity** measured across 5 conversation sessions: 6/6 references resolved, 24/24 turns coherent, zero hallucinated events.
- See [`vault/subconscious`](https://github.com/MidasMulli/subconscious) for the loop implementations.

---

## Midas — the agent that uses the stack

`agent/midas_ui.py` runs a Flask web UI on `:8450` (network-accessible from any device on the LAN) and exposes the agent over a simple `/api/chat` endpoint.

- **Deterministic L1+L2 router** (`agent/router.py`) — keyword patterns for 90% of routing decisions, falling back to a single-word LLM classifier only when no L1 pattern matches. Tool args constructed in code, not by the LLM.
- **Cognitive context assembly** (`agent/briefing_assembler.py`, `agent/synthesizer.py`) — stable per-session briefing built from the canonical memory state, refreshed every 5 turns. Per-query memories ride in the user message tail to avoid invalidating the verifier's prefix cache.
- **Streaming SSE** with sliding-window incremental decode (no O(n²) tokenizer.decode bug).
- **Continuous embedding PoC** (`agent/continuous_embed.py`) — streams a verifier response, embeds the partial output every N tokens via the CoreML MiniLM, retrieves top-K, logs mid-generation discoveries. **0.5% wall overhead at N=4** over a 164-token generation.
- **Self-observation tools** — `self_test`, `brain_snapshot`, `self_improve` are first-class tools the agent can call on itself.

---

## Hardware

MacBook Pro M5 Pro · 64 GB unified memory · 18 CPU cores (6P + 12S) · 20 GPU cores · Metal 4 · 307 GB/s DRAM · ANE dedicated 111 GB/s DMA channel

Cross-accelerator contention is model-dependent. ANE-side contention is model-invariant (+1.4% under Llama 70B, +0.38% under Gemma 4 31B — both inside the noise floor). GPU-side contention is verifier-scale-dependent (−4.7% at Llama 70B, −20.1% at Gemma 4 31B) because a faster verifier spends a larger fraction of wall clock in steady-state decode, where the symmetric 8B ANE extraction load is visible.

---

## Project status

The architecture and measurements behind this stack are documented across two papers (see below). The cognitive memory system is the primary product. The agent is the proof-of-concept. The hardware probes that landed in `nax-probe/` are the explanation for *why* this stack outperforms naive single-model deployment on the same machine.

---

## Papers

- **Paper 1:** "Every Cycle Counts: A Self-Correcting Cognitive Architecture on Heterogeneous Consumer Silicon" — hardware substrate and cognitive pipeline. [arXiv link TBD]
- **Paper 2:** "Five Roadblocks to Persistent Memory for Personal AI" — memory architecture taxonomy and measured outcomes. [arXiv link TBD]

---

## Related repos

- [subconscious](https://github.com/MidasMulli/subconscious) — the cognitive memory loops, separate package
- [ane-compiler](https://github.com/MidasMulli/ane-compiler) — model builder + fusion optimizer + dispatch orchestrator (GPT-2 229 tok/s, Llama-1B 50.2 tok/s, Llama-8B Q8 7.9 tok/s, Neuron 1,064 tok/s)
- [ane-dispatch](https://github.com/MidasMulli/ane-dispatch) — direct ANE dispatch + SharedEvents (37% faster than CoreML)
- [ane-toolkit](https://github.com/MidasMulli/ane-toolkit) — IOKit protocol decoder + Mach-O .hwx tooling
- [four-path-mlx](https://github.com/MidasMulli/four-path-mlx) — multi-source speculative decoding server — the inference-level results that the NAX hardware data explains
- [gdn-coreml](https://github.com/MidasMulli/gdn-coreml) — GatedDeltaNet SSM to CoreML converter for same-family ANE drafting
- [ane-perf](https://github.com/MidasMulli/ane-perf) — ANE hardware performance characterization via IOReport bandwidth histograms

## License

MIT.
