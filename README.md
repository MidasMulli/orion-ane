# orion-ane

Cognitive architecture for local LLM agents on Apple Silicon. Five model tiers running concurrently on a single MacBook Pro M5 Pro, coordinated by a self-correcting memory system that the agent never sees but the user benefits from on every turn.

Zero cloud. Zero contention. Sub-millisecond memory recall. The agent that uses this stack is **Midas** — a research partner that remembers what was shipped, what was killed, and what's still open across sessions.

---

## Stack

| Tier | Model | Hardware | Role | Speed |
|---|---|---|---|---|
| **Verifier** | Qwen 2.5-72B-Instruct-4bit | GPU (MLX-Metal) | conversation + reasoning | 6.5 cold / 8-36 warm tok/s |
| **Extractor** | Llama 3.1-8B-Instruct Q8 | ANE (CoreML, 72 dispatches) | typed fact extraction for the Subconscious | 7.9 tok/s |
| **Embedder** | MiniLM-L6-v2 | ANE (CoreML, CPU_AND_NE) | retrieval embedding | **0.84 ms/embed** (1,197 embeds/s) |
| **Classifier** | Neuron 80M (FFN-only) | ANE SRAM | domain routing | **905 µs** |
| **Drafter** | N-gram (truncate-on-miss, K=16) | software | spec decode | +17.3% over raw 5.32 tok/s baseline |

The verifier's prompt cache integration (Main 25) snapshots the system + briefing KV state on first turn and reuses it on every subsequent turn. **~30 seconds of TTFT eliminated per turn** at production briefing scale.

---

## Subconscious — the cognitive memory layer

The agent's primary product. The user never sees it; they just notice the agent remembers what they shipped.

- **5,000+ memories** in `LocalMemoryStore` (SQLite WAL + in-memory float32 numpy matrix). 13.9 MB on disk. Sub-ms cosine via single matmul. Replaced ChromaDB in Main 24.
- **Multi-path 5-signal retrieval** — `embedding (0.35) + entity (0.25) + type (0.15) + impact (0.15) + recency (0.10)` with `1.30x` canonical-state boost and activity-query category override.
- **Structured atom storage** — every memory has `atom_type`, `atom_entities`, `atom_impacts`, `atom_tense`, `atom_confidence`, `atom_core`, `source_role`. Retrieval uses all of them.
- **9 maintenance feedback loops** running hourly via launchd:
  1. `decay_scores` — exponential decay on relevance
  2. `consolidate_duplicates` — merge near-duplicate facts
  3. `resolve_contradictions` — 70B contradiction resolver during idle time
  4. `vault_sync` — supersede memories that conflict with the canonical knowledge files
  5. `production_state_sync` — sync the live infrastructure state into memory
  6. `semantic_supersession` — 3-signal paraphrase-aware supersession (cosine + tense + restate-vs-contradict)
  7. `canonical_state_inject` — parse `CLAUDE.md` tables into first-class canonical memories
  8. `meta_memory_inject` — parse session-log bullets into first-class activity memories ("what did we ship today")
  9. `vault_sweep` — surface deliverables on disk that no knowledge file references (closes the "completed work, never wired" recurring failure mode)
- **100% cross-session continuity** measured across 5 conversation sessions: 6/6 references resolved, 24/24 turns coherent, zero hallucinated events.
- See [`vault/subconscious`](https://github.com/MidasMulli/subconscious) for the loop implementations.

---

## Midas — the agent that uses the stack

`agent/midas_ui.py` runs a Flask web UI on `:8450` (network-accessible from any device on the LAN) and exposes the agent over a simple `/api/chat` endpoint.

- **Deterministic L1+L2 router** (`agent/router.py`) — keyword patterns for 90% of routing decisions, falling back to a single-word LLM classifier only when no L1 pattern matches. Tool args constructed in code, not by the LLM.
- **Cognitive context assembly** (`agent/briefing_assembler.py`, `agent/synthesizer.py`) — stable per-session briefing built from the canonical memory state, refreshed every 5 turns. Per-query memories ride in the user message tail to avoid invalidating the verifier's prefix cache.
- **Streaming SSE** with sliding-window incremental decode (no O(n²) tokenizer.decode bug).
- **Continuous embedding PoC** (`agent/continuous_embed.py`) — streams a 72B response, embeds the partial output every N tokens via the CoreML MiniLM, retrieves top-K, logs mid-generation discoveries. **0.5% wall overhead at N=4** over a 164-token generation.
- **Self-observation tools** — `self_test`, `brain_snapshot`, `self_improve` are first-class tools the agent can call on itself.

---

## Hardware

MacBook Pro M5 Pro · 64 GB unified memory · 18 CPU cores (6P + 12S) · 20 GPU cores · Metal 4 · 307 GB/s DRAM · ANE dedicated 111 GB/s DMA channel

Cross-accelerator contention measured at **+2.4% (within noise)** when GPU is idle vs busy — the ANE is bandwidth-isolated, not bandwidth-shared.

---

## Project status

The architecture and measurements behind this stack are submitted as a paper, **"Every Cycle Counts"**, awaiting arXiv endorsement at the time of writing. The paper covers heterogeneous cognitive architecture on Apple Silicon, 83% system recall on the gold set, 100% cross-session continuity, and the −4.7% concurrent contention floor.

The cognitive memory system is the primary product. The agent is the proof-of-concept. The hardware probes that landed in `nax-probe/` are the explanation for *why* this stack outperforms naive single-model deployment on the same machine.

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
