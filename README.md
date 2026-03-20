# Orion ANE

Two projects in one repo, both pushing Apple Silicon beyond what Apple intended.

## [Phantom Memory](memory/) — Persistent Memory + Continuous Enrichment for Local LLMs

Zero-cost persistent memory with a self-improving knowledge graph. Extracts facts, embeds them, stores in ChromaDB, writes to an Obsidian vault — then continuously enriches the vault in the background with five autonomous sweeps: reclassify, relate, stale-detect, pattern-find, and consolidate.

Mem0 stores facts. **Phantom thinks about them.**

```
┌─ GPU ── Your LLM (conversation + reasoning) ──────────┐
┌─ CPU ── Memory Daemon (extract, embed, store) ─────────┐  1,721 emb/sec (MiniLM, M5 CPU)
┌─ ANE ── Enricher (classify, relate, analyze) ──────────┐  ~2W
```

Three processors. Three loops. Near-zero contention (~3.8% measured). **[Full documentation →](memory/README.md)**

---

## ANE Training — Backpropagation on Apple Neural Engine

Training neural networks directly on Apple's Neural Engine (ANE) via reverse-engineered private APIs. No CoreML training APIs, no Metal, no GPU — pure ANE compute.

## Project Scope & Intent

I'm genuinely grateful for all the attention this project has received — I never expected a weekend research hack to blow up like this. Thank you to everyone who starred, forked, ran benchmarks on their own hardware, and shared the work. It means a lot.

That said, I want to set clear expectations about what this project is and isn't.

This is a **research project**, not a production framework.

The goal was to demonstrate that **training on the Apple Neural Engine — and potentially other NPUs — is possible**, and that the barrier has always been software support, not hardware capability. The ANE is a remarkably capable piece of silicon that Apple restricts to inference-only use through CoreML. This project bypasses that restriction using reverse-engineered private APIs to show what's possible when you give the hardware a chance.

### What This Project Is

- A proof of concept for ANE training via `_ANEClient` and `_ANECompiler` private APIs
- A set of benchmarks documenting real ANE performance characteristics (throughput, power, SRAM behavior)
- A reference for anyone exploring direct ANE access outside CoreML
- Research code that I update when I find something interesting

### What This Project Is Not

- A maintained framework or library
- A replacement for CoreML, MLX, llama.cpp, or any production inference stack
- A path to training large models on consumer hardware (yet)

### On The Hype

Some coverage of this project has overstated its implications. To be clear:

- Training works, but utilization is low (~5-9% of peak) with significant engineering challenges remaining
- Many element-wise operations still fall back to CPU
- This does **not** replace GPU training for anything beyond small research models today

The honest results — including all limitations — are documented in the accompanying articles:
- [Part 1: Reverse Engineering](https://maderix.substack.com/p/inside-the-m4-apple-neural-engine)
- [Part 2: Benchmarks](https://maderix.substack.com/p/inside-the-m4-apple-neural-engine-615)
- [Part 3: Training](https://maderix.substack.com/p/inside-the-m4-apple-neural-engine-c8b)

### On Maintenance

I don't intend to grow this into a large community project. My focus is on original research (compiler infrastructure for edge AI optimization), and maintaining an open-source framework takes time away from that.

That said:
- I'll keep pushing updates when I discover something interesting
- Bug fixes and benchmark contributions (especially on hardware I don't own) are welcome
- Feature requests will likely go unaddressed — but feel free to fork
- PRs will be merged at a relatively slow pace, otherwise I become the bottleneck for community growth around this tech

### Fork it, build on it

This is MIT licensed for a reason. Everyone now has access to AI-assisted development tools that can adapt and extend code in hours. If this project is useful to you — take it, modify it, build something better. If you do something cool with it, I'd love to hear about it.If in future, community decides to maintain one source of truth repo, I'm in full support of that.

---

## What This Is

A from-scratch implementation of transformer training (forward + backward pass) running on the ANE in Apple Silicon. The ANE is a 15.8 TFLOPS FP16 (M4) inference accelerator that Apple does not expose for training. This project reverse-engineers the `_ANEClient` / `_ANECompiler` private APIs and the MIL (Model Intermediate Language) format to run custom compute graphs — including backpropagation — directly on ANE hardware.

**Current results:**

| Model | Params | ms/step | Pipeline |
|-------|--------|---------|----------|
| Stories110M (12L, dim=768, MHA 12/12) | 109M | **91 ms** | Dynamic (no recompile) |
| Qwen3-0.6B (28L, dim=1024, GQA 16/8) | 596M | **412 ms** | Dynamic (no recompile) |

- All forward and backward dx passes on ANE, dW gradients on CPU (Accelerate cblas)
- Adam optimizer, gradient accumulation, checkpoint/resume via exec() restart
- GQA (Grouped-Query Attention) support with per-head tiling/reduction
- GPU↔ANE zero-copy pipeline via shared IOSurface (GPU prefill → ANE decode)

**INT8 W8A8 quantization — 1.88x throughput (M4, H16G):**

| Config | FP16 | INT8 W8A8 | Speedup |
|--------|------|-----------|---------|
| 128x conv 512ch 64x64 | 18.6 TOPS, 14.8ms | 35.1 TOPS, 7.8ms | **1.88x** |
| 64x conv 512ch 64x64 | 18.4 TOPS, 7.5ms | 34.1 TOPS, 4.0ms | **1.85x** |

INT8 activations halve L2 SRAM bandwidth between tiles via MIL `quantize`/`dequantize` ops. Weights use `constexpr_affine_dequantize` (int8 stored, fp16 at compile time).

## Architecture

The dynamic pipeline uses shared ANE kernels with weights packed into spatial dimensions (no recompilation when weights change):

**MHA models (Stories110M) — 6 kernels per layer:**

| Kernel | Function |
|--------|----------|
| `sdpaFwd` | QKV projection + SDPA + output projection |
| `ffnFused` | SwiGLU FFN (W1, W3, SiLU, W2) |
| `ffnBwdW2t` / `ffnBwdW13t` | FFN backward (split for memory) |
| `sdpaBwd1` / `sdpaBwd2` | SDPA backward |

**GQA models (Qwen3-0.6B) — 10 kernels per layer:**
Adds separate `woFwd`, `qBwd`, `kvBwd` kernels for grouped-query attention (Q_DIM ≠ DIM).

CPU handles: RMSNorm forward/backward, residual connections (DeepNet α scaling), loss computation, dW gradient accumulation (cblas_sgemm), Adam optimizer updates.

Key optimizations:
- **Channel-first CPU layout** — matches ANE IOSurface `[1,C,1,S]` format, eliminates all transpose overhead
- **vDSP vectorized RMSNorm** — 10x faster than naive (6.7ms → 0.7ms)
- **GCD async cblas overlap** — dW gradient sgemms run in parallel with ANE evals on a serial dispatch queue
- **Deferred cblas wait** — wait pushed into next step's forward pass for maximum overlap
- **ANE RMSNorm fusion** — RMSNorm folded into forward kernels as MIL ops (reduce_sum + pow + mul)
- **Wo^T fusion** — output projection backward merged into SDPA backward kernel
- **Forward taps** — Q, K, V, attention scores, hidden states exposed via concat outputs, avoiding CPU recompute
- **exec() restart** — bypasses ~119 ANE compile limit per process

## File Structure

```
├── memory/
│   ├── README.md               # Phantom Memory documentation
│   ├── daemon.py               # Memory daemon (extract, embed, store, vault write)
│   ├── enricher.py             # Continuous enrichment (5 sweeps, ANE-ready protocols)
│   ├── ane_server.py           # Persistent ANE inference server (Unix socket + HTTP)
│   ├── eval_tiers.py           # Three-tier validation suite (22/22)
│   ├── dashboard.py            # TUI monitoring dashboard
│   ├── mcp_server.py           # MCP server wrapper
│   └── scale_test.py           # Scalability benchmarks
├── agent/
│   ├── agent.py                # Phantom agent (Midas) — tools, boot, conversation loop
│   ├── browser.py              # Chrome CDP client for authenticated browsing
│   └── monitor.py              # System performance dashboard
├── api_exploration.m           # Initial ANE API discovery
├── inmem_basic.m               # In-memory MIL compilation proof-of-concept
├── inmem_bench.m               # ANE dispatch latency benchmarks
├── inmem_peak.m                # Peak TFLOPS measurement (2048x2048 matmul)
├── ane_int8_bench.m            # INT8 W8A8 vs FP16 throughput benchmark
├── sram_bench.m                # ANE SRAM bandwidth probing
├── sram_probe.m                # SRAM size/layout exploration
├── gpu_ane_share.m             # GPU↔ANE zero-copy IOSurface demo
├── gpu_prefill_ane_decode.m    # GPU prefill → ANE decode pipeline
├── bridge/
│   ├── ane_bridge.h            # C-callable ANE API (compile, eval, I/O)
│   ├── ane_bridge.m            # Bridge implementation (int8 + fp16 weight blobs)
│   ├── forward_pass.h          # C forward pass API (fused/unfused kernel modes)
│   ├── forward_pass.c          # Full transformer inference in C (vDSP + ANE)
│   └── Makefile
├── bench_speculative_local.py   # Same-process speculative decode benchmark
├── speculative/
│   ├── real_draft.py           # Qwen3-0.6B ANE draft model (Python loader + compiler)
│   ├── mlx_local_verifier.py   # In-process MLX verifier (persistent KV cache)
│   └── ane_draft.py            # ANE kernel compilation helpers
├── test_c_forward.py           # C vs Python forward pass comparison + benchmarks
└── training/
    ├── ane_runtime.h           # ANE private API wrapper (compile, eval, IOSurface)
    ├── ane_classifier.h        # Classifier fwd (32K conv), softmax, rmsnorm on ANE
    ├── train_large.m           # Static pipeline (weights as constants, recompiles)
    ├── training_dynamic/
    │   ├── train.m             # Dynamic training loop (model-agnostic)
    │   ├── config.h            # Derived sizes, structs, alloc helpers
    │   ├── mil_dynamic.h       # MIL generators for dynamic weight kernels (GQA-aware)
    │   ├── io.h                # IOSurface I/O, weight staging, GQA tile/reduce
    │   ├── models/
    │   │   ├── stories110m.h   # Stories110M config (12L, MHA)
    │   │   └── qwen3_06b.h    # Qwen3-0.6B config (28L, GQA)
    │   └── Makefile
    ├── dashboard.py            # Live training dashboard (blessed TUI)
    └── Makefile
```

## Training Data

Training requires pretokenized TinyStories data. To download:
```bash
cd training && bash download_data.sh
```
See [training/README.md](training/README.md) for detailed training instructions.

## Building

Requires macOS 15+ on Apple Silicon. Tested on M4 (training) and M5 Air 16GB (memory daemon, agent). macOS 26.3 (Tahoe).

```bash
# Dynamic pipeline (recommended) — model selected at build time
cd training/training_dynamic
make MODEL=stories110m    # Stories110M (12L, MHA, 109M params)
make MODEL=qwen3_06b      # Qwen3-0.6B (28L, GQA, 596M params)
./train --scratch          # train from random init
./train --resume           # resume from checkpoint

# Static pipeline (legacy — recompiles weights each step)
cd training && make train_large
./train_large ane_stories110M_ckpt.bin 256 100 1e-4

# INT8 benchmark
xcrun clang -O2 -fobjc-arc -framework Foundation -framework IOSurface -ldl \
  -o ane_int8_bench ane_int8_bench.m
./ane_int8_bench

# Bridge library (C-callable ANE API)
cd bridge && make
```

No external dependencies. Uses only system frameworks + private ANE APIs resolved at runtime via `objc_msgSend`.

## How It Works

1. **MIL generation** — Objective-C code constructs MIL program text at runtime, specifying convolutions (for linear layers), matmul (for attention), softmax, element-wise ops
2. **In-memory compilation** — `_ANEInMemoryModelDescriptor` compiles MIL text + weight blobs directly to ANE programs, no disk mlmodelc needed
3. **IOSurface I/O** — Input/output tensors passed via IOSurface shared memory in `[1, channels, 1, spatial]` format (fp16 or fp32; fp16 direct I/O is ~37% faster)
4. **Dynamic weights** — Activations and weights packed into a single spatial input dimension, sliced apart inside the MIL kernel. Weights change without recompilation.
5. **Gradient flow** — Forward taps expose intermediates needed for backward; backward kernels compute dx (input gradients) on ANE; dW (weight gradients) computed on CPU via cblas
6. **INT8 quantization** — `constexpr_affine_dequantize` for int8 weights, `quantize`/`dequantize` between layers for int8 activation caching in L2 SRAM (1.88x throughput)

## Limitations

- **SDPA causal masking** — ANE hardware ignores `attn_mask` in SDPA ops; causal attention is decomposed into separate Q@K^T (ANE) → mask+softmax (CPU) → scores@V (ANE)
- **~119 compile limit** — ANE compiler leaks resources; worked around via `exec()` restart with checkpoint
- **FP16 gradient underflow** — backward matmuls underflow in fp16; fixed with global loss scaling (`256 * NLAYERS`)
- **Single-input constraint** — multi-input ANE requests cause 0x1d error; inputs packed into spatial dimension instead

## Performance

**Training throughput (M4):**

| Model | Params | ms/step | Layers | Kernels/layer |
|-------|--------|---------|--------|---------------|
| Stories110M | 109M | 91 ms | 12 | 6 (MHA) |
| Qwen3-0.6B | 596M | 412 ms | 28 | 10 (GQA) |

**ANE peak throughput (M4, H16G):**

| Precision | Peak TOPS | Config |
|-----------|-----------|--------|
| FP16 | 18.6 | 128x conv 512ch 64x64 |
| INT8 W8A8 | 35.1 | 128x conv 512ch 64x64 |

**GPU↔ANE inference pipeline (M4, seq=256):**

| Model | GPU Prefill | ANE Decode | Total |
|-------|------------|------------|-------|
| Stories110M | 6.7ms | 1.9ms | 8.8ms |
| Qwen3-0.6B | 9.7ms | 2.3ms | 12.0ms |

**C forward pass — ANE-native token generation (M5 Air):**

Full transformer inference in C with fused ANE kernels. Zero Python in the hot loop — CPU ops (RMSNorm, RoPE, attention, SiLU) via Accelerate vDSP/cblas, all linear projections dispatched to ANE via direct `_ANEClient` eval.

| Metric | Unfused (206 dispatches) | Fused (122 dispatches) |
|--------|--------------------------|------------------------|
| ANE eval time | ~139ms | 33.8ms |
| Total latency | ~145ms/tok | 37.6ms/tok |
| Throughput | ~7 tok/s | **26.6 tok/s** (avg), **32.7 tok/s** (peak) |
| CPU ops | 0.37ms | 0.68ms |

Kernel fusion concatenates Q+K+V weights into a single ANE kernel (3 dispatches → 1) and Gate+Up into another (2 → 1), reducing per-layer dispatches from 7 to 4. The fused kernels are also individually faster — larger matrices let the ANE batch more efficiently.

**Speculative decoding — ANE draft + GPU verify (M5 Air, 16GB):**

Same-process speculative decoding with Qwen3-0.6B on ANE drafting K=5 tokens, verified by Qwen3-4B on GPU via MLX. Both models loaded in one process, no HTTP overhead, batch verification in a single forward pass.

| Method | Model | Tokens | Time | Throughput |
|--------|-------|--------|------|------------|
| GPU-only | Qwen3-4B (MLX) | 30 | 693ms | **43.3 tok/s** |
| Speculative | 0.6B ANE + 4B GPU | 31 | 6,031ms | 5.1 tok/s |

| Metric | Value |
|--------|-------|
| Acceptance rate | 48.3% (14/29 rounds accepted) |
| ANE draft time | 2,686ms (44.5%) |
| GPU verify time | 3,345ms (55.5%) |
| Tokens per round | 3.4 avg |

**Verdict:** The architecture works correctly — tokenizer-compatible models, proper KV cache management, batch verification, incremental drafting. But speculative decoding requires the draft model to be *much* faster than the verifier per-token. On M5 Air 16GB, the ANE draft (44ms/tok) is slower than the GPU verifier (23ms/tok for 4B), so there's no speedup. The correct configuration needs a larger, slower verifier — e.g., 14B+ on 32GB+ hardware where GPU autoregressive drops to ~10 tok/s but ANE stays at 26 tok/s.

## Disclaimer

This project uses Apple's private, undocumented APIs (`_ANEClient`, `_ANECompiler`, `_ANEInMemoryModelDescriptor`). These APIs are not covered by any public stability guarantee and may change or break with any macOS update. This is independent research into Apple Neural Engine architecture, using APIs discovered through runtime introspection for research and educational purposes under fair use and interoperability provisions (see *Sega v. Accolade*, 1992; DMCA §1201(f)). No Apple proprietary code or binaries are included in this repository. This project is not affiliated with or endorsed by Apple Inc. Use at your own risk.

## License

MIT — see [LICENSE](LICENSE)

---

## Related

- [dual-path-inference](https://github.com/MidasMulli/dual-path-inference) — Initial GPU+ANE concurrency proof-of-concept (archived, evolved into this repo)
- [four-path-mlx](https://github.com/MidasMulli/four-path-mlx) — Multi-source speculative decoding server using N-gram + PLD + ANE + GPU
- [gdn-coreml](https://github.com/MidasMulli/gdn-coreml) — GatedDeltaNet SSM to CoreML converter for same-family ANE drafting


