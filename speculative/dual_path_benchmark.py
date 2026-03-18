"""
Dual-Path Inference Benchmark: The Real Value Proposition
=========================================================

Three tests measuring wall-clock time for complete agentic workflows:

Test 1: Model Swap Tax — measured swap overhead for GPU model loading
Test 2: Parallel Task Decomposition — ANE classifies WHILE GPU analyzes
Test 3: Agent Routing Overhead — ANE routes, GPU executes
"""

import os
import sys
import time
import gc
import threading
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

# ── ISDA Sections ────────────────────────────────────────────────────
SIMPLE_SECTIONS = [
    {"section": "Section 1 - Interpretation",
     "text": "In this Agreement, 'Agreement' means this ISDA Master Agreement including the Schedule."},
    {"section": "Section 4(a) - General Conditions",
     "text": "Each party will make each payment specified in each Confirmation as being payable by it."},
    {"section": "Section 7 - Transfer",
     "text": "Neither party may transfer any of its rights or obligations under this Agreement without the prior written consent of the other party."},
    {"section": "Section 8 - Contractual Currency",
     "text": "Each payment shall be made in the relevant currency. If any payment is received in a different currency, the payee may convert it."},
    {"section": "Section 9 - Miscellaneous",
     "text": "All notices shall be in writing and delivered to the address specified in the Schedule."},
    {"section": "Section 10 - Offices",
     "text": "If a party acts through an Office other than its head office, it shall ensure that Office maintains proper records."},
    {"section": "Section 12 - Jurisdiction",
     "text": "With respect to any Proceedings, each party irrevocably submits to the jurisdiction of the English courts."},
]

COMPLEX_SECTIONS = [
    {"section": "Section 5(a) - Events of Default",
     "text": "The occurrence of any of the following shall constitute an Event of Default: failure to pay or deliver within 3 Local Business Days; breach of Agreement continued for 30 days; credit support default; misrepresentation; cross-default where the threshold amount exceeds 3% of shareholders equity."},
    {"section": "Section 5(b) - Termination Events",
     "text": "Illegality: it becomes unlawful for a party to perform any obligation. Tax Event: a party is required to pay additional amounts due to a change in tax law. Credit Event Upon Merger: the creditworthiness of the surviving entity is materially weaker."},
    {"section": "Section 6 - Early Termination",
     "text": "If an Event of Default has occurred, the Non-defaulting Party may designate an Early Termination Date by not more than 20 days notice. Close-out Amount shall be determined using market quotations or replacement transactions. Set-off applies across all Terminated Transactions."},
    {"section": "Schedule Part 5(f) - Bespoke",
     "text": "Notwithstanding Section 5(a)(vii), Cross Default shall apply with Threshold Amount of USD 50,000,000 and shall include Specified Indebtedness of all affiliates. Additional Termination Event: failure to maintain credit rating of at least BBB- by S&P or Baa3 by Moody's."},
    {"section": "CSA Paragraph 13 - Elections",
     "text": "Independent Amount: 5% of notional for transactions with maturity over 5 years. Minimum Transfer Amount: USD 500,000. Rounding: nearest USD 10,000. Threshold: zero if rating falls below A-/A3. Eligible Credit Support: cash in USD, EUR, GBP; US Treasuries with haircut of 2%."},
]

AGENT_TASKS = [
    {"input": "What is the threshold amount for cross-default?",
     "intent": "extraction", "complexity": "simple"},
    {"input": "Analyze whether the Additional Termination Event for credit rating downgrade is standard market practice and what risks it poses to a BBB+ rated counterparty",
     "intent": "analysis", "complexity": "complex"},
    {"input": "List all the eligible collateral types in the CSA",
     "intent": "extraction", "complexity": "simple"},
    {"input": "Compare the Early Termination provisions with the 2002 ISDA standard and identify all non-standard modifications",
     "intent": "analysis", "complexity": "complex"},
    {"input": "What is the Minimum Transfer Amount?",
     "intent": "extraction", "complexity": "simple"},
    {"input": "Draft a memo explaining how the cross-default threshold interacts with the affiliate indebtedness inclusion",
     "intent": "generation", "complexity": "complex"},
]

MAX_TOKENS_SIMPLE = 30
MAX_TOKENS_COMPLEX = 80


def main():
    import mlx.core as mx
    from mlx_lm import load as mlx_load
    from mlx_lm.models.cache import make_prompt_cache

    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  DUAL-PATH INFERENCE BENCHMARK                                 ║")
    print("║  What does the GPU waste time on that the ANE handles free?     ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print()

    # ══════════════════════════════════════════════════════════════════
    # TEST 1: MODEL SWAP TAX (measured in isolation to avoid OOM)
    # ══════════════════════════════════════════════════════════════════
    print("═" * 70)
    print("TEST 1: MODEL SWAP TAX")
    print("═" * 70)
    print()
    print("Measuring model load times (cold swap from disk)...")
    print()

    # Measure 0.6B load time
    swap_times_06b = []
    for run in range(3):
        gc.collect()
        t0 = time.time()
        m, tok = mlx_load("mlx-community/Qwen3-0.6B-4bit")
        c = make_prompt_cache(m)
        logits = m(mx.array([[1]]), cache=c)
        mx.eval(logits)
        elapsed = (time.time() - t0) * 1000
        swap_times_06b.append(elapsed)
        del m, c, logits
        gc.collect()
        print(f"  0.6B load #{run+1}: {elapsed:.0f}ms")

    # Measure 8B load time
    swap_times_8b = []
    for run in range(3):
        gc.collect()
        t0 = time.time()
        m, tok = mlx_load("mlx-community/Qwen3-8B-4bit")
        c = make_prompt_cache(m)
        logits = m(mx.array([[1]]), cache=c)
        mx.eval(logits)
        elapsed = (time.time() - t0) * 1000
        swap_times_8b.append(elapsed)
        del m, c, logits
        gc.collect()
        print(f"  8B load  #{run+1}: {elapsed:.0f}ms")

    avg_swap_06b = sum(swap_times_06b) / len(swap_times_06b)
    avg_swap_8b = sum(swap_times_8b) / len(swap_times_8b)

    print(f"\n  Average swap to 0.6B: {avg_swap_06b:.0f}ms")
    print(f"  Average swap to 8B:   {avg_swap_8b:.0f}ms")
    print()

    # 4-step agentic loop swap cost: load 0.6B → classify → load 8B → generate → load 0.6B → validate → load 8B → refine
    swap_overhead = avg_swap_06b + avg_swap_8b + avg_swap_06b + avg_swap_8b
    print(f"  4-step agentic loop swap overhead:")
    print(f"    Step 1: swap → 0.6B  {avg_swap_06b:6.0f}ms")
    print(f"    Step 2: swap → 8B    {avg_swap_8b:6.0f}ms")
    print(f"    Step 3: swap → 0.6B  {avg_swap_06b:6.0f}ms")
    print(f"    Step 4: swap → 8B    {avg_swap_8b:6.0f}ms")
    print(f"    ─────────────────────────────")
    print(f"    TOTAL DEAD TIME:     {swap_overhead:6.0f}ms  ← zero useful work done")
    print(f"    Dual-path swap cost:      0ms  ← both models always loaded")
    print()

    # ══════════════════════════════════════════════════════════════════
    # Load models for Tests 2 & 3
    # ══════════════════════════════════════════════════════════════════
    gc.collect()
    print("Loading models for Tests 2 & 3...")

    print("  Loading ANE 0.6B...")
    from real_draft import RealDraftModel
    ane_model = RealDraftModel(model_name="Qwen/Qwen3-0.6B")
    ane_model.load_and_compile(
        status_fn=lambda msg: print(f"    {msg}"),
        fused=True
    )
    gc.collect()

    print("  Loading GPU 8B...")
    gpu_model, gpu_tok = mlx_load("mlx-community/Qwen3-8B-4bit")
    print("  Both models loaded.")
    print()

    # ── Helpers ──────────────────────────────────────────────────────
    def gpu_generate(prompt, max_tokens):
        t0 = time.time()
        ids = gpu_tok.encode(prompt)
        cache = make_prompt_cache(gpu_model)
        x = mx.array([ids])
        logits = gpu_model(x, cache=cache)
        mx.eval(logits)
        tokens = []
        for _ in range(max_tokens):
            tok = mx.argmax(logits[0, -1, :]).item()
            tokens.append(tok)
            if tok == gpu_tok.eos_token_id:
                break
            x = mx.array([[tok]])
            logits = gpu_model(x, cache=cache)
            mx.eval(logits)
        text = gpu_tok.decode(tokens)
        elapsed = (time.time() - t0) * 1000
        return text, elapsed

    def ane_generate(prompt, max_tokens):
        t0 = time.time()
        ids = ane_model.encode(prompt)
        ane_model.reset_cache()
        logits = None
        for i, tid in enumerate(ids):
            logits = ane_model.forward_token(tid, i)
        pos = len(ids)
        tokens = []
        for _ in range(max_tokens):
            tok = int(np.argmax(logits))
            tokens.append(tok)
            logits = ane_model.forward_token(tok, pos)
            pos += 1
        text = ane_model.decode(tokens)
        elapsed = (time.time() - t0) * 1000
        return text, elapsed

    # ══════════════════════════════════════════════════════════════════
    # TEST 2: PARALLEL TASK DECOMPOSITION
    # ══════════════════════════════════════════════════════════════════
    print("═" * 70)
    print("TEST 2: PARALLEL TASK DECOMPOSITION")
    print("═" * 70)
    print(f"  {len(SIMPLE_SECTIONS)} simple sections (classify) + {len(COMPLEX_SECTIONS)} complex sections (analyze)")
    print()

    classify_t = "Classify as STANDARD or NON-STANDARD (one word): {text}"
    analyze_t = "Analyze this ISDA provision, identify risks: {text}"

    # ── Sequential: GPU 8B does everything ───────────────────────────
    print("── Sequential: GPU 8B processes all 12 sections ──")
    t0_total = time.time()
    simple_ms_seq = 0
    complex_ms_seq = 0

    for s in SIMPLE_SECTIONS:
        _, ms = gpu_generate(classify_t.format(text=s["text"]), MAX_TOKENS_SIMPLE)
        simple_ms_seq += ms
        print(f"    {s['section'][:35]:<35} {ms:6.0f}ms  classify")

    for s in COMPLEX_SECTIONS:
        _, ms = gpu_generate(analyze_t.format(text=s["text"]), MAX_TOKENS_COMPLEX)
        complex_ms_seq += ms
        print(f"    {s['section'][:35]:<35} {ms:6.0f}ms  analyze")

    total_seq = (time.time() - t0_total) * 1000
    print(f"    {'─' * 50}")
    print(f"    Simple:  {simple_ms_seq:7.0f}ms  |  Complex: {complex_ms_seq:7.0f}ms")
    print(f"    TOTAL:   {total_seq:7.0f}ms")
    print()

    # ── Parallel: ANE classifies WHILE GPU analyzes ──────────────────
    print("── Parallel: ANE classifies simple || GPU analyzes complex ──")

    ane_results = []
    gpu_results = []

    def ane_worker():
        for s in SIMPLE_SECTIONS:
            t0 = time.time()
            text, _ = ane_generate(classify_t.format(text=s["text"]), MAX_TOKENS_SIMPLE)
            ms = (time.time() - t0) * 1000
            ane_results.append((s["section"][:30], ms, text.strip()[:30]))

    def gpu_worker():
        for s in COMPLEX_SECTIONS:
            t0 = time.time()
            text, _ = gpu_generate(analyze_t.format(text=s["text"]), MAX_TOKENS_COMPLEX)
            ms = (time.time() - t0) * 1000
            gpu_results.append((s["section"][:30], ms))

    t0_total = time.time()
    t_ane = threading.Thread(target=ane_worker)
    t_gpu = threading.Thread(target=gpu_worker)
    t_ane.start()
    t_gpu.start()
    t_ane.join()
    t_gpu.join()
    total_parallel = (time.time() - t0_total) * 1000

    ane_subtotal = sum(ms for _, ms, _ in ane_results)
    gpu_subtotal = sum(ms for _, ms in gpu_results)

    print("    ANE (simple — runs for FREE alongside GPU):")
    for name, ms, out in ane_results:
        print(f"      {name:<30} {ms:6.0f}ms  → {out}")
    print(f"      {'ANE total:':<30} {ane_subtotal:6.0f}ms")
    print()
    print("    GPU (complex — the real work):")
    for name, ms in gpu_results:
        print(f"      {name:<30} {ms:6.0f}ms")
    print(f"      {'GPU total:':<30} {gpu_subtotal:6.0f}ms")
    print()

    speedup_t2 = total_seq / total_parallel
    time_saved = total_seq - total_parallel
    free_work = ane_subtotal  # ANE work done during GPU time

    print(f"    Sequential (GPU-only):   {total_seq:7.0f}ms")
    print(f"    Parallel (ANE || GPU):   {total_parallel:7.0f}ms")
    print(f"    Speedup:                  {speedup_t2:.2f}x")
    print(f"    Time saved:              {time_saved:7.0f}ms")
    print(f"    Free ANE compute:        {free_work:7.0f}ms of work at zero GPU cost")
    print()

    # ══════════════════════════════════════════════════════════════════
    # TEST 3: AGENT ROUTING OVERHEAD
    # ══════════════════════════════════════════════════════════════════
    print("═" * 70)
    print("TEST 3: AGENT ROUTING OVERHEAD (6 tasks × 5 steps)")
    print("═" * 70)
    print("  Steps 1,2,3,5 = routing (lightweight)")
    print("  Step 4 = execution (heavy reasoning)")
    print()

    intent_p = "Classify intent as extraction, analysis, or generation (one word): {input}"
    skill_p = "Select tool: search, analyze, or generate (one word): {input}"
    param_p = "Extract the key entity (one phrase): {input}"
    exec_p = "Based on ISDA document context, {input}"
    valid_p = "Is this output relevant? Yes or no: {output}"

    # ── GPU-only: 8B does all 5 steps ────────────────────────────────
    print("── GPU-only: 8B handles all 5 steps ──")
    t0_total = time.time()
    routing_gpu = 0
    exec_gpu = 0

    for task in AGENT_TASKS:
        t0 = time.time()
        gpu_generate(intent_p.format(input=task["input"]), 10)
        gpu_generate(skill_p.format(input=task["input"]), 10)
        gpu_generate(param_p.format(input=task["input"]), 15)
        route_ms = (time.time() - t0) * 1000

        t0e = time.time()
        max_t = MAX_TOKENS_COMPLEX if task["complexity"] == "complex" else MAX_TOKENS_SIMPLE
        r4, _ = gpu_generate(exec_p.format(input=task["input"]), max_t)
        exec_ms = (time.time() - t0e) * 1000

        t0v = time.time()
        gpu_generate(valid_p.format(output=r4[:80]), 10)
        val_ms = (time.time() - t0v) * 1000

        routing = route_ms + val_ms
        routing_gpu += routing
        exec_gpu += exec_ms
        total_task = routing + exec_ms
        c = task["complexity"][0].upper()
        print(f"    [{c}] route={routing:5.0f}ms  exec={exec_ms:5.0f}ms  total={total_task:5.0f}ms  {task['input'][:42]}")

    total_gpu_only = (time.time() - t0_total) * 1000
    print(f"    {'─' * 70}")
    print(f"    Routing (on GPU):  {routing_gpu:7.0f}ms  ({routing_gpu/total_gpu_only*100:.0f}% of workflow)")
    print(f"    Execution:         {exec_gpu:7.0f}ms")
    print(f"    TOTAL:             {total_gpu_only:7.0f}ms")
    print()

    # ── Dual-path: ANE routes, GPU executes ──────────────────────────
    print("── Dual-path: ANE routes (steps 1,2,3,5) + GPU executes (step 4) ──")
    t0_total = time.time()
    routing_dual = 0
    exec_dual = 0

    for task in AGENT_TASKS:
        t0 = time.time()
        ane_generate(intent_p.format(input=task["input"]), 10)
        ane_generate(skill_p.format(input=task["input"]), 10)
        ane_generate(param_p.format(input=task["input"]), 15)
        route_ms = (time.time() - t0) * 1000

        t0e = time.time()
        max_t = MAX_TOKENS_COMPLEX if task["complexity"] == "complex" else MAX_TOKENS_SIMPLE
        r4, _ = gpu_generate(exec_p.format(input=task["input"]), max_t)
        exec_ms = (time.time() - t0e) * 1000

        t0v = time.time()
        ane_generate(valid_p.format(output=r4[:80]), 10)
        val_ms = (time.time() - t0v) * 1000

        routing = route_ms + val_ms
        routing_dual += routing
        exec_dual += exec_ms
        total_task = routing + exec_ms
        c = task["complexity"][0].upper()
        print(f"    [{c}] route={routing:5.0f}ms  exec={exec_ms:5.0f}ms  total={total_task:5.0f}ms  {task['input'][:42]}")

    total_dual = (time.time() - t0_total) * 1000
    print(f"    {'─' * 70}")
    print(f"    Routing (on ANE):  {routing_dual:7.0f}ms  (frees GPU entirely)")
    print(f"    Execution:         {exec_dual:7.0f}ms")
    print(f"    TOTAL:             {total_dual:7.0f}ms")
    print()

    # ══════════════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ══════════════════════════════════════════════════════════════════
    rt_speedup = total_gpu_only / total_dual if total_dual > 0 else 0
    routing_freed = routing_gpu - routing_dual

    print()
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  FINAL RESULTS                                                 ║")
    print("╠══════════════════════════════════════════════════════════════════╣")
    print("║                                                                ║")
    print(f"║  TEST 1: Model Swap Tax                                        ║")
    print(f"║    0.6B load: {avg_swap_06b:5.0f}ms    8B load: {avg_swap_8b:5.0f}ms                    ║")
    print(f"║    4-step agent loop swap overhead: {swap_overhead:5.0f}ms                   ║")
    print(f"║    Dual-path swap overhead:             0ms                    ║")
    print("║                                                                ║")
    print(f"║  TEST 2: Parallel Decomposition (12 ISDA sections)             ║")
    print(f"║    Sequential (GPU-only): {total_seq:6.0f}ms                              ║")
    print(f"║    Parallel (ANE || GPU): {total_parallel:6.0f}ms                              ║")
    print(f"║    Speedup: {speedup_t2:.2f}x                                            ║")
    print(f"║    ANE did {len(SIMPLE_SECTIONS)} sections for FREE during GPU time              ║")
    print("║                                                                ║")
    print(f"║  TEST 3: Agent Routing (6 tasks × 5 steps)                     ║")
    print(f"║    GPU-only:   {total_gpu_only:6.0f}ms  (routing: {routing_gpu/total_gpu_only*100:.0f}% on GPU)          ║")
    print(f"║    Dual-path:  {total_dual:6.0f}ms  (routing: offloaded to ANE)        ║")
    print(f"║    Speedup: {rt_speedup:.2f}x                                            ║")
    print("║                                                                ║")
    print("╠══════════════════════════════════════════════════════════════════╣")
    print("║                                                                ║")
    print("║  THE VALUE PROPOSITION:                                         ║")
    print(f"║    Model swap tax eliminated:     {swap_overhead:5.0f}ms per agentic loop       ║")
    print(f"║    Parallel decomposition:        {speedup_t2:.2f}x on mixed workloads        ║")
    print(f"║    GPU freed from routing:        {routing_freed:5.0f}ms per 6-task batch       ║")
    print("║                                                                ║")
    print("║  Same hardware. Zero additional cost. Two compute paths.        ║")
    print("║                                                                ║")
    print("╚══════════════════════════════════════════════════════════════════╝")


if __name__ == "__main__":
    main()
