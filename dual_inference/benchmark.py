"""
Dual-Path Inference Benchmark
==============================

Tests the smart routing engine against GPU-only baseline across
real ISDA workflows of increasing complexity.

Scenarios:
  1. Simple batch (all classify) — should route to GPU (no parallel opportunity)
  2. Complex batch (all analyze) — should route to GPU
  3. Mixed batch (classify + analyze) — parallel opportunity → speedup
  4. Compound ISDA workflow — decomposed into parallel subtasks
  5. Heavy compound — 20-section document processing
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from engine import DualPathEngine, Task, ComputePath


# ── Test Data ────────────────────────────────────────────────────────

# ISDA sections — simple (formulaic, standard language)
SIMPLE_SECTIONS = [
    "In this Agreement, 'Agreement' means this ISDA Master Agreement including the Schedule.",
    "Each party will make each payment specified in each Confirmation as being payable by it.",
    "Neither party may transfer any of its rights without the prior written consent of the other party.",
    "Each payment shall be made in the relevant currency specified in the Confirmation.",
    "All notices shall be in writing and delivered to the address specified in the Schedule.",
    "If a party acts through an Office other than its head office, it shall ensure proper records.",
    "With respect to any Proceedings, each party submits to the jurisdiction of the English courts.",
    "This Agreement constitutes the entire agreement and supersedes all prior negotiations.",
    "No amendment shall be effective unless in writing and signed by each party.",
    "Each party represents that it has authority to enter into this Agreement.",
]

# ISDA sections — complex (bespoke, non-standard, require analysis)
COMPLEX_SECTIONS = [
    "The occurrence of any Event of Default: failure to pay within 3 Local Business Days; breach continued for 30 days; credit support default; misrepresentation; cross-default where threshold exceeds 3% of shareholders equity including affiliate Specified Indebtedness.",
    "Illegality: unlawful to perform any obligation. Tax Event: required to pay additional amounts due to change in tax law enacted after trade date. Credit Event Upon Merger: creditworthiness of surviving entity is materially weaker as determined by the Calculation Agent.",
    "Non-defaulting Party may designate Early Termination Date by not more than 20 days notice. Close-out Amount determined using market quotations or replacement transactions. Set-off applies across all Terminated Transactions and any other amounts owing.",
    "Cross Default shall apply with Threshold Amount of USD 50,000,000 and shall include Specified Indebtedness of all affiliates. Additional Termination Event: failure to maintain credit rating of at least BBB- by S&P or Baa3 by Moody's.",
    "Independent Amount: 5% of notional for transactions with maturity over 5 years. Minimum Transfer Amount: USD 500,000. Threshold: zero if rating falls below A-/A3. Eligible Credit Support: cash in USD/EUR/GBP; US Treasuries haircut 2%; investment grade corporate bonds haircut 5%.",
    "Upon occurrence of Additional Termination Event relating to NAV decline exceeding 25% in any rolling 12-month period, the Non-Affected Party may terminate all outstanding Transactions. Calculation of NAV shall reference audited financial statements.",
    "Force Majeure Event: if by reason of force majeure or act of state, a party is prevented from making or receiving payment for 8 Local Business Days. Waiting Period: 14 calendar days. Each party may terminate affected transactions after the Waiting Period.",
    "Automatic Early Termination shall apply to Party B but not Party A. Notwithstanding Section 6(a), if an Event of Default specified in Section 5(a)(vii)(1)(3)(6) occurs with respect to Party B, an Early Termination Date shall occur immediately.",
]

# Compound ISDA prompts (single prompt requiring decomposition)
COMPOUND_PROMPTS = [
    {
        "name": "Classify + Analyze",
        "description": "Classify all 18 sections AND analyze non-standard provisions",
        "simple_prompt": "Classify as STANDARD or NON-STANDARD (one word): {text}",
        "complex_prompt": "Analyze this ISDA provision. Identify non-standard language, risk implications, and comparison to market standard: {text}",
    },
    {
        "name": "Extract + Draft",
        "description": "Extract key terms from 10 simple sections AND draft risk memo for 8 complex sections",
        "simple_prompt": "Extract the key defined term from this clause (one phrase): {text}",
        "complex_prompt": "Draft a risk assessment paragraph for this ISDA provision, covering counterparty exposure, regulatory implications, and recommended modifications: {text}",
    },
    {
        "name": "Validate + Recommend",
        "description": "Validate standard sections AND generate recommendations for bespoke provisions",
        "simple_prompt": "Is this clause standard ISDA 2002 Master Agreement language? Yes or No: {text}",
        "complex_prompt": "For this non-standard ISDA provision, provide: (1) how it deviates from market standard, (2) risk to our position, (3) recommended counter-proposal language: {text}",
    },
]


def print_header(title):
    print()
    print("═" * 74)
    print(f"  {title}")
    print("═" * 74)


def print_results(label, results, wall_ms):
    gpu_count = sum(1 for r in results if r.path == ComputePath.GPU)
    ane_count = sum(1 for r in results if r.path == ComputePath.ANE)
    gpu_ms = sum(r.elapsed_ms for r in results if r.path == ComputePath.GPU)
    ane_ms = sum(r.elapsed_ms for r in results if r.path == ComputePath.ANE)
    total_inference = sum(r.elapsed_ms for r in results)

    print(f"  {label}:")
    print(f"    Tasks:      {len(results)} total  ({gpu_count} GPU, {ane_count} ANE)")
    print(f"    Inference:  {total_inference:,.0f}ms  (GPU: {gpu_ms:,.0f}ms, ANE: {ane_ms:,.0f}ms)")
    print(f"    Wall clock: {wall_ms:,.0f}ms")
    if total_inference > wall_ms and ane_count > 0:
        overlap = total_inference - wall_ms
        print(f"    Overlap:    {overlap:,.0f}ms  (ANE work hidden behind GPU)")
    return wall_ms


def main():
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║  DUAL-PATH INFERENCE ENGINE — BENCHMARK                            ║")
    print("║  Smart routing: ANE parallel + GPU, zero-cost classification        ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print()

    # ── Initialize Engine ────────────────────────────────────────────
    engine = DualPathEngine(verbose=True)
    engine.load()
    print()

    all_gpu_times = []
    all_dual_times = []

    # ══════════════════════════════════════════════════════════════════
    # SCENARIO 1: All simple tasks (no parallel opportunity)
    # ══════════════════════════════════════════════════════════════════
    print_header("SCENARIO 1: All Simple Tasks (5 classify)")

    tasks = [Task(
        prompt=f"Classify as STANDARD or NON-STANDARD (one word): {text}",
        max_tokens=15,
        task_type="classify",
    ) for text in SIMPLE_SECTIONS[:5]]

    print(f"  Router decision: {engine.scheduler.schedule(tasks)}")
    print()

    t0 = time.time()
    gpu_results = engine.execute(tasks, mode="gpu_only")
    gpu_ms = (time.time() - t0) * 1000

    t0 = time.time()
    dual_results = engine.execute(tasks, mode="auto")
    dual_ms = (time.time() - t0) * 1000

    print_results("GPU-only", gpu_results, gpu_ms)
    print_results("Dual-path", dual_results, dual_ms)
    print(f"    Speedup: {gpu_ms/dual_ms:.2f}x")
    all_gpu_times.append(gpu_ms)
    all_dual_times.append(dual_ms)

    # ══════════════════════════════════════════════════════════════════
    # SCENARIO 2: All complex tasks (no parallel opportunity)
    # ══════════════════════════════════════════════════════════════════
    print_header("SCENARIO 2: All Complex Tasks (5 analyze)")

    tasks = [Task(
        prompt=f"Analyze this ISDA provision in detail, identify risks: {text}",
        max_tokens=80,
        task_type="analyze",
    ) for text in COMPLEX_SECTIONS[:5]]

    plan = engine.scheduler.schedule(tasks)
    print(f"  Router: {len(plan.gpu_tasks)} GPU, {len(plan.ane_tasks)} ANE, parallel={plan.parallel}")
    print()

    t0 = time.time()
    gpu_results = engine.execute(tasks, mode="gpu_only")
    gpu_ms = (time.time() - t0) * 1000

    t0 = time.time()
    dual_results = engine.execute(tasks, mode="auto")
    dual_ms = (time.time() - t0) * 1000

    print_results("GPU-only", gpu_results, gpu_ms)
    print_results("Dual-path", dual_results, dual_ms)
    print(f"    Speedup: {gpu_ms/dual_ms:.2f}x")
    all_gpu_times.append(gpu_ms)
    all_dual_times.append(dual_ms)

    # ══════════════════════════════════════════════════════════════════
    # SCENARIO 3: Mixed batch (parallel opportunity!)
    # ══════════════════════════════════════════════════════════════════
    print_header("SCENARIO 3: Mixed Batch (7 simple + 5 complex)")

    simple_tasks = [Task(
        prompt=f"Classify as STANDARD or NON-STANDARD (one word): {text}",
        max_tokens=15,
        task_type="classify",
    ) for text in SIMPLE_SECTIONS[:7]]

    complex_tasks = [Task(
        prompt=f"Analyze this ISDA provision. Identify non-standard language and risk implications: {text}",
        max_tokens=80,
        task_type="analyze",
    ) for text in COMPLEX_SECTIONS[:5]]

    tasks = simple_tasks + complex_tasks
    plan = engine.scheduler.schedule(tasks)
    print(f"  Router: {len(plan.gpu_tasks)} GPU, {len(plan.ane_tasks)} ANE, parallel={plan.parallel}")
    print()

    t0 = time.time()
    gpu_results = engine.execute(tasks, mode="gpu_only")
    gpu_ms = (time.time() - t0) * 1000

    t0 = time.time()
    dual_results = engine.execute(tasks, mode="auto")
    dual_ms = (time.time() - t0) * 1000

    print_results("GPU-only", gpu_results, gpu_ms)
    print_results("Dual-path", dual_results, dual_ms)
    speedup = gpu_ms / dual_ms
    print(f"    Speedup: {speedup:.2f}x  {'✅ Parallel win!' if speedup > 1.1 else ''}")
    all_gpu_times.append(gpu_ms)
    all_dual_times.append(dual_ms)

    # ══════════════════════════════════════════════════════════════════
    # SCENARIO 4: Compound ISDA Workflows
    # ══════════════════════════════════════════════════════════════════
    for compound in COMPOUND_PROMPTS:
        print_header(f"SCENARIO 4: {compound['name']} — {compound['description']}")

        # Decompose: simple sections get simple prompt, complex get complex prompt
        tasks = []
        for text in SIMPLE_SECTIONS:
            tasks.append(Task(
                prompt=compound["simple_prompt"].format(text=text),
                max_tokens=20,
                task_type="classify",
            ))
        for text in COMPLEX_SECTIONS:
            tasks.append(Task(
                prompt=compound["complex_prompt"].format(text=text),
                max_tokens=120,
                task_type="analyze",
            ))

        plan = engine.scheduler.schedule(tasks)
        print(f"  Decomposed: {len(tasks)} subtasks → {len(plan.gpu_tasks)} GPU, {len(plan.ane_tasks)} ANE")
        print(f"  Parallel execution: {plan.parallel}")
        print()

        t0 = time.time()
        gpu_results = engine.execute(tasks, mode="gpu_only")
        gpu_ms = (time.time() - t0) * 1000

        t0 = time.time()
        dual_results = engine.execute(tasks, mode="auto")
        dual_ms = (time.time() - t0) * 1000

        print_results("GPU-only", gpu_results, gpu_ms)
        print()
        print_results("Dual-path", dual_results, dual_ms)
        speedup = gpu_ms / dual_ms
        print(f"    Speedup: {speedup:.2f}x  {'✅ Parallel win!' if speedup > 1.1 else ''}")
        all_gpu_times.append(gpu_ms)
        all_dual_times.append(dual_ms)

    # ══════════════════════════════════════════════════════════════════
    # SCENARIO 5: Heavy Workload — 18 sections, long analysis
    # ══════════════════════════════════════════════════════════════════
    print_header("SCENARIO 5: Full Document — 10 simple + 8 complex, detailed analysis")

    tasks = []
    for text in SIMPLE_SECTIONS:
        tasks.append(Task(
            prompt=f"Extract the key defined term or obligation (one phrase): {text}",
            max_tokens=20,
            task_type="extract",
        ))
    for text in COMPLEX_SECTIONS:
        tasks.append(Task(
            prompt=f"Provide a comprehensive risk analysis of this ISDA provision. Cover: (1) deviation from market standard, (2) counterparty exposure, (3) regulatory implications, (4) recommended modifications with draft language: {text}",
            max_tokens=150,
            task_type="analyze",
        ))

    plan = engine.scheduler.schedule(tasks)
    print(f"  Full document: {len(tasks)} sections → {len(plan.gpu_tasks)} GPU, {len(plan.ane_tasks)} ANE")
    print(f"  Parallel execution: {plan.parallel}")
    print()

    t0 = time.time()
    gpu_results = engine.execute(tasks, mode="gpu_only")
    gpu_ms = (time.time() - t0) * 1000

    t0 = time.time()
    dual_results = engine.execute(tasks, mode="auto")
    dual_ms = (time.time() - t0) * 1000

    print_results("GPU-only", gpu_results, gpu_ms)
    print()
    print_results("Dual-path", dual_results, dual_ms)
    speedup = gpu_ms / dual_ms
    print(f"    Speedup: {speedup:.2f}x  {'✅ Parallel win!' if speedup > 1.1 else ''}")
    all_gpu_times.append(gpu_ms)
    all_dual_times.append(dual_ms)

    # ══════════════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ══════════════════════════════════════════════════════════════════
    print()
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║  BENCHMARK SUMMARY                                                 ║")
    print("╠══════════════════════════════════════════════════════════════════════╣")

    scenario_names = [
        "S1: All simple (no parallel)",
        "S2: All complex (no parallel)",
        "S3: Mixed batch (7+5)",
        "S4a: Classify + Analyze",
        "S4b: Extract + Draft",
        "S4c: Validate + Recommend",
        "S5: Full Document (10+8)",
    ]

    print("║                                                                    ║")
    print(f"║  {'Scenario':<30} {'GPU-only':>9} {'Dual':>9} {'Speedup':>9}   ║")
    print(f"║  {'─'*30} {'─'*9} {'─'*9} {'─'*9}   ║")

    for i, name in enumerate(scenario_names):
        if i < len(all_gpu_times):
            gpu = all_gpu_times[i]
            dual = all_dual_times[i]
            speedup = gpu / dual
            marker = "✅" if speedup > 1.1 else "──"
            print(f"║  {name:<30} {gpu/1000:>8.1f}s {dual/1000:>8.1f}s {speedup:>8.2f}x {marker} ║")

    total_gpu = sum(all_gpu_times)
    total_dual = sum(all_dual_times)
    total_speedup = total_gpu / total_dual

    print(f"║  {'─'*30} {'─'*9} {'─'*9} {'─'*9}   ║")
    print(f"║  {'TOTAL':<30} {total_gpu/1000:>8.1f}s {total_dual/1000:>8.1f}s {total_speedup:>8.2f}x    ║")
    print("║                                                                    ║")
    print("║  Zero-cost routing. ANE only fires in parallel with GPU.            ║")
    print("║  Same hardware. Two compute paths. Faster workflows.                ║")
    print("║                                                                    ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")


if __name__ == "__main__":
    main()
