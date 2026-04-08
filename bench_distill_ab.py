#!/usr/bin/env python3
"""
A/B Benchmark: Pretrained vs Distilled 0.6B Draft Model
========================================================

Runs each test in a separate subprocess to avoid ANE kernel slot exhaustion.

Usage:
  ~/.mlx-env/bin/python3 bench_distill_ab.py
  ~/.mlx-env/bin/python3 bench_distill_ab.py --distilled-weights distill/distilled_qwen3_06b_4b_teacher/
"""

import subprocess
import sys
import os
import json
import time

PYTHON = os.path.expanduser("~/.mlx-env/bin/python3")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def run_single_test(weights_path=None, label="test"):
    """Run a single acceptance rate test in a subprocess."""
    worker = os.path.join(SCRIPT_DIR, "_bench_worker.py")
    cmd = [PYTHON, worker]
    if weights_path:
        cmd.extend(["--weights", weights_path])

    print(f"\n▸ Running {label}...")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600,
                           cwd=SCRIPT_DIR)

    if result.returncode != 0:
        print(f"  ERROR: Worker failed!")
        print(result.stderr[-2000:] if result.stderr else "no stderr")
        return None

    # Parse JSON result from last line
    lines = result.stdout.strip().split('\n')
    # Print worker output (except last JSON line)
    for line in lines[:-1]:
        print(f"  {line}")

    try:
        data = json.loads(lines[-1])
    except (json.JSONDecodeError, IndexError):
        print(f"  ERROR: Could not parse worker output")
        print(result.stdout[-1000:])
        return None

    elapsed = time.time() - t0
    print(f"  Completed in {elapsed:.0f}s")
    return data


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--distilled-weights",
                       default="distill/distilled_qwen3_06b_4b_teacher/",
                       help="Path to distilled weights safetensors")
    args = parser.parse_args()

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  DISTILLATION A/B BENCHMARK                                 ║")
    print("║  Pretrained vs Distilled 0.6B — Acceptance Rate Test        ║")
    print("║  Each test runs in separate process (ANE kernel isolation)  ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    # Test A: Pretrained baseline
    pre = run_single_test(weights_path=None, label="PRETRAINED (Baseline)")

    # Test B: Distilled
    dis = run_single_test(weights_path=args.distilled_weights,
                         label="DISTILLED (4B Teacher)")

    if not pre or not dis:
        print("\n  ERROR: One or both tests failed. Cannot compare.")
        return

    # Final comparison
    avg_pre = pre['avg_acceptance']
    avg_dis = dis['avg_acceptance']
    tpr_pre = pre['avg_tokens_per_round']
    tpr_dis = dis['avg_tokens_per_round']

    print(f"\n╔══════════════════════════════════════════════════════════════╗")
    print(f"║  FINAL RESULTS                                               ║")
    print(f"╠══════════════════════════════════════════════════════════════╣")
    print(f"║                    Pretrained    Distilled    Delta          ║")
    print(f"║  Acceptance:       {avg_pre:5.1%}         {avg_dis:5.1%}      {avg_dis-avg_pre:+5.1%}         ║")
    print(f"║  Tokens/round:     {tpr_pre:5.1f}         {tpr_dis:5.1f}      {tpr_dis-tpr_pre:+5.1f}         ║")
    print(f"╠══════════════════════════════════════════════════════════════╣")

    if avg_dis > avg_pre:
        improve = (avg_dis - avg_pre) / avg_pre * 100
        print(f"║  → Distillation improved acceptance by {improve:.0f}%!               ║")
    elif avg_dis < avg_pre:
        degrade = (avg_pre - avg_dis) / avg_pre * 100
        print(f"║  → Distillation degraded acceptance by {degrade:.0f}%                ║")
    else:
        print(f"║  → No change                                                 ║")

    print(f"╚══════════════════════════════════════════════════════════════╝")

    # Per-prompt breakdown
    print(f"\n  Per-prompt breakdown:")
    print(f"  {'Prompt':>50} {'Pretrained':>10} {'Distilled':>10} {'Delta':>8}")
    print(f"  {'─'*50} {'─'*10} {'─'*10} {'─'*8}")
    for i, (p, d) in enumerate(zip(pre['prompts'], dis['prompts'])):
        pre_rate = p['acceptance_rate']
        dis_rate = d['acceptance_rate']
        delta = dis_rate - pre_rate
        prompt = p['prompt'][:50]
        print(f"  {prompt:>50} {pre_rate:9.1%} {dis_rate:9.1%} {delta:+7.1%}")


if __name__ == "__main__":
    main()
