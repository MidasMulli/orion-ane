#!/usr/bin/env python3
"""Quick test: what does the distilled model generate vs pretrained?"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "speculative"))

from real_draft import RealDraftModel
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--weights", default=None)
args = parser.parse_args()

label = "distilled" if args.weights else "pretrained"
print(f"Testing {label} model...")

draft = RealDraftModel()
draft.load_and_compile(lambda msg: print(f"  {msg}"), fused=True, weights_path=args.weights)

if not draft.compiled:
    print("FAILED to compile!")
    sys.exit(1)

prompts = [
    "The capital of France is",
    "Machine learning is",
    "1 + 1 =",
]

for prompt in prompts:
    ids = draft.encode(prompt)
    draft.prefill(ids)
    tokens = draft.draft_continue(20)
    text = draft.decode([t[0] for t in tokens])
    print(f"\n  Prompt: {prompt}")
    print(f"  Output: {text}")
