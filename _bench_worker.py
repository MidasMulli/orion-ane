#!/usr/bin/env python3
"""Worker for A/B benchmark — runs in subprocess to isolate ANE kernel slots."""

import sys
import os
import time
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "speculative"))

from real_draft import RealDraftModel
from mlx_local_verifier import MLXLocalVerifier
import mlx.core as mx

K_DRAFT = 5
MAX_TOKENS = 50
VERIFIER_MODEL = "mlx-community/Qwen3-4B-Instruct-2507-4bit"

PROMPTS = [
    "The capital of France is",
    "In quantum mechanics, the uncertainty principle states that",
    "The process of photosynthesis converts",
    "Machine learning algorithms can be broadly classified into",
    "The Treaty of Westphalia in 1648",
]


def speculative_generate(draft_model, verifier, prompt, max_tokens, k_draft):
    prompt_ids = draft_model.encode(prompt)
    generated_ids = []
    total_ane_ms = 0
    total_gpu_ms = 0
    total_accepted = 0
    total_rejected = 0
    n_rounds = 0

    t0 = time.time()
    draft_model.prefill(prompt_ids)
    total_ane_ms += (time.time() - t0) * 1000

    t0 = time.time()
    verifier.reset_cache()
    x = mx.array([prompt_ids])
    v_logits = verifier.model(x, cache=verifier.cache)
    mx.eval(v_logits)
    total_gpu_ms += (time.time() - t0) * 1000

    while len(generated_ids) < max_tokens:
        n_rounds += 1
        k = min(k_draft, max_tokens - len(generated_ids))

        t0 = time.time()
        drafts = draft_model.draft_continue(k)
        total_ane_ms += (time.time() - t0) * 1000
        draft_ids = [d[0] for d in drafts]

        t0 = time.time()
        v_tok_0 = mx.argmax(v_logits[0, -1, :]).item()
        x = mx.array([draft_ids])
        v_logits_batch = verifier.model(x, cache=verifier.cache)
        mx.eval(v_logits_batch)
        v_toks = mx.argmax(v_logits_batch[0], axis=-1).tolist()
        gpu_ms = (time.time() - t0) * 1000

        accepted = 0
        if v_tok_0 == draft_ids[0]:
            accepted = 1
            for i in range(1, k):
                if v_toks[i - 1] == draft_ids[i]:
                    accepted += 1
                else:
                    break

        if accepted == k:
            generated_ids.extend(draft_ids)
            bonus_tok = v_toks[k - 1]
            generated_ids.append(bonus_tok)
            total_accepted += k
            bonus_pos = len(prompt_ids) + len(generated_ids) - 1
            draft_model.feed_tokens([bonus_tok], bonus_pos)
            x = mx.array([[bonus_tok]])
            v_logits = verifier.model(x, cache=verifier.cache)
            mx.eval(v_logits)
        else:
            if accepted > 0:
                generated_ids.extend(draft_ids[:accepted])
            total_accepted += accepted
            total_rejected += 1
            correction_tok = v_tok_0 if accepted == 0 else v_toks[accepted - 1]
            generated_ids.append(correction_tok)

            t_rp = time.time()
            all_ids = list(prompt_ids) + list(generated_ids)
            verifier.reset_cache()
            x = mx.array([all_ids])
            v_logits = verifier.model(x, cache=verifier.cache)
            mx.eval(v_logits)
            gpu_ms += (time.time() - t_rp) * 1000

            rollback_pos = len(prompt_ids) + len(generated_ids) - 1
            draft_model.feed_tokens([correction_tok], rollback_pos)

        total_gpu_ms += gpu_ms

    acceptance_rate = total_accepted / (total_accepted + total_rejected) \
        if (total_accepted + total_rejected) > 0 else 0

    return {
        "tokens": len(generated_ids),
        "total_ms": total_ane_ms + total_gpu_ms,
        "n_rounds": n_rounds,
        "total_accepted": total_accepted,
        "total_rejected": total_rejected,
        "acceptance_rate": acceptance_rate,
        "tokens_per_round": len(generated_ids) / n_rounds if n_rounds > 0 else 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=None, help="Custom weights path (None=pretrained)")
    args = parser.parse_args()

    label = "distilled" if args.weights else "pretrained"

    # Load verifier
    print(f"Loading verifier...", file=sys.stderr)
    verifier = MLXLocalVerifier(model_path=VERIFIER_MODEL)
    verifier.load(lambda msg: print(f"  {msg}", file=sys.stderr))

    # Load draft model
    print(f"Loading {label} draft model...", file=sys.stderr)
    draft = RealDraftModel()
    draft.load_and_compile(lambda msg: print(f"  {msg}", file=sys.stderr),
                          fused=True, weights_path=args.weights)

    if not draft.compiled:
        print(f"ANE compilation failed!", file=sys.stderr)
        sys.exit(1)

    # Run benchmarks
    results = []
    for i, prompt in enumerate(PROMPTS):
        spec = speculative_generate(draft, verifier, prompt, MAX_TOKENS, K_DRAFT)
        spec['prompt'] = prompt
        results.append(spec)
        # Human-readable to stdout
        print(f"  Prompt {i+1}/{len(PROMPTS)}: Acceptance={spec['acceptance_rate']:.1%} "
              f"({spec['total_accepted']}/{spec['total_accepted']+spec['total_rejected']}) "
              f"Tok/rnd={spec['tokens_per_round']:.1f}")

    avg_accept = float(np.mean([r['acceptance_rate'] for r in results]))
    avg_tpr = float(np.mean([r['tokens_per_round'] for r in results]))

    print(f"  Average: Acceptance={avg_accept:.1%}  Tok/rnd={avg_tpr:.1f}")

    # Final line: JSON for parent process
    output = {
        "label": label,
        "avg_acceptance": avg_accept,
        "avg_tokens_per_round": avg_tpr,
        "prompts": results,
    }
    print(json.dumps(output))


if __name__ == "__main__":
    main()
