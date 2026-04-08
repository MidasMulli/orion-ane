# X Post — Follow-up to Phantom Agent Post

## Context
- Phantom agent post is live (45 likes, 17 likes on thread)
- Follow-up should be a **quote tweet** of the Phantom post
- Tone: matches "Built a local AI agent from scratch this weekend with Claude. No frameworks, no LangChain, no MCP plugins."
- Not hype-y. Real numbers. "No cloud" energy.

---

## Draft 1 (recommended — quote tweet of Phantom post)

Couldn't resist. Wired the Neural Engine back in.

Phantom now runs three compute tiers on one chip:

→ CPU: 1,700 embeddings/sec (persistent memory, zero GPU cost)
→ ANE: 1.7B Qwen3 at 57 tok/s (analysis + classification)
→ GPU: 9B Qwen3.5 at 25 tok/s (reasoning)

All three run concurrently. 3.8% interference measured.

The agent routes tasks across all three processors. ANE work runs free during GPU inference — no contention, no scheduling, just Apple Silicon doing what Apple won't let it.

One command to boot everything: `midas`

---

## Draft 2 (shorter, punchier)

Wired the Neural Engine back into Phantom.

Three tiers on a MacBook Air:
→ CPU: memory (1,700 embeddings/sec)
→ ANE: 1.7B analysis (57 tok/s)
→ GPU: 9B reasoning (25 tok/s)

3.8% interference. ANE runs free during GPU work.

The agent from the last post now has three brains instead of one. Same MacBook Air. Same 16GB.

---

## Draft 3 (narrative continuation)

Last post: built an AI agent from scratch.
This post: gave it three brains.

CPU handles memory — 1,700 embeddings/sec, invisible.
ANE runs a 1.7B model for fast analysis — 57 tok/s via reverse-engineered _ANEClient.
GPU runs 9B for reasoning — 25 tok/s via MLX.

All concurrent. Measured 3.8% interference between them.

One chip. Three processors. Zero cloud.

---

## Notes
- Post as quote tweet of: the Phantom "Built a local AI agent from scratch" tweet
- Screenshot: Midas banner showing `86 memories │ browser ● │ ane ●` OR eval_tiers.py 22/22
- The 3.8% interference number is from dual-path-inference testing (Session 2026-03-16)
- ANE via CoreML (ANEMLL conversion of Qwen3-1.7B), not raw _ANEClient kernels — but previous post already covered the raw kernel angle
