"""
Phase 6A+6B: Multi-step reasoning and self-verification.

6A: Chain of reasoning for complex analytical queries
    Decompose → Reason → Synthesize (3 LLM calls)

6B: Self-verification for uncertain domains
    Generate → Review → Correct if needed (2-3 LLM calls)

Triggered by signal_bus reasoning_mode:
  "single" (default) → normal single-pass generation
  "chain" → decompose + reason + synthesize
  "verify" → generate + self-review + optional correction
"""


def chain_of_reasoning(llm_fn, query, context, system_base=""):
    """Multi-step reasoning for analytical queries.

    Cost: 3x generation time. Use only for analytical/debugging queries.

    Returns: synthesized response string
    """
    # Step 1: Decompose (max 200 tokens)
    decompose_prompt = [
        {"role": "system", "content":
         system_base + "\nBreak this into 2-3 specific sub-questions. "
         "Output only the numbered sub-questions. Be precise."},
        {"role": "user", "content": query},
    ]
    if context:
        decompose_prompt.append({"role": "user", "content": f"Context:\n{context[:2000]}"})

    decomposition = llm_fn(decompose_prompt, max_tokens=200, temperature=0.1)

    # Step 2: Reason (max 600 tokens)
    reason_prompt = [
        {"role": "system", "content":
         system_base + "\nAnswer each sub-question with specific reasoning. "
         "Reference numbers from context. Show your work."},
        {"role": "user", "content": f"Sub-questions:\n{decomposition}"},
    ]
    if context:
        reason_prompt.append({"role": "user", "content": f"Context:\n{context[:2000]}"})

    reasoning = llm_fn(reason_prompt, max_tokens=600, temperature=0.3)

    # Step 3: Synthesize (max 400 tokens)
    synth_prompt = [
        {"role": "system", "content":
         system_base + "\nSynthesize a clear final response from the reasoning below. "
         "Note contradictions. Be direct. Don't repeat the sub-questions."},
        {"role": "user", "content":
         f"Original question: {query}\n\nReasoning:\n{reasoning}"},
    ]

    synthesis = llm_fn(synth_prompt, max_tokens=400, temperature=0.3)

    return synthesis


def verified_generation(llm_fn, query, context, system_base=""):
    """Generate then self-verify for uncertain domains.

    Cost: 2-3x generation time.

    Returns: verified (or corrected) response string
    """
    # Step 1: Generate
    gen_prompt = [
        {"role": "system", "content": system_base},
        {"role": "user", "content": query},
    ]
    if context:
        gen_prompt.append({"role": "user", "content": f"Context:\n{context[:2000]}"})

    response = llm_fn(gen_prompt, max_tokens=600, temperature=0.3)

    # Step 2: Self-review
    review_prompt = [
        {"role": "system", "content":
         "Review this response for: "
         "1) factual errors 2) unsupported claims 3) logical inconsistencies. "
         "If the response is sound, say VERIFIED. "
         "If there are issues, list them specifically."},
        {"role": "user", "content":
         f"Question: {query}\n\nResponse to review:\n{response}"},
    ]

    review = llm_fn(review_prompt, max_tokens=300, temperature=0.1)

    if "VERIFIED" in review.upper():
        return response

    # Step 3: Correct
    correct_prompt = [
        {"role": "system", "content":
         system_base + "\nProvide a corrected response addressing the issues below. "
         "Be specific and accurate."},
        {"role": "user", "content":
         f"Question: {query}\n\nIssues found:\n{review}"},
    ]
    if context:
        correct_prompt.append({"role": "user", "content": f"Context:\n{context[:2000]}"})

    corrected = llm_fn(correct_prompt, max_tokens=600, temperature=0.3)

    return corrected
