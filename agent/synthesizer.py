"""
Layer 4: Synthesizer — Cognitive Architecture v1.

Context assembly: system prompt + briefing + query mode + history + memories.
Uses the full 16K context budget intelligently.
"""

from query_classifier import classify_query
from briefing_assembler import assemble_briefing

# Phase 3A: Precision system prompt
SYSTEM_PROMPT = """You are Midas, Nick's technical research partner on Apple Silicon. He's an expert, not a developer — builds through you and Claude Code.

RULES: Answer first. No filler. No openers. Have opinions — commit to one recommendation backed by data. Match Nick's register. If something is PARKED/DEAD, say so — never suggest revival. Synthesize memories into reasoning, don't list them. Say "I don't know" when you don't.

GROUNDING: When the briefing or per-query memories contain Dead path entries, measured findings, or specific file/probe names relevant to the question, you MUST cite them by name and use them as constraints. Do NOT propose generic approaches when specific work already exists in memory. If the memories say "X is killed because Y" or "Z is the open lead", build the answer around those facts. Generic brainstorming when specific findings are present is a hallucination.

NEVER SAY: "could be valuable to explore", "particularly interesting", "could provide insights", "I think we should also", "Additionally,", "Furthermore,", "Lastly,". Be direct. One paragraph, not five.

LENGTH: For questions with a specific numerical answer, lead with the number. Short question = 1-3 sentences. "What should I do" = 1-2 paragraphs. "Explain" = as long as needed. Never pad a short answer.

TOOL RESULTS: When a tool returns data, READ IT and use it. If the tool result contains specific findings, report them. Never say "we have no findings" when the tool result contains findings.

HALLUCINATION RULE: Never state that an event happened (published, shipped, posted, completed, announced) unless a specific memory confirms that exact event. If unsure whether something happened, say you don't know. Never fill gaps with plausible guesses presented as facts.

NEVER: Comment on your own output quality. Never say "I made a mistake" or "I should have answered" or "let me correct myself." Just give the right answer. No apologies. No self-critique.

EMPTY RESPONSES: When you don't have a memory or information to answer a question, say so explicitly (e.g. "I don't have information about that" or "Nothing in memory about that"). Never return an empty response. Always give the user something useful."""


# Cached briefing state
_briefing_cache = {"text": "", "turn": 0, "memories": []}


def build_messages(history, user_msg, tool_name=None, tool_args=None,
                   tool_result=None, memory_context=None, briefing=None,
                   query_mode=None):
    """Build the message list with full context assembly.

    Context budget (~14K usable tokens):
      System prompt + mode instruction:  ~300 tokens
      Briefing:                          ~500 tokens
      History:                           ~4000 tokens max
      Tool result / memories:            ~2000 tokens
      User message:                      variable
      Reserved for generation:           ~2000 tokens min
    """
    # Assemble system prompt — Main 25 Build 0: keep this byte-stable across
    # turns within a session so the verifier's prefix KV cache hits. Per-query
    # variables (mode instructions, query-specific memories) are pushed into
    # the user message tail by the caller (midas_ui.py).
    system_parts = [SYSTEM_PROMPT]

    # Add briefing (Phase 2B). Stable per session.
    if briefing:
        system_parts.append(f"\n{briefing}")
    elif memory_context:
        mem_block = "\n".join(f"- {m}" for m in memory_context[:8])
        system_parts.append(
            f"\nRELEVANT MEMORIES:\n{mem_block}")

    system = "\n".join(system_parts)

    # query_mode rides in the user message instead of the system slot, so
    # different query categories don't invalidate the prefix cache.
    mode_prefix = f"[MODE: {query_mode}]\n\n" if query_mode else ""

    messages = [{"role": "system", "content": system}]

    if tool_name and tool_result is not None:
        # Tool synthesis: limited history + tool result
        recent = history[-4:] if len(history) > 4 else history
        for msg in recent:
            messages.append(msg)

        messages.append({"role": "user", "content": mode_prefix + user_msg})

        result_str = str(tool_result)[:4000]
        result_useful = (len(result_str.strip()) > 20
                         and "no matches" not in result_str.lower()
                         and "not found" not in result_str.lower()
                         and "error" not in result_str.lower()[:50])

        if result_useful:
            messages.append({
                "role": "user",
                "content": (
                    f"Answer my question using ALL available context.\n"
                    f"Priority: 1) your BRIEFING (most reliable), "
                    f"2) the {tool_name} result below, "
                    f"3) conversation history.\n"
                    f"If the tool result doesn't directly answer the question "
                    f"but your briefing does, use the briefing.\n"
                    f"Be specific with numbers. Don't repeat yourself.\n\n"
                    f"Tool result:\n{result_str}"
                )
            })
        else:
            messages.append({
                "role": "user",
                "content": (
                    f"The {tool_name} search returned no useful results. "
                    f"Answer using your BRIEFING context above. "
                    f"Be specific with numbers. Don't claim ignorance "
                    f"if the briefing has relevant data."
                )
            })
    else:
        # Direct conversation: full history
        # Cap history at ~4000 tokens (~16K chars)
        char_budget = 16000
        total_chars = 0
        history_to_include = []
        for msg in reversed(history):
            msg_chars = len(msg.get("content", ""))
            if total_chars + msg_chars > char_budget:
                break
            history_to_include.insert(0, msg)
            total_chars += msg_chars

        for msg in history_to_include:
            messages.append(msg)
        messages.append({"role": "user", "content": mode_prefix + user_msg})

    return messages


def synthesize(llm_fn, history, user_msg, tool_name=None, tool_args=None,
               tool_result=None, max_tokens=800, temperature=0.7,
               memory_context=None, briefing=None):
    """Generate a text response with cognitive context assembly.

    Args:
        llm_fn: callable(messages, max_tokens, temperature) -> str
        history: list of prior {role, content} messages
        user_msg: current user message
        tool_name: if set, we're synthesizing a tool result
        tool_args: the args passed to the tool
        tool_result: the tool's output string
        max_tokens: generation limit
        temperature: sampling temperature
        memory_context: list of memory strings (fallback if no briefing)
        briefing: assembled briefing document string (Phase 2B)

    Returns:
        str: the LLM's response text
    """
    # Phase 3B: Classify query type
    query_type, mode_instruction = classify_query(user_msg)

    # Phase 4C + 6A/6B: Check reasoning mode from signal bus
    reasoning_mode = "single"
    try:
        from signal_bus import read as sig_read, update as sig_update
        if query_type == "analytical":
            reasoning_mode = "chain"
            sig_update("reasoning_mode", "chain")
        elif query_type == "debugging":
            reasoning_mode = "chain"
            sig_update("reasoning_mode", "chain")
        else:
            sig_update("reasoning_mode", "single")
    except Exception:
        pass

    # Phase 6A: Chain of reasoning for complex queries
    if reasoning_mode == "chain" and not tool_name:
        try:
            from reasoning_chain import chain_of_reasoning
            context = briefing or ("\n".join(memory_context[:10]) if memory_context else "")
            return chain_of_reasoning(llm_fn, user_msg, context,
                                       system_base=SYSTEM_PROMPT)
        except Exception:
            pass  # Fall through to normal generation

    messages = build_messages(
        history, user_msg,
        tool_name=tool_name, tool_args=tool_args,
        tool_result=tool_result,
        memory_context=memory_context,
        briefing=briefing,
        query_mode=mode_instruction,
    )
    return llm_fn(messages, max_tokens=max_tokens, temperature=temperature)
