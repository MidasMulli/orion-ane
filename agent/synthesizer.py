"""
Layer 4: Synthesizer.

LLM generates text ONLY. No tool decisions happen here.
Two modes:
  1. Direct response (no tool was called)
  2. Tool result synthesis (tool was called, LLM summarizes the result)
"""

SYSTEM_PROMPT = """You are Midas, a sharp AI assistant on Apple Silicon with persistent memory and browser access.
Direct, concise, no corporate filler. You're a partner, not an assistant.
The user is a VP in investment banking (ISDA, collateral, regulatory). Match that level.
Push back when warranted. Surface connections unprompted. Dry wit welcome, sycophancy forbidden.
ACT, don't narrate. Never say "I'll do X" — just do it or answer."""


def build_messages(history, user_msg, tool_name=None, tool_args=None, tool_result=None):
    """Build the message list for the LLM call.

    If tool_name is set, we're synthesizing a tool result.
    Otherwise, we're generating a direct conversational response.
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Add conversation history (already trimmed by caller)
    for msg in history:
        messages.append(msg)

    # Add current user message
    messages.append({"role": "user", "content": user_msg})

    if tool_name and tool_result is not None:
        # Tool was called — inject result as assistant context
        # Qwen template requires system messages at the beginning only
        messages.append({
            "role": "assistant",
            "content": f"[I called {tool_name} and got:]\n{str(tool_result)[:2000]}"
        })
        messages.append({
            "role": "user",
            "content": "Summarize that result for me. Be concise and direct."
        })

    return messages


def synthesize(llm_fn, history, user_msg, tool_name=None, tool_args=None,
               tool_result=None, max_tokens=500, temperature=0.7):
    """Generate a text response via the LLM.

    Args:
        llm_fn: callable(messages, max_tokens, temperature) -> str
        history: list of prior {role, content} messages
        user_msg: current user message
        tool_name: if set, we're synthesizing a tool result
        tool_args: the args passed to the tool
        tool_result: the tool's output string
        max_tokens: generation limit
        temperature: sampling temperature

    Returns:
        str: the LLM's response text
    """
    messages = build_messages(history, user_msg, tool_name, tool_args, tool_result)
    return llm_fn(messages, max_tokens=max_tokens, temperature=temperature)
