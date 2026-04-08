"""
Phase 3B: Query-type classifier. CPU keyword-based, <1ms.

Types: FACTUAL, ANALYTICAL, CREATIVE, STATUS, RESEARCH, DEBUGGING, CASUAL
Each type gets a mode instruction appended to the system prompt.
"""

import re

PATTERNS = {
    "factual": [
        r'\bwhat is\b', r'\bhow many\b', r'\bwhen did\b', r'\bwho is\b',
        r'\bhow much\b', r'\bwhat was\b', r'\bdefine\b', r'\bwhat does\b',
        r'\btell me about\b', r'\bwhat are the\b',
    ],
    "analytical": [
        r'\bwhy does\b', r'\bwhy did\b', r'\bwhat do you think\b',
        r'\banalyze\b', r'\bcompare\b', r'\bexplain why\b',
        r'\bwhat caused\b', r'\bwhat would happen\b', r'\bhow does .* work\b',
        r'\bbreak down\b', r'\bwhat.s the relationship\b',
    ],
    "creative": [
        r'\bwrite\b', r'\bdraft\b', r'\bcreate\b', r'\bdesign\b',
        r'\bbuild\b', r'\bhelp me with\b', r'\bcompose\b', r'\bgenerate\b',
    ],
    "status": [
        r'\bhow.s it going\b', r'\bhow is it going\b',
        r'\bwhere are we\b', r'\bwhat.s next\b',
        r'\bstatus\b', r'\bupdate\b', r'\bcatch me up\b',
        r'\bwhat.s new\b', r'\bwhat are we working on\b',
        r'\bwhat did i miss\b', r'\bproject\b',
    ],
    "research": [
        r'\bsearch\b', r'\bfind\b', r'\blatest\b', r'\blook up\b',
        r'\bcheck\b.*\b(web|x|twitter|reddit)\b',
        r'\bcurrent\b.*\bnews\b', r'\brecent\b',
    ],
    "debugging": [
        r'\bnot working\b', r'\bbroken\b', r'\berror\b', r'\bbug\b',
        r'\bfailed\b', r'\bcrash', r'\bwhy isn.t\b', r'\bfixing\b',
        r'\bdiagnose\b', r'\btroubleshoot\b', r'\bstale\b',
        r'\bwrong\b.*\b(data|result|answer|output)\b',
        r'\breturning\b.*\b(wrong|stale|old)\b',
    ],
}

MODE_INSTRUCTIONS = {
    "factual": "Direct answer. Cite specific numbers from briefing and memories. One paragraph max.",
    "analytical": "Structure: observation, interpretation, challenges to interpretation, next steps. Reference data.",
    "creative": "Focus on the deliverable. Don't explain the plan, just execute it.",
    "status": "Read the briefing. Current state + top 3 priorities. Be specific.",
    "research": "Use tools first, synthesize after. Don't answer from memory if current information is needed.",
    "debugging": "Step through possible causes. For each: how to diagnose, what the fix is.",
    "casual": "Match the energy. Short input = short output.",
}


def classify_query(message):
    """Classify a user message. Returns (type_name, mode_instruction)."""
    lower = message.lower().strip()

    # Very short messages = casual
    if len(lower.split()) <= 3 and not any(
        re.search(p, lower) for ps in PATTERNS.values() for p in ps):
        return "casual", MODE_INSTRUCTIONS["casual"]

    # Check patterns (first match wins)
    for qtype, patterns in PATTERNS.items():
        for p in patterns:
            if re.search(p, lower):
                return qtype, MODE_INSTRUCTIONS[qtype]

    # Default: if it ends with '?', factual. Otherwise casual.
    if lower.endswith('?'):
        return "factual", MODE_INSTRUCTIONS["factual"]

    return "casual", MODE_INSTRUCTIONS["casual"]
