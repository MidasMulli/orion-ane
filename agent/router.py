"""
Deterministic intent routing. No LLM calls in Layer 1.
Layer 2 uses LLM only for single-word classification.
"""

import re

# ── Helpers ──────────────────────────────────────────────────────────────────

def strip_greeting(text):
    """Remove hi/hey/hello from start of message."""
    return re.sub(r'^(hi|hey|hello|yo|sup|good morning|good afternoon)\s*[,.]?\s*',
                  '', text, flags=re.I).strip()

def extract_query(msg):
    """Pull the actual question/query from a message."""
    msg = strip_greeting(msg)
    msg = re.sub(r'^(can you |could you |please |would you |i need you to )',
                 '', msg, flags=re.I)
    msg = re.sub(r'^(tell me |show me |find me |get me |look up |search for )',
                 '', msg, flags=re.I)
    return msg.strip() or msg

def extract_vault_path(msg):
    """Match vault file references."""
    lower = msg.lower()
    if 'roadmap' in lower: return 'Roadmap.md'
    if 'decision log' in lower or 'decision' in lower: return 'Decision Log.md'
    if 'infrastructure' in lower: return 'Infrastructure Map.md'
    if 'home' in lower: return 'HOME.md'
    match = re.search(r'([\w/-]+\.md)', msg)
    return match.group(1) if match else ''

def extract_command(msg):
    """Pull shell command from message."""
    match = re.search(r'`(.+?)`', msg)
    if match: return match.group(1)
    match = re.search(r'(?:run|execute)\s+(.+)', msg, re.I)
    if match: return match.group(1).strip()
    return msg

def extract_search_query(msg):
    """Pull the search query, stripping routing phrases."""
    q = strip_greeting(msg)
    q = re.sub(r'^(search|google|look up|find|search for|search google for)\s+',
               '', q, flags=re.I)
    q = re.sub(r'^(what is|what are|who is|where is|how is|how do|when did)\s+',
               '', q, flags=re.I)
    return q.strip() or extract_query(msg)


# ── Layer 1: Keyword Patterns ────────────────────────────────────────────────

PATTERNS = [
    # Memory store
    {
        'keywords': ['remember this', 'remember that', 'save this', 'note that',
                     'store this', "don't forget", 'keep in mind', 'store in memory'],
        'tool': 'memory_ingest',
        'extract': lambda msg: msg,
        'args': lambda msg: {'role': 'user', 'text': msg},
    },
    # Memory recall
    {
        'keywords': ['do you remember', 'what did i say', 'recall ',
                     'what do you know about', 'have i mentioned',
                     'search memory', 'check memory', 'in memory'],
        'tool': 'memory_recall',
        'extract': lambda msg: extract_query(msg),
        'args': lambda msg: {'query': extract_query(msg)},
    },
    # Memory stats
    {
        'keywords': ['how many memories', 'memory stats', 'memory count',
                     'how much do you remember', 'in your memory',
                     'how many facts'],
        'tool': 'memory_stats',
        'extract': lambda msg: None,
        'args': lambda msg: {},
    },
    # Memory insights
    {
        'keywords': ['memory insights', 'enricher insights', 'entity patterns',
                     'relationship graph', 'what patterns'],
        'tool': 'memory_insights',
        'extract': lambda msg: None,
        'args': lambda msg: {},
    },
    # Vault insight (cross-reference — must be before vault_read)
    {
        'keywords': ['cross-reference', 'cross reference', 'vault insight',
                     'vault and memory', 'cross-ref'],
        'tool': 'vault_insight',
        'extract': lambda msg: extract_query(msg),
        'args': lambda msg: {'topic': extract_query(msg)},
    },
    # Vault read
    {
        'keywords': ['check vault', 'read vault', 'vault read', 'from the vault',
                     'in the vault', 'read the roadmap', 'read the decision',
                     'infrastructure map', 'check roadmap', 'use vault_read',
                     'check the vault', 'the vault'],
        'tool': 'vault_read',
        'extract': lambda msg: extract_vault_path(msg),
        'args': lambda msg: {'path': extract_vault_path(msg)},
    },
    # Scanner
    {
        'keywords': ['scan candidates', 'clear candidates', 'process scans',
                     'scan digest', 'check scans', 'any scans', 'new scans',
                     'unreviewed scans', 'scanner stats'],
        'tool': 'scan_digest',
        'extract': lambda msg: None,
        'args': lambda msg: {
            'mode': 'clear' if any(w in msg.lower() for w in ['clear', 'process']) else
                    'unreviewed' if 'unreviewed' in msg.lower() else
                    'stats' if 'stats' in msg.lower() else 'latest'
        },
    },
    # Shell
    {
        'keywords': ['run command', 'execute command', 'shell command',
                     'run `', 'execute `', 'use shell', 'run:'],
        'tool': 'shell',
        'extract': lambda msg: extract_command(msg),
        'args': lambda msg: {'command': extract_command(msg)},
    },
    # Search (web) — must be after more specific patterns
    {
        'keywords': ['search for', 'search google', 'google ', 'look up',
                     'find out about', 'latest news on', 'current price of',
                     'search the web', 'look online'],
        'tool': 'browse_search',
        'extract': lambda msg: extract_search_query(msg),
        'args': lambda msg: {'query': extract_search_query(msg)},
    },
    # X feed
    {
        'keywords': ['x feed', 'twitter feed', 'scan x', 'check x',
                     'check twitter', 'what\'s on x', 'scan my feed',
                     'what\'s on twitter', 'on twitter'],
        'tool': 'browse_x_feed',
        'extract': lambda msg: None,
        'args': lambda msg: {'count': 5},
    },
    # Playbook
    {
        'keywords': ['playbook', 'update playbook', 'read playbook',
                     'self-knowledge', 'improvement queue'],
        'tool': 'playbook_update',
        'extract': lambda msg: None,
        'args': lambda msg: {'section': 'full', 'action': 'read'},
    },
    # Message Claude
    {
        'keywords': ['message claude', 'tell claude', 'leave claude a note',
                     'claude inbox'],
        'tool': 'message_claude',
        'extract': lambda msg: msg,
        'args': lambda msg: {'message': msg, 'priority': 'medium'},
    },
]


def layer1_route(message):
    """
    Keyword-based routing. Returns (tool_name, args_dict) or None.
    Zero model calls. Instant.
    """
    lower = message.lower()
    lower_stripped = strip_greeting(lower)

    for pattern in PATTERNS:
        if any(kw in lower_stripped for kw in pattern['keywords']):
            args = pattern['args'](message)
            return (pattern['tool'], args)

    return None


def layer2_classify(message, llm_fn):
    """
    LLM-based intent classification. Single word output.
    Only called when Layer 1 has no match.
    """
    prompt = (
        "Classify this message into ONE category. Reply with ONLY the category name.\n"
        "SEARCH = needs live web data (news, prices, current events)\n"
        "MEMORY_STORE = user wants to save/remember something\n"
        "MEMORY_RECALL = user asks about past conversations\n"
        "VAULT = user asks about project docs/roadmap\n"
        "CONVERSATION = general questions, definitions, analysis, opinions\n"
        f"\nMessage: \"{message}\"\n"
        "Category:"
    )

    result = llm_fn(prompt, max_tokens=8, temperature=0.0)
    category = result.strip().upper().split()[0] if result.strip() else "CONVERSATION"

    valid = ('SEARCH', 'MEMORY_STORE', 'MEMORY_RECALL', 'VAULT', 'CONVERSATION')
    if category not in valid:
        return 'CONVERSATION'

    return category


def route(message, llm_fn=None):
    """
    Main routing function.
    Returns (tool_name, args_dict) or ('conversation', {}).
    """
    # Layer 1: deterministic
    result = layer1_route(message)
    if result:
        return result

    # Layer 2: lightweight LLM classification (if llm available)
    if llm_fn:
        category = layer2_classify(message, llm_fn)

        if category == 'SEARCH':
            q = extract_search_query(message)
            return ('browse_search', {'query': q})
        elif category == 'MEMORY_STORE':
            return ('memory_ingest', {'role': 'user', 'text': message})
        elif category == 'MEMORY_RECALL':
            return ('memory_recall', {'query': extract_query(message)})
        elif category == 'VAULT':
            return ('vault_read', {'path': extract_vault_path(message)})

    return ('conversation', {})
