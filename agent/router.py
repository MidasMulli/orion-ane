"""
Deterministic intent routing. No LLM calls in Layer 1.
Layer 2 uses LLM only for single-word classification.
"""

import re

# ── SCGP helpers ────────────────────────────────────────────────────────────

def _extract_entity_name(msg):
    """Extract entity name from message like 'what do we know about Goldman Sachs'."""
    lower = msg.lower()
    for prefix in ['what do we know about', 'build dossier for', 'counterparty dossier for',
                   'counterparty intel on', 'counterparty intelligence on',
                   'look up entity', 'check gleif for', 'lei lookup for', 'search gleif for',
                   'find lei for', 'entity lookup for', 'company dossier for',
                   'counterparty profile for', 'look up', 'check gleif', 'lei lookup',
                   'find lei']:
        if prefix in lower:
            idx = lower.index(prefix) + len(prefix)
            name = msg[idx:].strip().rstrip('?.,!')
            # Strip trailing qualifiers like "on GLEIF", "in EDGAR"
            name = re.sub(r'\s+(?:on|in|from|via)\s+(?:gleif|edgar|sec)\s*$', '', name, flags=re.I)
            if name:
                return name
    # Fallback: take the last quoted string or capitalized words
    quoted = re.findall(r'"([^"]+)"', msg)
    if quoted:
        return quoted[-1]
    # Take trailing proper nouns (capitalized words)
    words = msg.split()
    proper = []
    for w in reversed(words):
        if w[0].isupper() and w.lower() not in ('what', 'who', 'how', 'can', 'the', 'a', 'an', 'is'):
            proper.insert(0, w)
        elif proper:
            break
    return ' '.join(proper) if proper else msg

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
    if any(k in lower for k in ['working on', 'been doing', 'project status',
                                  'current projects', 'what are we building']):
        return 'Roadmap.md'
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
                     'check the vault', 'the vault',
                     'working on', 'been doing', 'project status',
                     'what are we building', 'current projects'],
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
    # Message Claude (before self_test — "message claude: ...stress test" must match here first)
    {
        'keywords': ['message claude', 'tell claude', 'leave claude a note',
                     'claude inbox'],
        'tool': 'message_claude',
        'extract': lambda msg: msg,
        'args': lambda msg: {'message': msg, 'priority': 'medium'},
    },
    # Self-test (first-class tool, not shell)
    {
        'keywords': ['stress test', 'run tests', 'test yourself', 'self test',
                     'self-test', 'light test', 'hardcore test', 'deep test',
                     'check yourself', 'how are you performing', 'run diagnostics'],
        'tool': 'self_test',
        'extract': lambda msg: 'hardcore' if any(w in msg.lower()
                      for w in ['hardcore', 'full', 'deep', 'stress', 'all']) else 'light',
        'args': lambda msg: {'mode': 'hardcore' if any(w in msg.lower()
                      for w in ['hardcore', 'full', 'deep', 'stress', 'all']) else 'light'},
    },
    # Brain snapshot (first-class tool)
    {
        'keywords': ['brain monitor', 'show monitor', 'open monitor',
                     'profiler', 'run profiler', 'show internals',
                     'your brain', 'system snapshot', 'brain snapshot',
                     'show diagnostics', 'how did you route that'],
        'tool': 'brain_snapshot',
        'extract': lambda msg: 'last' if any(w in msg.lower()
                      for w in ['last', 'that', 'previous']) else 'session',
        'args': lambda msg: {'scope': 'last' if any(w in msg.lower()
                      for w in ['last', 'that', 'previous']) else 'session'},
    },
    # Heartbeat dashboard
    {
        'keywords': ['heartbeat', 'open heartbeat', 'launch heartbeat',
                     'open dashboard', 'launch dashboard', 'show dashboard',
                     'system dashboard', 'monitoring dashboard'],
        'tool': 'heartbeat',
        'extract': lambda msg: None,
        'args': lambda msg: {},
    },
    # Self-improve (first-class tool)
    {
        'keywords': ['improve yourself', 'optimize yourself', 'run improver',
                     'check for improvements', 'any improvements'],
        'tool': 'self_improve',
        'extract': lambda msg: 'analyze',
        'args': lambda msg: {'mode': 'analyze'},
    },
    # Shell
    {
        'keywords': ['run command', 'execute command', 'shell command',
                     'run `', 'execute `', 'use shell', 'run:'],
        'tool': 'shell',
        'extract': lambda msg: extract_command(msg),
        'args': lambda msg: {'command': extract_command(msg)},
    },
    # SCGP: counterparty dossier (must be before registry and search — "what do we know about" is pipeline)
    {
        'keywords': ['counterparty dossier', 'what do we know about', 'build dossier',
                     'entity dossier', 'company dossier', 'counterparty profile',
                     'counterparty intel', 'counterparty intelligence'],
        'tool': 'scgp_pipeline',
        'extract': lambda msg: _extract_entity_name(msg),
        'args': lambda msg: {'entity_name': _extract_entity_name(msg)},
    },
    # SCGP: GLEIF / LEI lookup (before search — "look up" + "gleif"/"lei" is SCGP, not web search)
    {
        'keywords': ['gleif', 'lei lookup', 'gleif search',
                     'search gleif', 'find lei', 'entity lookup', 'look up entity'],
        'tool': 'scgp_registry',
        'extract': lambda msg: _extract_entity_name(msg),
        'args': lambda msg: {'entity_name': _extract_entity_name(msg)},
    },
    # SCGP: ISDA extraction / classification
    {
        'keywords': ['classify counterparty', 'entity classification', 'classify entity',
                     'counterparty classification', 'extract isda', 'agreement provisions',
                     'extract provisions', 'isda extraction', 'parse isda',
                     'extract agreement', 'classify this counterparty'],
        'tool': 'scgp_convert',
        'extract': lambda msg: None,
        'args': lambda msg: {'schema': 'agreement_provision' if any(
            k in msg.lower() for k in ['extract', 'provision', 'isda', 'parse', 'agreement']
        ) else 'counterparty_intelligence'},
    },
    # Search (web) — must be after more specific patterns (SCGP, memory, vault)
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
        'keywords': ['the playbook', 'my playbook', 'update playbook', 'read playbook',
                     'show playbook', 'self-knowledge', 'improvement queue'],
        'tool': 'playbook_update',
        'extract': lambda msg: None,
        'args': lambda msg: {'section': 'full', 'action': 'read'},
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
        "SCGP = counterparty lookup, GLEIF/LEI search, ISDA extraction, entity dossier\n"
        "CONVERSATION = general questions, definitions, analysis, opinions\n"
        f"\nMessage: \"{message}\"\n"
        "Category:"
    )

    result = llm_fn(prompt, max_tokens=8, temperature=0.0)
    category = result.strip().upper().split()[0] if result.strip() else "CONVERSATION"

    valid = ('SEARCH', 'MEMORY_STORE', 'MEMORY_RECALL', 'VAULT', 'SCGP', 'CONVERSATION')
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
        elif category == 'SCGP':
            return ('scgp_pipeline', {'entity_name': _extract_entity_name(message)})

    return ('conversation', {})
