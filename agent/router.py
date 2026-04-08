"""
Deterministic intent routing. No LLM calls in Layer 1.
Layer 2 uses LLM self-routing — 70B picks the tool AND constructs args.
"""

import json
import re

# ── Router helpers ──────────────────────────────────────────────────────────

# Note: domain-specific entity-lookup helpers were removed in Main 26
# housekeeping. Entity questions are answered via memory recall directly.

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

def _research_args(msg):
    """Build research_probe args from user message.

    The 70B will construct specific commands via L2 routing.
    L1 provides a template based on keywords.
    """
    lower = msg.lower()
    task = msg.strip()

    # Common probe templates
    if 'ane' in lower or 'neural engine' in lower:
        tag = 'ane'
        commands = [
            'kextstat | grep -i ane',
            'ioreg -p IODeviceTree -n ane -r -d 2 2>&1 | head -40',
            'strings /usr/libexec/aned | grep -i "' + task.split()[-1] + '" | head -20',
        ]
    elif 'gpu' in lower or 'metal' in lower:
        tag = 'silicon'
        commands = [
            'system_profiler SPDisplaysDataType 2>&1 | head -20',
            'ioreg -p IOService -n AGXAccelerator -r -d 2 2>&1 | head -40',
        ]
    elif 'memory' in lower or 'dram' in lower:
        tag = 'memory'
        commands = ['vm_stat', 'memory_pressure', 'sysctl hw.memsize']
    else:
        tag = 'general'
        commands = [f'echo "Research task: {task}"']

    return {'task': task, 'commands': commands, 'tag': tag}


def _vault_args(msg):
    """Smart vault_read argument selection.

    - Explicit vault query ("vault for X") → query search
    - Broad status queries ("how's it going") → path read (Roadmap with smart truncation)
    - Specific topic not in a known file → query search with extracted keywords
    - Explicit file reference → path read
    """
    # Explicit vault query pattern
    vq = extract_vault_query(msg)
    if vq:
        return {'query': vq}

    # Explicit file reference
    vp = extract_vault_path(msg)
    if vp:
        return {'path': vp}

    # Broad status queries → path read (smart truncation handles the rest)
    lower = msg.lower()
    broad_status = ["how's it going", "hows it going", "how is it going",
                    "what's new", "whats new", "what is new",
                    "what's up", "whats up", "catch me up", "what did i miss",
                    "what's happening", "project status", "working on",
                    "current projects", "what are we building", "been doing"]
    if any(b in lower for b in broad_status):
        return {'path': 'Roadmap.md'}

    # Topic-specific query — extract keywords and search
    # Strip common prefixes
    cleaned = re.sub(r'^(do we have|what do we know|tell me|anything on|anything about)\s+',
                     '', msg, flags=re.IGNORECASE).strip().rstrip('?.,!')
    cleaned = re.sub(r'^(about|on|for|regarding)\s+', '', cleaned, flags=re.IGNORECASE).strip()
    if len(cleaned) > 3:
        return {'query': cleaned}

    return {'path': 'Roadmap.md'}


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


def extract_vault_query(msg):
    """Extract a search query for vault content from the message."""
    lower = msg.lower()
    # "check the vault for X" / "vault for X" / "in the vault about X"
    for prefix in ['check the vault for ', 'vault for ', 'in the vault about ',
                   'in the vault for ', 'from the vault about ', 'vault about ']:
        if prefix in lower:
            idx = lower.index(prefix) + len(prefix)
            return msg[idx:].strip().rstrip('?.,!')
    return ''

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


# ── Project context keywords — ANY match forces vault_read ──────────────────
# The 70B must NEVER answer project questions from imagination.

PROJECT_CONTEXT_KEYWORDS = {
    # Measurements and metrics
    'tok/s', 'tokens per second', 'acceptance', 'acceptance rate',
    'latency', 'throughput', 'overhead', 'baseline', 'speedup',
    # Project names and technologies
    'eagle', 'eagle-3', 'spec decode', 'speculative', 'n-gram', 'ngram',
    'ane ', 'neural engine', 'coreml', 'mlx', 'anemll',
    'isd', 'drafter', 'draft model', 'verify', 'verification',
    'contention', 'amx', 'sme', 'smopa', 'nax',
    'swift layer', 'layer skip', 'self-spec', 'early exit',
    # Infrastructure
    'production server', 'port 8899', 'midas agent',
    'vault agent', 'enricher', 'daemon',
    # Repo and file names
    'claude.md', 'roadmap', 'four-path', 'orion', 'phantom',
    'ane-toolkit', 'ane-compiler', 'ane-dispatch',
    # Domain context that's project-specific
    'project ', 'our current', 'our model', 'our server', 'we measured',
    'we tested', 'we built', 'we shipped', 'still viable',
    'dead path', 'dead paths', 'parked', 'killed',
    'turbo quant', 'turboquant', 'prewarm',
    'should i focus', 'what to focus', 'focus on',
    'revisit', 'should we try', 'take another crack',
    'living model', 'endorsement',
}


def _has_project_context(msg):
    """Check if message contains project-specific keywords."""
    lower = msg.lower()
    return any(kw in lower for kw in PROJECT_CONTEXT_KEYWORDS)


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
    # Deep research (ANE RE, opcodes, hardware findings, measurements)
    {
        'keywords': ['opcode', 'opcodes', 'dispatch floor', 'pipeline stage',
                     'hwx', 'binary format', 'reverse engineer', 'deep dive',
                     'hardware finding', 'haven\'t explored', 'unexplored',
                     'what don\'t we know', 'what haven\'t we', 'research gap',
                     'ane architecture', 'tile architecture', 'dma channel',
                     'sram cliff', 'pwl', 'espresso', 'softmax pass',
                     'contention measurement', 'dispatch overhead',
                     'what did we find', 'what did we measure', 'what did we discover',
                     'our findings', 'our measurements', 'our research'],
        'tool': 'vault_research',
        'extract': lambda msg: extract_query(msg),
        'args': lambda msg: {'query': extract_query(msg)},
    },
    # Vault read
    {
        'keywords': ['check vault', 'read vault', 'vault read', 'from the vault',
                     'in the vault', 'read the roadmap', 'read the decision',
                     'infrastructure map', 'check roadmap', 'use vault_read',
                     'check the vault', 'the vault',
                     'working on', 'been doing', 'project status',
                     'what are we building', 'current projects',
                     "how's it going", 'hows it going', "what's new",
                     'whats new', "what's going on", "what's happening",
                     'catch me up', 'what did i miss', "what's up", 'whats up'],
        'tool': 'vault_read',
        'extract': lambda msg: extract_vault_path(msg) or 'Roadmap.md',
        'args': lambda msg: _vault_args(msg),
    },
    # Research probe
    {
        'keywords': ['probe', 'investigate', 'research probe', 'run probe',
                     'hardware probe', 'silicon probe', 'ane probe',
                     'reverse engineer', 'dig into'],
        'tool': 'research_probe',
        'extract': lambda msg: msg,
        'args': lambda msg: _research_args(msg),
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
    # Note: domain-specific entity-lookup pipelines were removed in Main 26
    # housekeeping. Entity questions go through memory recall.
    # Search (web) — must be BEFORE vault to catch "what's new on reddit" etc.
    {
        'keywords': ['search for', 'search google', 'google ', 'look up',
                     'find out about', 'latest news on', 'current price of',
                     'search the web', 'look online',
                     'on reddit', 'on hacker news', 'on hn ',
                     'on the web', 'online about'],
        'tool': 'browse_search',
        'extract': lambda msg: extract_search_query(msg),
        'args': lambda msg: {'query': extract_search_query(msg)},
    },
    # X feed
    {
        'keywords': ['x feed', 'twitter feed', 'scan x', 'check x',
                     'check twitter', 'what\'s on x', 'scan my feed',
                     'what\'s on twitter', 'on twitter', 'search x ',
                     'on x right now', 'interesting on x', 'what\'s happening on x',
                     'from x', 'x right now', 'x timeline',
                     'new on x', 'new on twitter', 'happening on x'],
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


_WEB_INDICATORS = {'reddit', 'hacker news', 'twitter', 'on the web', 'online',
                    'on hn', 'google', 'internet', 'browse', 'website'}


def layer1_route(message):
    """
    Keyword-based routing. Returns (tool_name, args_dict) or None.
    Zero model calls. Instant.
    """
    lower = message.lower()
    lower_stripped = strip_greeting(lower)

    # Pre-check: if the message has web indicators AND a status keyword,
    # skip vault_read so the search pattern can catch it
    has_web_indicator = any(w in lower for w in _WEB_INDICATORS)

    for pattern in PATTERNS:
        if any(kw in lower_stripped for kw in pattern['keywords']):
            # Skip vault_read if message has web indicators (let search catch it)
            if pattern['tool'] == 'vault_read' and has_web_indicator:
                continue
            args = pattern['args'](message)
            return (pattern['tool'], args)

    return None


TOOL_DESCRIPTIONS = """Available tools (use ONLY if the user's request requires one):

- vault_read: Read project docs/roadmap/status from the Obsidian vault. Args: {"path": "Roadmap.md"} or {"query": "search term"}. Use for: project status, what we're working on, roadmap, decisions, infrastructure, "how's it going", "what's new", status updates.
- vault_research: Deep search of ANE reverse engineering files, hardware measurements, opcode catalogs, agent reports. Args: {"query": "search term"}. Use for: specific hardware findings, opcodes, measurements, "what did we find", "what haven't we explored", research gaps, deep technical questions about ANE/hardware.
- vault_insight: Cross-reference vault docs and memory on a topic. Args: {"topic": "some topic"}. Use for: deep research on a topic across vault and memory.
- memory_recall: Search stored conversation memories. Args: {"query": "search term"}. Use for: "what's in memory about X", "do you remember X", past conversations.
- memory_ingest: Store something in memory. Args: {"role": "user", "text": "content to store"}. Use for: "remember this", "save this".
- memory_stats: Check memory statistics. Args: {}. Use for: "how many memories", "memory stats".
- browse_search: Search the web via browser. Args: {"query": "search term"}. Use for: web search, current events, latest news, prices, "search for X".
- browse_x_feed: Scan X/Twitter feed. Args: {"count": 5}. Use for: "check X", "scan my feed", "what's on X/Twitter".
- scan_digest: Check scanner candidates. Args: {"mode": "latest"|"unreviewed"|"clear"|"stats"}. Use for: "check scans", "new scans", "scan digest".
- shell: Run a shell command. Args: {"command": "ls -la"}. Use for: "run ls", "execute command".
- message_claude: Send a message to Claude's inbox. Args: {"message": "text", "priority": "medium"}. Use for: "message claude", "tell claude".
- self_test: Run agent tests. Args: {"mode": "light"|"hardcore"}. Use for: "test yourself", "run tests".
- brain_snapshot: Agent routing diagnostics. Args: {"scope": "session"|"last"}. Use for: "how did you route that", "show diagnostics".
- playbook_update: Read/update the agent playbook. Args: {"section": "full", "action": "read"}. Use for: "show playbook", "read playbook".
- heartbeat: Launch monitoring dashboard. Args: {}. Use for: "open dashboard", "heartbeat".
- self_improve: Analyze for improvements. Args: {"mode": "analyze"}. Use for: "improve yourself", "check for improvements".
- research_probe: Run research commands and save findings to vault. Args: {"task": "description", "commands": ["cmd1", "cmd2"], "tag": "ane|silicon|memory|performance"}. Use for: "probe X", "investigate X", "research X", "run these commands and report", hardware analysis, reverse engineering tasks.

IMPORTANT: If the user is asking a general knowledge question, making conversation, or asking you to explain something, do NOT use a tool. Just respond directly."""


def layer2_llm_route(message, llm_fn):
    """
    LLM self-routing. The 70B sees the full tool list and decides:
      a) TOOL: tool_name {"arg1": "val"} — to call a tool
      b) Plain text — for direct conversation

    Returns (tool_name, args_dict) or ('conversation', {}).
    """
    prompt = (
        f"You are Midas, a local AI agent. The user said:\n"
        f'"{message}"\n\n'
        f"{TOOL_DESCRIPTIONS}\n\n"
        f"If a tool is needed, respond with EXACTLY this format on the first line:\n"
        f'TOOL: tool_name {{"arg1": "value1"}}\n\n'
        f"If no tool is needed (general knowledge, conversation, explanations, math), "
        f"respond with just the word CONVERSATION\n\n"
        f"Your response:"
    )

    result = llm_fn(
        [{"role": "user", "content": prompt}],
        max_tokens=120, temperature=0.0,
    )
    result = result.strip()

    # Parse all lines looking for TOOL: calls. The 70B sometimes emits
    # CONVERSATION first and then self-corrects with TOOL: on a later line.
    # If ANY line has a valid TOOL: call, use it. Otherwise conversation.
    def _parse_tool_line(line):
        """Try to extract (tool_name, args) from a TOOL: line. Returns None on failure."""
        # Strip LLM artifacts
        for artifact in ['assistant', '<|eot_id|>', '<|eom_id|>']:
            line = line.replace(artifact, '').strip()
        if not line.upper().startswith('TOOL:'):
            return None
        rest = line[5:].strip()
        brace_idx = rest.find('{')
        if brace_idx > 0:
            tool_name = rest[:brace_idx].strip()
            json_str = rest[brace_idx:]
            try:
                args = json.loads(json_str)
                return (tool_name, args)
            except json.JSONDecodeError:
                # Try to find matching closing brace
                try:
                    depth = 0
                    for i, c in enumerate(json_str):
                        if c == '{':
                            depth += 1
                        elif c == '}':
                            depth -= 1
                            if depth == 0:
                                args = json.loads(json_str[:i + 1])
                                return (tool_name, args)
                except json.JSONDecodeError:
                    pass
        else:
            tool_name = rest.split()[0] if rest.split() else None
            if tool_name and tool_name.lower() != 'none':
                return (tool_name, {})
        return None

    # Scan all lines for a valid TOOL: call
    for line in result.split('\n'):
        parsed = _parse_tool_line(line.strip())
        if parsed:
            return parsed

    # No valid TOOL: found — conversation
    return ('conversation', {})


def _is_anaphoric(message):
    """Detect messages whose object is just a pronoun (that, this, it, them).
    These refer to the prior conversation turn — routing to a tool is wrong
    because the tool can't resolve what 'that' means.
    """
    stripped = extract_query(message).lower().strip().rstrip('?.,!')
    # After stripping "can you / please / tell me / etc.", is what's left
    # just verb(s) + pronoun? e.g. "research that", "explain this", "look into it"
    anaphoric = re.match(
        r'^([\w\s]{1,20}?)\s+(that|this|it|them|those|these|the above|what you said|'
        r'what you found|the last one|the result|the results)$',
        stripped
    )
    if not anaphoric:
        return False
    # Make sure the verb part is short (1-3 words) to avoid false positives
    verb_part = anaphoric.group(1).strip()
    return len(verb_part.split()) <= 3


def route(message, llm_fn=None):
    """
    Main routing function.
    Returns (tool_name, args_dict) or ('conversation', {}).
    """
    # Layer 1: deterministic keyword routing (must run first — "remember that" etc.)
    result = layer1_route(message)
    if result:
        return result

    # Anaphoric check: "research that", "explain this" — tools can't resolve
    # pronoun references, route to conversation where history is available.
    # Runs AFTER layer 1 so "remember that" still hits memory_ingest.
    if _is_anaphoric(message):
        # Even anaphoric references to project topics must check vault
        if _has_project_context(message):
            return ('vault_read', {'query': extract_query(message)})
        return ('conversation', {})

    # PROJECT CONTEXT GATE: any project-specific keywords force vault lookup.
    # The 70B must NEVER answer project questions from imagination.
    if _has_project_context(message):
        query = extract_vault_query(message) or extract_query(message)
        # Deep research questions → vault_research (searches ANE RE, agent reports, session logs)
        lower = message.lower()
        research_markers = ['opcode', 'dispatch floor', 'pipeline', 'hwx', 'deep dive',
                           'haven\'t explored', 'unexplored', 'what did we find',
                           'what did we measure', 'our findings', 'our research',
                           'research gap', 'what don\'t we know', 'hardware finding',
                           'tile architecture', 'sram', 'pwl', 'binary format']
        if any(m in lower for m in research_markers):
            return ('vault_research', {'query': query or message})
        return ('vault_read', {'query': query} if query else {'path': 'Roadmap.md'})

    # Layer 2: LLM self-routing — 70B picks the tool AND constructs args
    if llm_fn:
        return layer2_llm_route(message, llm_fn)

    return ('conversation', {})
