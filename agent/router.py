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
    """Remove hi/hey/hello from start of message.

    Requires a word boundary after the greeting so we don't eat the
    leading 'hi' from 'his', 'history', 'highlight', 'hierarchy', etc.
    Pre-existing bug surfaced by Main 38 Session 6 fix B (github
    keywords): 'his repos contain ...' was being mangled to 's repos
    contain ...' before pattern matching, causing the github pattern
    to miss.
    """
    return re.sub(
        r'^(hi|hey|hello|yo|sup|good morning|good afternoon)\b\s*[,.]?\s*',
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


# Main 62 pilot-fix 1: guard the bare `m\d+` session shorthand against
# Apple Silicon model-name collisions ("m5 pro", "m4 max", "m2 ultra",
# "m1 mini", "m3 air") and iPhone/ANE generation names (A17 Pro, H17G,
# etc.). Without this guard, T19 ("how does that compare to our m5 pro?")
# routed to knowledge/session_milestones.md as "Main 5".
_APPLE_SILICON_HW_CONTEXT_RE = re.compile(
    r"\b[MmAaHh](\d{1,2})\s*(?:pro|max|ultra|air|mini|chip|g\b|bionic)\b",
    re.IGNORECASE,
)


def _is_apple_silicon_hw_token(msg: str) -> bool:
    """True iff msg contains an Apple Silicon model-name token
    ("m5 pro", "a17 pro", "h17g") that would otherwise collide with the
    bare `m\\d+` session regex. Used to suppress the single-session
    fallback in _session_query_args. Does NOT suppress the multi-session
    range/pair/last-N patterns (those require "main"/"session"/"m"
    adjacent to a dash/and/to, which is unambiguous)."""
    return bool(_APPLE_SILICON_HW_CONTEXT_RE.search(msg))


def _session_query_args(msg):
    """Main 46: build vault_read args for session-number queries.

    Searches session_milestones.md and the corresponding closing report.

    Main 55 P1b: detect multi-session windows, enumerated pairs, and
    "last N sessions" forms. Range queries route to subconscious_state_
    synthesis.md (the multi-session aggregator) with an OR-expanded query
    so the vault_read keyword scorer matches every session in the window,
    not just the first anchor.

    Main 62 pilot-fix: the bare single-session `m\\d+` regex now requires
    that the message not contain an Apple Silicon model-name token
    ("m5 pro", "a17 pro", "h17g"). The explicit "main N" / "session N"
    forms are unaffected.
    """
    # Range: "main 46-50", "main 46 to 50", "main 46 – 50"
    m_range = re.search(
        r'\b(?:main|session|m)\s*(\d{1,3})\s*(?:-|–|—|→|to|through|thru|\.\.+)\s*(?:main|session|m)?\s*(\d{1,3})\b',
        msg, re.IGNORECASE)
    if m_range:
        lo, hi = int(m_range.group(1)), int(m_range.group(2))
        if hi < lo:
            lo, hi = hi, lo
        # Cap span to prevent runaway OR query
        hi = min(hi, lo + 20)
        nums = range(lo, hi + 1)
        expanded = " OR ".join(f"main {n}" for n in nums)
        return {'query': expanded,
                'path': 'subconscious_state_synthesis.md'}

    # Enumerated pair: "main 46 and main 48", "main 53 & main 54",
    # "main 46, main 50"
    m_pair = re.search(
        r'\b(?:main|session|m)\s*(\d{1,3})\s*(?:and|&|,|plus|\+)\s*(?:main|session|m)?\s*(\d{1,3})\b',
        msg, re.IGNORECASE)
    if m_pair:
        a, b = m_pair.group(1), m_pair.group(2)
        expanded = f"main {a} OR main {b}"
        return {'query': expanded,
                'path': 'subconscious_state_synthesis.md'}

    # "last N sessions" / "past N sessions" / "recent N sessions"
    m_last = re.search(
        r'\b(?:last|past|recent|previous)\s+(\d{1,3})\s+sessions?\b',
        msg, re.IGNORECASE)
    if m_last:
        return {'query': extract_query(msg) or 'recent sessions',
                'path': 'subconscious_state_synthesis.md'}

    # Main 62 pilot-fix: prefer the explicit "main N"/"session N" form
    # over the bare "m N" form. If only the bare "m\d+" form matches and
    # the message contains an Apple Silicon HW token ("m5 pro", "a17",
    # "h17g"), skip the session route entirely.
    m_explicit = re.search(
        r'\b(?:main|session)\s*(\d{1,3})\b', msg, re.IGNORECASE)
    if m_explicit:
        num = m_explicit.group(1)
        return {'query': f'main {num}', 'path': 'knowledge/session_milestones.md'}
    if _is_apple_silicon_hw_token(msg):
        return {'query': extract_query(msg)}
    m = re.search(r'\bm\s*(\d{1,3})\b', msg, re.IGNORECASE)
    if m:
        num = m.group(1)
        return {'query': f'main {num}', 'path': 'knowledge/session_milestones.md'}
    return {'query': extract_query(msg)}


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

    # Main 42: narrowed to explicit roadmap/status intent. Casual phrases
    # ("what's up", "working on") now conversation-route and never reach here.
    lower = msg.lower()
    broad_status = ["project status", "read roadmap", "check roadmap",
                    "current projects", "what are we building"]
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

# M54 Phase 4c: browse_search-specific query builder. Distinct from
# extract_search_query (which is a general strip). This one knows:
#   (a) the sources the user typically targets on the web
#   (b) the active research threads, so "our research" / "our work"
#       expand into actual search keywords Google can index against
# Example:
#   "is there any recent posts on Hacker News that relates to our research?"
#   -> '"Hacker News" Apple Silicon ANE cognitive architecture local LLM'
_BROWSE_SOURCE_MAP = {
    "hacker news": "Hacker News", "hackernews": "Hacker News",
    "hn":          "Hacker News",
    "arxiv":       "arxiv.org", "arxiv.org": "arxiv.org",
    "reddit":      "reddit", "r/localllama": "r/LocalLLaMA",
    "github":      "github.com",
    "hugging face": "huggingface.co", "huggingface": "huggingface.co",
    "papers with code": "paperswithcode.com",
    "semantic scholar": "semanticscholar.org",
    "google scholar": "scholar.google.com",
    "stack overflow":  "stackoverflow.com",
}
# Active-thread keywords that expand "our research" into a Google-tractable
# string. Kept tight so the combined query stays under Google's effective
# length. Update these when the research focus shifts.
_OUR_RESEARCH_KEYWORDS = (
    "Apple Silicon", "ANE", "Neural Engine",
    "cognitive architecture", "local LLM inference",
)
_OUR_RESEARCH_PATTERNS = (
    "our research", "our work", "our project", "our stack",
    "our findings", "our paper",
)


# Main 62 Bug 1: anaphoric web-search detector. Matches messages like
# "can you do a web search?", "did you do a web search?", "would you
# search for it?" — i.e. questions/imperatives whose content noun is
# just a pronoun or missing entirely. When matched AND a prior_subject
# is available, _browse_search_query folds the prior subject in so the
# tool doesn't send "can you do a web search" as the literal Google
# query.
_ANAPHORIC_PATTERN = re.compile(
    r"^(?:can you|could you|will you|did you|would you|please)?\s*"
    r"(?:do|run|try|perform|go)?\s*"
    r"(?:a|an|the)?\s*"
    r"(?:web\s*search|websearch|search|lookup|google|browse|look\s*it\s*up)\s*"
    r"(?:for|on|about|up)?\s*"
    r"(?:it|that|this|them|those)?\s*[?.]?\s*$",
    re.IGNORECASE,
)


def _is_anaphoric_search(msg: str) -> bool:
    """Return True if `msg` is a pronoun/verb-only web-search request."""
    if not msg:
        return False
    stripped = msg.strip()
    if not stripped:
        return False
    return bool(_ANAPHORIC_PATTERN.match(stripped))


def _extract_prior_subject(prior_user_message: str) -> str:
    """Pull a content subject from the prior user turn for anaphoric fold-in.

    Best-effort: strip leading conversational framing, drop trailing
    punctuation, collapse whitespace. Returns '' if the prior turn
    itself looks like an anaphoric framing (no content).
    """
    if not prior_user_message:
        return ""
    s = prior_user_message.strip()
    if not s:
        return ""
    # If the prior turn was itself an anaphoric search request, no content
    if _is_anaphoric_search(s):
        return ""
    # Strip leading greeting + instruction verbs
    s = strip_greeting(s)
    s = re.sub(
        r'^(can you |could you |please |would you |i need you to |'
        r'tell me (about |)|show me |find me |get me |look up |search for |'
        r'what (is|are|do you know) (the |about )?|'
        r'do you know (the |anything about )?|'
        r'what.?s the |how (do|does) |why (do|does) )',
        '', s, flags=re.I,
    )
    s = re.sub(r'\s+', ' ', s).strip(' ?.,!')
    return s


def _browse_search_query(msg: str, prior_user_message: str = "") -> str:
    """Build a Google-ready query from a natural-language web-search request.

    Pipeline:
      0. Main 62 Bug 1: if the message is anaphoric ("can you do a web
         search?") AND a prior user turn is available, use the prior
         turn's subject as the query body instead of the literal
         anaphoric framing.
      1. Strip interrogative + directive framing via extract_search_query.
      2. Detect a source reference ("Hacker News", "arxiv", ...) and lift
         it out of the free text as a quoted site anchor.
      3. If the user referenced "our research" / "our work", strip it and
         append the active-thread keyword block so Google has real terms.
      4. Collapse whitespace and punctuation.
    """
    # Main 62 Bug 1: anaphoric fold-in. If the query has no content
    # (just "can you do a web search?"), use the prior turn's subject.
    if _is_anaphoric_search(msg):
        prior_subj = _extract_prior_subject(prior_user_message)
        if prior_subj:
            # Rebuild the msg so downstream source/"our research" detection
            # still runs on whatever the prior turn contained.
            msg = prior_subj
        else:
            # No prior context — return empty so tool_executor can gate
            # with a helpful error ("what would you like me to search?").
            return ""
    low = msg.lower()
    # Stage 3 (detected early) — "our research" trigger. Detect from the
    # RAW input because extract_search_query strips the trailing bind
    # phrase ("that relates to our research") before we'd get to check.
    expand = any(pat in low for pat in _OUR_RESEARCH_PATTERNS)
    # Stage 2 — source detection. First hit wins. Strip the source phrase
    # from the message so it doesn't also appear as body text.
    source = None
    stripped = msg
    for phrase, canonical in _BROWSE_SOURCE_MAP.items():
        if phrase in low:
            source = canonical
            # Strip all occurrences of the phrase (case-insensitive),
            # along with a leading "on "/"from " preposition if present.
            stripped = re.sub(
                rf'\b(on|from|at|in)?\s*{re.escape(phrase)}\b',
                ' ', stripped, flags=re.I)
            break
    # Stage 1 — run the general strip on the source-stripped message.
    body = extract_search_query(stripped)
    # Strip any residual "our research"/"our work" fragments that survived
    # extract_search_query (it only strips the trailing-bind form).
    for pat in _OUR_RESEARCH_PATTERNS:
        body = re.sub(
            rf'\b(on|to|for|about)?\s*{re.escape(pat)}\b',
            ' ', body, flags=re.I)
    body = re.sub(r'\s+', ' ', body).strip(' ?.,')
    parts: list[str] = []
    if source:
        parts.append(f'"{source}"')
    if body:
        parts.append(body)
    if expand:
        parts.append(" ".join(_OUR_RESEARCH_KEYWORDS))
    out = " ".join(parts).strip()
    return out or body or ""


def extract_search_query(msg):
    """Pull the search query, stripping routing phrases."""
    q = strip_greeting(msg)
    # Strip instruction prefixes that aren't the actual query
    q = re.sub(r'^(do a |please |can you |could you |would you )', '', q, flags=re.I)
    # M54 Phase 4b: added "do a search for", "web search for",
    # "go to <source> and search for", "go online and", "look online for".
    # Q07 / Hacker News failure: router chose browse_search correctly but
    # extract_search_query didn't strip the directive, so Google got the
    # full natural-language sentence as its query and returned garbage.
    q = re.sub(r'^(web\s*search|websearch|search the web|search online|'
               r'browse the web|do a web\s*search|do a websearch|'
               r'do a search|web\s*search for|search for|'
               r'go online and|look online for|go online for)'
               r'(\s+(for|on|about|and\s+find))?\s*', '', q, flags=re.I)
    # M54 Phase 4b: "go to arxiv and search for", "go to reddit and search".
    # M54 Phase 4c: make the source word OPTIONAL because _browse_search_query
    # strips the source upstream (leaving "go to  and search for" as the
    # residual). Also tolerate collapsed whitespace.
    q = re.sub(r'^go\s+to\s+(\w+(\.\w+)?)?\s*(and\s+)?(search|look|find|check)'
               r'(\s+(for|on|about))?\s*', '', q, flags=re.I)
    q = re.sub(r'^(search|google|look up|look on|look\s+for|look|find|'
               r'search for|search google for|tell me about|show me|'
               r'get me|find me)(\s+for)?\s+',
               '', q, flags=re.I)
    # M54 Phase 4b: interrogative web-search framing. User asks web
    # questions as questions, not imperatives. "is there any recent posts
    # on Hacker News ..." — the whole 70-char question was being passed
    # as the search query. Strip the question framing.
    q = re.sub(r'^(is there any |are there any |have there been any |'
               r'do you see any |can you find |can you search for |'
               r'any recent |any new )', '', q, flags=re.I)
    q = re.sub(r'^(what is|what are|who is|where is|how is|how do|when did)\s+',
               '', q, flags=re.I)
    # Trailing "that relates to our research" / "for our work" — these
    # are source-binding phrases that don't help Google. Strip them.
    q = re.sub(r'\s+(that\s+(relates|relate|pertains|is\s+related)\s+to\s+'
               r'our\s+(research|work|project|stack).*)$', '', q, flags=re.I)
    # Trailing question mark on interrogatives
    q = q.strip().rstrip('?').strip()
    if not q or q.lower() in ('search', 'web search', 'it', 'that', 'something'):
        return ""  # Empty query — tool_executor will gate with helpful error
    return q


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
    # Deep research (ANE RE, opcodes, hardware findings, measurements).
    # M54 Phase 3: removed generic possessive keywords ("our research",
    # "our findings", "our measurements", "what did we find/measure/discover",
    # "what don't we know", "what haven't we", "research gap") — these
    # conflicted with explicit external-source intent ("go to arxiv's
    # website ... related to OUR RESEARCH" routed here instead of
    # browse_search). Kept only specific technical terms that
    # unambiguously indicate vault content.
    {
        'keywords': ['opcode', 'opcodes', 'dispatch floor', 'pipeline stage',
                     'hwx', 'binary format', 'reverse engineer', 'deep dive',
                     'hardware finding',
                     'ane architecture', 'tile architecture', 'dma channel',
                     'sram cliff', 'pwl', 'espresso', 'softmax pass',
                     'contention measurement', 'dispatch overhead',
                     # M53: heavily-researched project topics. When the user
                     # asks about these, check the vault before hitting the
                     # web — we have deep dossiers on each.
                     'meta ray ban', 'meta ray-ban', 'rayban', 'hypernova',
                     'neural band', 'greatwhite', 'ar1 gen 1',
                     'snapdragon ar1'],
        'tool': 'vault_research',
        'extract': lambda msg: extract_query(msg),
        'args': lambda msg: {'query': extract_query(msg)},
    },
    # Main 46: Session-indexed retrieval. Catches "what shipped in main 40",
    # "main 40 summary", "M42 results", etc. Routes to vault_read with
    # targeted query against session_milestones.md and closing reports.
    {
        'keywords': [],  # uses regex match below instead of keywords
        'tool': 'vault_read',
        'extract': lambda msg: msg,
        'args': lambda msg: _session_query_args(msg),
        'regex': re.compile(
            r'\b(?:main|session|m)\s*(\d{1,3})\b', re.IGNORECASE),
    },
    # Vault read
    {
        # Main 42: removed casual phrases ("what's up", "working on",
        # "how's it going", etc.) — these should conversation-route where
        # the briefing answers naturally. Vault_read L1 is now explicit
        # vault-intent only. Casual status queries fall through to L2/conversation.
        'keywords': ['check vault', 'read vault', 'vault read', 'from the vault',
                     'in the vault', 'read the roadmap', 'read the decision',
                     'infrastructure map', 'check roadmap', 'use vault_read',
                     'check the vault', 'the vault'],
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
    # X feed — must be BEFORE browse_search so X-specific phrasings
    # ("search on X", "X posts", "posts on X about Y") route to the feed
    # scanner and not the general web search. Main 38 P2 fix (moved
    # before browse_search, expanded keyword list). Session 1 of Main 38
    # P2 showed the router dispatching browse_search or vault_read for 4
    # consecutive X queries because the X-feed list didn't cover
    # natural user phrasings like "X posts", "search on X", "top posts
    # on X for ANE research". User had to correct three times.
    {
        'keywords': [
            # Original Main 34 set
            'x feed', 'twitter feed', 'scan x', 'check x',
            'check twitter', "what's on x", 'scan my feed',
            "what's on twitter", 'on twitter', 'search x ',
            'on x right now', 'interesting on x', "what's happening on x",
            'from x', 'x right now', 'x timeline',
            'new on x', 'new on twitter', 'happening on x',
            # Main 38 P2 additions
            'x posts', 'posts on x', 'post on x',
            'top posts on x', 'top x posts',
            'search on x', 'search on twitter', 'search x for',
            'x for information', 'on x about', 'on x for',
            'x research', 'research on x', 'twitter posts',
            'do a search on x', 'do a search on twitter',
            'x thread', 'twitter thread',
            'interesting x posts', 'interesting twitter posts',
            'tweets about', 'tweet about',
        ],
        'tool': 'browse_x_feed',
        'extract': lambda msg: _extract_x_handle(msg),
        'args': lambda msg: _x_feed_args(msg),
    },
    # Search (web) — must be BEFORE vault to catch "what's new on reddit" etc.
    # Main 37 Fix 2: added explicit "web search" / "search the web" /
    # "search the internet" / "from the internet" / "can you search" /
    # "search online" phrasings so layer2 never has to decide. Any of
    # these keywords deterministically dispatches browse_search.
    {
        'keywords': ['search for', 'search google', 'google ', 'look up',
                     'find out about', 'latest news on', 'current price of',
                     'search the web', 'search the internet',
                     'from the internet', 'can you search',
                     'can you web search', 'can you do a web search',
                     'web search', 'websearch', 'do a websearch',
                     'do a web search', 'search online', 'look online',
                     'look on line', 'look online for', 'search on line',
                     'look up online', 'on reddit', 'on hacker news',
                     'on hn ', 'on the web', 'online about',
                     'browse the web',
                     # Main 38 Session 6 additions: github / repo intent.
                     'search his repos', 'search their repos',
                     'search the repos', 'search github', 'on github',
                     'github repos', 'github repo', 'check his repos',
                     'check their repos', 'his repos', 'their repos',
                     "what's on github",
                     # M54 Phase 3: external-source names. When the user
                     # names an external source explicitly, route to web
                     # search. Covers "review arxiv for papers" style
                     # queries that had no L1 match.
                     ' arxiv', 'arxiv.org', 'on arxiv', 'new papers on',
                     'recent papers', 'from huggingface', 'huggingface.co',
                     'semantic scholar', 'paperswithcode',
                     # Main 57 P3: external-source names that appeared
                     # as topic-qualifiers in live validation. T22
                     # ("any recent Hacker News posts about ANE") was
                     # matched at vault_read because the existing "on
                     # hacker news" / "on hn " triggers require the
                     # leading preposition. Adding bare proper-noun
                     # forms + "posts" variants so the "posts about X"
                     # framing also routes to web search.
                     'hacker news', 'hackernews', 'hn posts',
                     'hn post', 'reddit posts', 'reddit post',
                     'arxiv posts', 'arxiv post'],
        'tool': 'browse_search',
        'extract': lambda msg: _browse_search_query(msg),
        'args': lambda msg: {'query': _browse_search_query(msg)},
    },
    # M54 Phase 3: vault_research fallback for generic possessive intent.
    # Lower priority than browse_search so "search the web for our
    # research" correctly goes to browse_search. Gated on no web
    # indicator (see layer1_route) — these keywords only deterministically
    # route to vault when the query doesn't also indicate web intent.
    {
        'keywords': ['our research', 'our findings', 'our measurements',
                     'what did we find', 'what did we measure',
                     'what did we discover', "what don't we know",
                     "what haven't we", 'research gap',
                     "haven't explored", 'unexplored',
                     # M54 Phase 4: "have we researched" / "what have we
                     # researched" — Q03 (ANE enclave) fell through to L2
                     # vault_read because none of the above matched.
                     'have we researched', 'what have we researched',
                     'what have we found', 'have we explored',
                     'have we investigated', 'have we studied',
                     'have we documented'],
        'tool': 'vault_research',
        'extract': lambda msg: extract_query(msg),
        'args': lambda msg: {'query': extract_query(msg)},
        '_gate_no_web': True,  # M54: skip if has_web_indicator
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
                    'on hn', 'google', 'internet', 'browse', 'website',
                    'web search', 'websearch', 'search the web', 'search online'}

# Main 38 P2: X/Twitter-specific tokens. Any one of these in the query
# forces routing to browse_x_feed and blocks browse_search from catching
# a parallel keyword. Fixes the Main 38 Session 1 bug where "search on X"
# matched browse_search because "search" was in both keyword lists and
# browse_search ran first.
_X_TOKENS = (
    ' x ', ' x.', ' x?', 'x post', 'x feed', 'x thread', 'x timeline',
    # M54 Phase 3: word-boundary fixes for short ' x' tokens.
    # "on xariv" used to match ' on x' — false positive on any word
    # starting with x after "on". Add trailing whitespace/punct.
    ' on x ', ' on x.', ' on x?', ' on x,',
    'from x ', 'from x.', 'from x?',
    ' x for ', 'x about ', 'x research',
    ' twitter', 'tweet',
)


def _has_x_intent(lower: str) -> bool:
    """Return True if the message clearly references X/Twitter content."""
    # Pad with spaces so word-boundary matches work for tokens starting
    # or ending with whitespace (e.g., ' x ' matching start or end).
    padded = f" {lower} "
    if any(tok in padded for tok in _X_TOKENS):
        return True
    # An explicit @handle (not part of an email) is itself X intent.
    return bool(_X_HANDLE_AT_RE.search(lower))


# Main 38 Session 4 fix: extract a target X handle from the user message
# so browse_x_feed can navigate to a specific user's profile instead of
# always scraping the home timeline. Session 4 turn 4 hit this: query
# "what's the most recent post from maderix on X" routed to browse_x_feed
# correctly but the tool returned home-feed tweets from @EpochTimesAd
# because the tool had no handle parameter. The fix: detect @handle,
# "from <name>", "by <name>", "<name>'s posts/tweets/feed" and pass the
# extracted handle through to scan_x_feed.
_X_HANDLE_AT_RE = re.compile(r'(?<![A-Za-z0-9.])@([A-Za-z0-9_]{2,15})')
_X_HANDLE_FROM_RE = re.compile(
    r'\b(?:from|by)\s+([A-Za-z][A-Za-z0-9_]{1,14})\b', re.IGNORECASE)
_X_HANDLE_POSS_RE = re.compile(
    r"\b([A-Za-z][A-Za-z0-9_]{1,14})['\u2019]s\s+(?:\w+\s+){0,3}"
    r"(?:post|posts|tweet|tweets|feed|thread|threads)\b",
    re.IGNORECASE)
_X_HANDLE_STOPWORDS = {
    'the', 'twitter', 'tweet', 'tweets', 'post', 'posts', 'feed',
    'now', 'today', 'yesterday', 'recently', 'lately', 'home', 'someone',
    'people', 'anyone', 'everyone', 'somebody', 'them', 'him', 'her',
    'user', 'users', 'a', 'an', 'my', 'our', 'their', 'his',
}


def _extract_x_handle(message: str):
    """Return a target X handle (no @) extracted from `message`, or None.

    Priority: explicit @handle > possessive form > "from/by <name>".
    Stopword filter prevents 'from the internet', 'from now', etc. from
    being mistaken for handles.
    """
    m = _X_HANDLE_AT_RE.search(message)
    if m:
        return m.group(1)
    m = _X_HANDLE_POSS_RE.search(message)
    if m and m.group(1).lower() not in _X_HANDLE_STOPWORDS:
        return m.group(1)
    m = _X_HANDLE_FROM_RE.search(message)
    if m and m.group(1).lower() not in _X_HANDLE_STOPWORDS:
        return m.group(1)
    return None


def _x_feed_args(message: str) -> dict:
    args = {'count': 5}
    # Main 62 pilot-fix 4: detect 2+ explicit @handles for
    # "difference between @X and @Y posts" / "compare @X and @Y" queries.
    # Emit `handles` list so tool_executor can fetch both and the
    # synthesizer sees merged results. Falls back to the original single
    # `handle` path for 0/1 @handles.
    at_handles = _X_HANDLE_AT_RE.findall(message)
    # dedup while preserving order
    seen = set()
    dedup = []
    for h in at_handles:
        hl = h.lower()
        if hl not in seen:
            seen.add(hl)
            dedup.append(h)
    if len(dedup) >= 2:
        args['handles'] = dedup
        # Also set primary handle for tool_executor back-compat fallback
        args['handle'] = dedup[0]
        return args
    h = _extract_x_handle(message)
    if h:
        args['handle'] = h
    return args


# Main 62 Bug 2: explicit web-search phrases force browse_search
# routing regardless of other keyword overlaps. T52 ("do a web search
# on Hacker News to see if there are any recent posts on Apple silicon
# reverse engineering") routed to vault_research because "reverse
# engineer" matched the vault_research rule list first. These phrases
# are an unambiguous user signal and must win.
_WEB_SEARCH_OVERRIDE = (
    "web search", "websearch", "search the web", "search online",
    "do a search on", "search on google", "look it up online",
    "search the internet", "do a web search", "google for",
    "go online and search", "can you search online",
)


def _is_explicit_web_search(msg: str) -> bool:
    low = msg.lower()
    return any(p in low for p in _WEB_SEARCH_OVERRIDE)


def layer1_route(message, prior_user_message: str = ""):
    """
    Keyword-based routing. Returns (tool_name, args_dict) or None.
    Zero model calls. Instant.

    `prior_user_message` (Main 62 Bug 1) is the user's previous turn,
    used by _browse_search_query to resolve anaphoric follow-ups like
    "can you do a web search?" into a concrete query.
    """
    lower = message.lower()
    lower_stripped = strip_greeting(lower)

    # Main 62 Bug 2: priority override. Explicit web-search phrases
    # go to browse_search regardless of what else matches.
    if _is_explicit_web_search(message):
        return ('browse_search',
                {'query': _browse_search_query(message, prior_user_message)})

    # Main 62 Bug 1: anaphoric web-search follow-up. "Did you do a web
    # search?" / "Can you do a web search?" on their own match none of
    # the browse_search keywords, but they are unambiguous search
    # requests. If we have a prior user message with content, route to
    # browse_search with the prior subject folded in.
    if _is_anaphoric_search(message) and prior_user_message:
        return ('browse_search',
                {'query': _browse_search_query(message, prior_user_message)})

    # Pre-check: if the message has web indicators AND a status keyword,
    # skip vault_read so the search pattern can catch it
    has_web_indicator = any(w in lower for w in _WEB_INDICATORS)
    # Main 38 P2: if the message has X/Twitter intent, skip both
    # vault_read and browse_search so browse_x_feed captures it.
    has_x_intent = _has_x_intent(lower)

    # Main 62 pilot-fix 1: Apple Silicon HW token gate. "m5 pro", "a17",
    # "h17g" look like session shorthand to the bare `m\d+` regex. If the
    # message contains such a token AND has no explicit "main"/"session"
    # word, suppress the session-indexed vault_read pattern so we don't
    # misroute Apple Silicon hardware questions to session_milestones.md.
    apple_si_hw = _is_apple_silicon_hw_token(message)
    has_explicit_session_word = bool(
        re.search(r'\b(?:main|session)\b', lower))

    for pattern in PATTERNS:
        # Main 46: support regex-based patterns alongside keyword lists
        matched = any(kw in lower_stripped for kw in pattern['keywords'])
        if not matched and 'regex' in pattern:
            matched = bool(pattern['regex'].search(lower))
        if matched:
            # Main 62 pilot-fix 1: suppress session-indexed vault_read
            # on Apple Silicon HW tokens unless "main"/"session" is
            # explicitly in the message.
            if (pattern['tool'] == 'vault_read'
                    and pattern.get('regex') is not None
                    and apple_si_hw
                    and not has_explicit_session_word):
                continue
            # Skip vault_read if message has web indicators (let search catch it)
            if pattern['tool'] == 'vault_read' and has_web_indicator:
                continue
            # M54 Phase 3: explicit _gate_no_web gating on fallback patterns
            # like the possessive-intent vault_research (our research,
            # our findings). These only route to vault when there's no
            # web indicator present.
            if pattern.get('_gate_no_web') and has_web_indicator:
                continue
            # Main 38 P2: X-intent queries must go to browse_x_feed.
            # Block browse_search, vault_read, AND vault_research when X
            # intent detected. Session 2 turn 4 regression: query
            # "search for the top 2 most interesting X posts that would
            # compliment our research" routed to vault_research because
            # "our research" matched its keyword list and X-intent only
            # blocked search + vault_read. Now also blocks vault_research.
            if has_x_intent and pattern['tool'] in (
                    'browse_search', 'vault_read', 'vault_research'):
                continue
            args = pattern['args'](message)
            return (pattern['tool'], args)

    # Main 38 Session 3 fix A: positive X-intent dispatcher. If the
    # query contains clear X/Twitter tokens but no keyword pattern
    # matched (typically because the phrasing uses verbs the keyword
    # list doesn't enumerate, e.g. "posted lately on X"), default to
    # browse_x_feed. Prevents L2 from mis-routing to vault_read on
    # X-specific queries just because L1 fell through. Session 3 turns
    # 2 and 10 both hit this pattern.
    if has_x_intent:
        return ('browse_x_feed', _x_feed_args(message))

    return None


TOOL_DESCRIPTIONS = """Available tools (use ONLY if the user's request requires one):

- vault_read: Read project docs/roadmap/status from the Obsidian vault. Args: {"path": "Roadmap.md"} or {"query": "search term"}. Use for: project status, what we're working on, roadmap, decisions, infrastructure, "how's it going", "what's new", status updates.
- vault_research: Deep search of ANE reverse engineering files, hardware measurements, opcode catalogs, agent reports. Args: {"query": "search term"}. Use for: specific hardware findings, opcodes, measurements, "what did we find", "what haven't we explored", research gaps, deep technical questions about ANE/hardware.
- vault_insight: Cross-reference vault docs and memory on a topic. Args: {"topic": "some topic"}. Use for: deep research on a topic across vault and memory.
- memory_recall: Search stored conversation memories. Args: {"query": "search term"}. Use for: "what's in memory about X", "do you remember X", past conversations.
- memory_ingest: Store something in memory. Args: {"role": "user", "text": "content to store"}. Use for: "remember this", "save this".
- memory_stats: Check memory statistics. Args: {}. Use for: "how many memories", "memory stats".
- browse_search: Search the GENERAL web via Google. Args: {"query": "search term"}. Use for: general web search, current events, news, stock prices, wiki-style lookups. DO NOT USE for anything mentioning X, Twitter, tweets, or X posts — use browse_x_feed for those.
- browse_x_feed: Scan X/Twitter feed for posts, threads, research discussions. Args: {"count": 5}. Use for any query mentioning X, Twitter, tweets, X posts, posts on X, search X, search on X, X research, X threads, "what's on X", "check X", "scan my feed", "interesting X posts", or any phrasing where the user asks about content FROM X rather than content ABOUT the letter X.
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


def route(message, llm_fn=None, prior_user_message: str = ""):
    """
    Main routing function.
    Returns (tool_name, args_dict) or ('conversation', {}).

    `prior_user_message` (Main 62 Bug 1): previous user turn, passed to
    layer1_route so anaphoric web-search follow-ups can fold the prior
    subject into the Google query.
    """
    # Layer 1: deterministic keyword routing (must run first — "remember that" etc.)
    result = layer1_route(message, prior_user_message=prior_user_message)
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
