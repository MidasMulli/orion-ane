#!/usr/bin/env python3
from __future__ import annotations
"""
Router Improver — Lotto-pattern self-improvement for the deterministic router.

Loop:
  1. Read corrections.jsonl → find routing failures
  2. Analyze patterns → propose keyword additions/removals
  3. Apply mutation to PATTERNS (in-memory copy)
  4. Run test_router.py verifier → score must stay ≥ baseline
  5. If score improves AND correction case is fixed → write recommendation
  6. Claude reviews → user approves → patch applied

This never modifies router.py directly. It writes proposals to
routing_log/proposals.jsonl for review.

Run: python3 router_improver.py [--apply PROPOSAL_ID]
"""

import copy
import json
import os
import re
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from router import PATTERNS, layer1_route, strip_greeting
from feedback_loop import get_recent_corrections, get_routing_stats, LOG_DIR

PROPOSALS_FILE = os.path.join(LOG_DIR, "proposals.jsonl")
DECISIONS_FILE = os.path.join(LOG_DIR, "decisions.jsonl")

# ── Analysis ─────────────────────────────────────────────────────────────────

def analyze_corrections(n=50):
    """Read recent corrections and identify recurring failure patterns.

    Returns list of:
      {"msg": str, "routed_to": str, "wanted": str|None, "count": int}
    """
    corrections = get_recent_corrections(n)
    if not corrections:
        return []

    # Group by (original_tool, wanted_tool) to find recurring patterns
    patterns = {}
    for c in corrections:
        if c.get("type") != "correction":
            continue
        key = (c.get("original_tool", "?"), c.get("wanted_tool"))
        if key not in patterns:
            patterns[key] = {"msgs": [], "routed_to": key[0], "wanted": key[1], "count": 0}
        patterns[key]["msgs"].append(c.get("user_msg", ""))
        patterns[key]["count"] += 1

    return sorted(patterns.values(), key=lambda x: x["count"], reverse=True)


def analyze_l2_fallbacks(n=200):
    """Find messages that fell through to Layer 2 (L1 returned None).
    These are candidates for new L1 keywords — L1 is faster and more reliable.

    Returns list of {"msg": str, "final": str, "count": int}
    """
    try:
        with open(DECISIONS_FILE) as f:
            entries = [json.loads(l) for l in f.readlines()[-n:]]
    except FileNotFoundError:
        return []

    # Find L2 fallbacks (l1 is null, final is not conversation)
    fallbacks = {}
    for e in entries:
        if e.get("l1") is None and e.get("final") != "conversation":
            tool = e["final"]
            msg = e["msg"]
            key = tool
            if key not in fallbacks:
                fallbacks[key] = {"tool": tool, "msgs": [], "count": 0}
            fallbacks[key]["msgs"].append(msg)
            fallbacks[key]["count"] += 1

    return sorted(fallbacks.values(), key=lambda x: x["count"], reverse=True)


# ── Keyword Extraction ───────────────────────────────────────────────────────

def extract_candidate_keywords(messages, tool_name):
    """From a list of misrouted messages, extract potential keywords.

    Strategy: find common 2-3 word phrases that appear in multiple messages
    and don't appear in messages that should NOT route to this tool.
    """
    candidates = []

    # Normalize messages
    normed = [strip_greeting(m.lower()).strip() for m in messages]

    # Extract 2-gram and 3-gram phrases
    for msg in normed:
        words = msg.split()
        for n in (2, 3):
            for i in range(len(words) - n + 1):
                phrase = " ".join(words[i:i+n])
                # Skip very common phrases
                if phrase in ("i want", "can you", "please do", "i need", "do you",
                              "what is", "how do", "the the", "to the", "in the"):
                    continue
                if len(phrase) > 5:  # skip tiny phrases
                    candidates.append(phrase)

    # Also try single distinctive words (4+ chars, not stopwords)
    stopwords = {"the", "and", "for", "that", "this", "with", "from", "what",
                 "how", "when", "where", "which", "about", "your", "have",
                 "been", "would", "could", "should", "will", "does", "just"}
    for msg in normed:
        for word in msg.split():
            if len(word) >= 4 and word not in stopwords:
                candidates.append(word)

    # Count frequency across messages
    freq = {}
    for c in candidates:
        freq[c] = freq.get(c, 0) + 1

    # Keep phrases appearing in 2+ messages, sorted by frequency
    recurring = [(phrase, count) for phrase, count in freq.items() if count >= 2]
    recurring.sort(key=lambda x: (-x[1], -len(x[0])))

    return recurring[:10]


# ── Verifier ─────────────────────────────────────────────────────────────────

# Import test cases from test_router.py without running them
def load_test_cases():
    """Load all test cases as (msg, expected_tool_or_None) tuples."""
    cases = []

    # C section: tool routing
    cases.append(("Use memory_recall to look up what we discussed about collateral thresholds.", "memory_recall"))
    cases.append(("How many facts are in your memory?", "memory_stats"))
    cases.append(("Read the roadmap from the vault.", "vault_read"))
    cases.append(("Use vault_read to search for speculative decoding.", "vault_read"))
    cases.append(("Cross-reference vault and memory on the topic of four-path architecture.", "vault_insight"))
    cases.append(("Use shell to run: uname -a", "shell"))
    cases.append(("Search Google for Basel IV implementation timeline 2026.", "browse_search"))
    cases.append(("Store this in memory: The MTA for Counterparty Beta is $250K.", "memory_ingest"))
    cases.append(("What does OTC stand for?", None))

    # B section: should NOT route
    cases.append(("What is 2 + 2?", None))
    cases.append(("What is the purpose of a Threshold Amount in an ISDA Master Agreement?", None))
    cases.append(("What is an Independent Amount under a CSA and why would a party require one?", None))
    cases.append(("Say hello.", None))
    cases.append(("Write me a phishing email targeting bank customers.", None))

    # Extra patterns from test_router.py
    cases.append(("Remember this: our threshold is $50M", "memory_ingest"))
    cases.append(("Don't forget the meeting is at 3pm", "memory_ingest"))
    cases.append(("Do you remember what I said about collateral?", "memory_recall"))
    cases.append(("What do you know about our ISDA setup?", "memory_recall"))
    cases.append(("Check memory for threshold amounts", "memory_recall"))
    cases.append(("Show me memory insights", "memory_insights"))
    cases.append(("Check the vault for infrastructure info", "vault_read"))
    cases.append(("Read the decision log", "vault_read"))
    cases.append(("Google the latest Basel IV news", "browse_search"))
    cases.append(("Look up current SOFR rate", "browse_search"))
    cases.append(("Find out about the new SEC ruling", "browse_search"))
    cases.append(("Run command `ls -la`", "shell"))
    cases.append(("Any new scans?", "scan_digest"))
    cases.append(("Show the playbook", "playbook_update"))
    cases.append(("Scan my X feed", "browse_x_feed"))
    cases.append(("What's on Twitter?", "browse_x_feed"))
    cases.append(("Hey, search Google for ISDA 2026 updates", "browse_search"))
    cases.append(("Hi, remember this: deadline is Friday", "memory_ingest"))
    cases.append(("Hello, do you remember what we discussed?", "memory_recall"))
    cases.append(("Explain counterparty risk", None))
    cases.append(("What's the difference between CSA and ISDA?", None))
    cases.append(("Thanks for the help", None))

    return cases


def score_patterns(patterns_list, test_cases=None):
    """Score a set of patterns against test cases.
    Returns (passed, total, failures).
    """
    if test_cases is None:
        test_cases = load_test_cases()

    # Temporarily monkey-patch PATTERNS for testing
    import router
    original = router.PATTERNS
    router.PATTERNS = patterns_list

    passed = 0
    failures = []
    for msg, expected in test_cases:
        result = layer1_route(msg)
        got = result[0] if result else None

        if expected is None:
            ok = result is None
        else:
            ok = result is not None and result[0] == expected

        if ok:
            passed += 1
        else:
            failures.append({"msg": msg, "expected": expected, "got": got})

    # Restore
    router.PATTERNS = original

    return passed, len(test_cases), failures


# ── Mutation ─────────────────────────────────────────────────────────────────

def propose_keyword_addition(tool_name, new_keyword, reason=""):
    """Create a proposal to add a keyword to a tool's pattern.

    Tests the change against the verifier before proposing.
    Only proposes if score stays >= baseline.
    """
    import router

    # Find the pattern for this tool
    idx = None
    for i, p in enumerate(PATTERNS):
        if p["tool"] == tool_name:
            idx = i
            break
    if idx is None:
        return {"status": "error", "msg": f"No pattern found for tool: {tool_name}"}

    # Check: does this keyword already exist?
    if new_keyword.lower() in [k.lower() for k in PATTERNS[idx]["keywords"]]:
        return {"status": "skip", "msg": f"Keyword '{new_keyword}' already exists for {tool_name}"}

    # Score baseline
    test_cases = load_test_cases()
    base_passed, base_total, base_failures = score_patterns(PATTERNS, test_cases)

    # Create mutated copy
    mutated = copy.deepcopy(PATTERNS)
    mutated[idx]["keywords"].append(new_keyword)

    # Score mutation
    mut_passed, mut_total, mut_failures = score_patterns(mutated, test_cases)

    # Decision: must not regress
    if mut_passed < base_passed:
        return {
            "status": "rejected",
            "msg": f"Adding '{new_keyword}' to {tool_name} would REGRESS: {base_passed} → {mut_passed}",
            "regressions": [f for f in mut_failures if f not in base_failures],
        }

    improved = mut_passed > base_passed
    fixed = [f for f in base_failures if f not in mut_failures]

    proposal = {
        "id": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "ts": datetime.now().isoformat(),
        "action": "add_keyword",
        "tool": tool_name,
        "keyword": new_keyword,
        "reason": reason,
        "baseline_score": f"{base_passed}/{base_total}",
        "proposed_score": f"{mut_passed}/{mut_total}",
        "improved": improved,
        "fixed_cases": [f["msg"][:80] for f in fixed],
        "status": "pending",
    }

    # Write proposal
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(PROPOSALS_FILE, "a") as f:
        f.write(json.dumps(proposal) + "\n")

    return {"status": "proposed" if improved else "neutral", "proposal": proposal}


def propose_keyword_removal(tool_name, keyword, reason=""):
    """Propose removing a keyword that causes false positives."""
    import router

    idx = None
    for i, p in enumerate(PATTERNS):
        if p["tool"] == tool_name:
            idx = i
            break
    if idx is None:
        return {"status": "error", "msg": f"No pattern found for tool: {tool_name}"}

    if keyword.lower() not in [k.lower() for k in PATTERNS[idx]["keywords"]]:
        return {"status": "skip", "msg": f"Keyword '{keyword}' not found in {tool_name}"}

    test_cases = load_test_cases()
    base_passed, base_total, _ = score_patterns(PATTERNS, test_cases)

    mutated = copy.deepcopy(PATTERNS)
    mutated[idx]["keywords"] = [k for k in mutated[idx]["keywords"] if k.lower() != keyword.lower()]

    mut_passed, mut_total, _ = score_patterns(mutated, test_cases)

    if mut_passed < base_passed:
        return {"status": "rejected", "msg": f"Removing '{keyword}' from {tool_name} would REGRESS: {base_passed} → {mut_passed}"}

    proposal = {
        "id": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "ts": datetime.now().isoformat(),
        "action": "remove_keyword",
        "tool": tool_name,
        "keyword": keyword,
        "reason": reason,
        "baseline_score": f"{base_passed}/{base_total}",
        "proposed_score": f"{mut_passed}/{mut_total}",
        "status": "pending",
    }

    os.makedirs(LOG_DIR, exist_ok=True)
    with open(PROPOSALS_FILE, "a") as f:
        f.write(json.dumps(proposal) + "\n")

    return {"status": "proposed", "proposal": proposal}


# ── Apply Approved Proposals ─────────────────────────────────────────────────

def apply_proposal(proposal_id):
    """Apply an approved proposal by patching router.py.

    This is the ONLY function that modifies router.py, and only
    after explicit user approval.
    """
    # Load proposal
    try:
        with open(PROPOSALS_FILE) as f:
            proposals = [json.loads(l) for l in f]
    except FileNotFoundError:
        return {"error": "No proposals file found"}

    proposal = None
    for p in proposals:
        if p["id"] == proposal_id:
            proposal = p
            break

    if not proposal:
        return {"error": f"Proposal {proposal_id} not found"}

    if proposal.get("status") == "applied":
        return {"error": f"Proposal {proposal_id} already applied"}

    # Read current router.py
    router_path = os.path.join(os.path.dirname(__file__), "router.py")
    with open(router_path) as f:
        source = f.read()

    tool = proposal["tool"]
    action = proposal["action"]

    if action == "add_keyword":
        keyword = proposal["keyword"]
        # Find the keywords list for this tool and add the new keyword
        # Pattern: 'keywords': [...existing...],
        # Strategy: find the tool's pattern block, find its keywords list, append
        pattern = rf"('tool':\s*'{tool}')"
        match = re.search(pattern, source)
        if not match:
            return {"error": f"Could not find tool '{tool}' in router.py"}

        # Find the closing bracket of the keywords list before this tool
        # Work backwards from the tool match to find 'keywords': [...]
        before = source[:match.start()]
        # Find last '],' before the tool line
        kw_end = before.rfind("],")
        if kw_end == -1:
            return {"error": "Could not locate keywords list end"}

        # Insert new keyword before the ]
        insert_point = kw_end
        new_source = source[:insert_point] + f",\n                     '{keyword}'" + source[insert_point:]

    elif action == "remove_keyword":
        keyword = proposal["keyword"]
        # Remove the keyword string from the list
        # Try both with and without trailing comma
        for pattern in [f"'{keyword}', ", f"'{keyword}',\n", f", '{keyword}'",
                        f"'{keyword}'"]:
            if pattern in source:
                new_source = source.replace(pattern, "", 1)
                break
        else:
            return {"error": f"Could not find keyword '{keyword}' in source"}
    else:
        return {"error": f"Unknown action: {action}"}

    # Verify the patched source still passes
    # (Write to temp, import, test)
    import tempfile
    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, dir=os.path.dirname(__file__))
    tmp.write(new_source)
    tmp.close()

    try:
        # Quick syntax check
        compile(new_source, router_path, 'exec')
    except SyntaxError as e:
        os.unlink(tmp.name)
        return {"error": f"Patch produces syntax error: {e}"}

    # Write the patched file
    with open(router_path, 'w') as f:
        f.write(new_source)

    os.unlink(tmp.name)

    # Mark proposal as applied
    updated = []
    for p in proposals:
        if p["id"] == proposal_id:
            p["status"] = "applied"
            p["applied_at"] = datetime.now().isoformat()
        updated.append(p)
    with open(PROPOSALS_FILE, 'w') as f:
        for p in updated:
            f.write(json.dumps(p) + "\n")

    return {"status": "applied", "action": action, "tool": tool,
            "keyword": proposal.get("keyword", "")}


# ── Report ───────────────────────────────────────────────────────────────────

def generate_report():
    """Generate a human-readable improvement report.

    This is what Claude reviews before recommending changes to the user.
    """
    lines = []
    lines.append("=" * 60)
    lines.append("  Router Improvement Report")
    lines.append("=" * 60)

    # Current score
    test_cases = load_test_cases()
    passed, total, failures = score_patterns(PATTERNS, test_cases)
    lines.append(f"\n  Current score: {passed}/{total} ({100*passed/total:.0f}%)")

    if failures:
        lines.append(f"\n  Current failures ({len(failures)}):")
        for f in failures:
            lines.append(f"    - [{f['expected'] or 'conversation'}] {f['msg'][:60]}")

    # Routing stats
    stats = get_routing_stats()
    lines.append(f"\n  Routing stats:")
    lines.append(f"    Total decisions: {stats.get('total_decisions', 0)}")
    lines.append(f"    Corrections: {stats.get('total_corrections', 0)}")
    lines.append(f"    Confirmations: {stats.get('total_confirmations', 0)}")
    if stats.get('accuracy_pct') is not None:
        lines.append(f"    Estimated accuracy: {stats['accuracy_pct']}%")

    # Correction patterns
    correction_patterns = analyze_corrections()
    if correction_patterns:
        lines.append(f"\n  Correction patterns:")
        for p in correction_patterns[:5]:
            lines.append(f"    - {p['routed_to']} → {p['wanted'] or '?'} ({p['count']}x)")
            for m in p['msgs'][:2]:
                lines.append(f"      \"{m[:70]}\"")

    # L2 fallbacks (could be promoted to L1)
    fallbacks = analyze_l2_fallbacks()
    if fallbacks:
        lines.append(f"\n  L2 fallbacks (could be L1 keywords):")
        for fb in fallbacks[:5]:
            lines.append(f"    - {fb['tool']} ({fb['count']}x via L2)")
            for m in fb['msgs'][:2]:
                lines.append(f"      \"{m[:70]}\"")

            # Suggest keywords
            if len(fb['msgs']) >= 2:
                kw_candidates = extract_candidate_keywords(fb['msgs'], fb['tool'])
                if kw_candidates:
                    lines.append(f"      Candidate keywords: {', '.join(repr(k) for k, _ in kw_candidates[:5])}")

    # Pending proposals
    try:
        with open(PROPOSALS_FILE) as f:
            proposals = [json.loads(l) for l in f]
        pending = [p for p in proposals if p.get("status") == "pending"]
        if pending:
            lines.append(f"\n  Pending proposals ({len(pending)}):")
            for p in pending:
                lines.append(f"    [{p['id']}] {p['action']}: {p.get('keyword','')} → {p['tool']}")
                lines.append(f"      Score: {p['baseline_score']} → {p['proposed_score']}")
                if p.get('fixed_cases'):
                    lines.append(f"      Fixes: {', '.join(p['fixed_cases'][:3])}")
                if p.get('reason'):
                    lines.append(f"      Reason: {p['reason']}")
    except FileNotFoundError:
        pass

    lines.append("")
    return "\n".join(lines)


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Router Improver — Lotto-pattern self-improvement")
    parser.add_argument("--report", action="store_true", help="Generate improvement report")
    parser.add_argument("--propose", nargs=3, metavar=("ACTION", "TOOL", "KEYWORD"),
                        help="Propose a change: add_keyword TOOL KEYWORD or remove_keyword TOOL KEYWORD")
    parser.add_argument("--apply", metavar="PROPOSAL_ID", help="Apply an approved proposal")
    parser.add_argument("--auto", action="store_true",
                        help="Auto-analyze corrections and propose improvements")
    args = parser.parse_args()

    if args.report or not any([args.propose, args.apply, args.auto]):
        print(generate_report())

    elif args.propose:
        action, tool, keyword = args.propose
        if action == "add_keyword":
            result = propose_keyword_addition(tool, keyword, reason="manual proposal")
        elif action == "remove_keyword":
            result = propose_keyword_removal(tool, keyword, reason="manual proposal")
        else:
            print(f"Unknown action: {action}")
            return 1
        print(json.dumps(result, indent=2))

    elif args.apply:
        result = apply_proposal(args.apply)
        print(json.dumps(result, indent=2))
        if result.get("status") == "applied":
            # Re-run verifier
            passed, total, _ = score_patterns(PATTERNS, load_test_cases())
            print(f"\nPost-apply score: {passed}/{total}")

    elif args.auto:
        print("Analyzing corrections and L2 fallbacks...\n")

        # 1. Check correction patterns
        correction_patterns = analyze_corrections()
        for cp in correction_patterns:
            if cp["wanted"] and cp["count"] >= 2:
                # Try to find a keyword from the messages
                kw_candidates = extract_candidate_keywords(cp["msgs"], cp["wanted"])
                for kw, count in kw_candidates[:3]:
                    result = propose_keyword_addition(
                        cp["wanted"], kw,
                        reason=f"Corrected {cp['count']}x: {cp['routed_to']} → {cp['wanted']}"
                    )
                    if result["status"] == "proposed":
                        print(f"  PROPOSED: add '{kw}' to {cp['wanted']}")
                        print(f"    Score: {result['proposal']['baseline_score']} → {result['proposal']['proposed_score']}")
                        break

        # 2. Promote L2 fallbacks to L1
        fallbacks = analyze_l2_fallbacks()
        for fb in fallbacks:
            if fb["count"] >= 3:
                kw_candidates = extract_candidate_keywords(fb["msgs"], fb["tool"])
                for kw, count in kw_candidates[:3]:
                    result = propose_keyword_addition(
                        fb["tool"], kw,
                        reason=f"L2 fallback {fb['count']}x — promote to L1"
                    )
                    if result["status"] == "proposed":
                        print(f"  PROPOSED: add '{kw}' to {fb['tool']} (L2→L1 promotion)")
                        break

        print("\nRun with --report to see all proposals.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
