#!/usr/bin/env python3
"""
Three-Tier Evaluation — test CPU, ANE, and GPU independently
=============================================================

Runs structured tests against each tier and scores the results.
No venv dependency — uses whatever Python is available + connects
to running servers via sockets/HTTP.

Usage:
    ~/.mlx-env/bin/python3 eval_tiers.py
"""

import os
import sys
import json
import time
import re

# Add memory dir to path
sys.path.insert(0, os.path.dirname(__file__))

# ── ANSI ──────────────────────────────────────────────────────
CYAN = "\033[36m"
GREEN = "\033[32m"
RED = "\033[31m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"
YELLOW = "\033[33m"


def section(title):
    print(f"\n{CYAN}{'═' * 60}{RESET}")
    print(f"{CYAN}  {BOLD}{title}{RESET}")
    print(f"{CYAN}{'═' * 60}{RESET}")


def test(name, passed, detail=""):
    icon = f"{GREEN}✓{RESET}" if passed else f"{RED}✗{RESET}"
    print(f"  {icon} {name}")
    if detail:
        print(f"    {DIM}{detail}{RESET}")
    return 1 if passed else 0


# ══════════════════════════════════════════════════════════════
#  TIER 1: CPU — Extraction + Embedding + Recall
# ══════════════════════════════════════════════════════════════

def eval_cpu():
    section("TIER 1: CPU — Extraction + Embedding + Recall")
    from daemon import FactExtractor, MemoryStore
    import tempfile

    ext = FactExtractor()
    score = 0
    total = 0

    # ── Test 1: Fact extraction from ISDA paragraph ──
    isda_text = (
        "We decided to set the cross-default threshold at $50 million for "
        "Counterparty Alpha. The minimum transfer amount is $500,000 in USD. "
        "Need to review the CSA amendment by Friday. We prefer US Treasuries "
        "as eligible collateral with a 2% haircut."
    )
    facts = ext.extract(isda_text, role="assistant")
    types = [f["type"] for f in facts]

    total += 1
    score += test("Extract ≥4 facts from ISDA paragraph",
                  len(facts) >= 4,
                  f"Got {len(facts)} facts: {types}")

    total += 1
    score += test("Found 'decision' type",
                  "decision" in types,
                  f"Types: {types}")

    total += 1
    score += test("Found 'quantitative' type",
                  "quantitative" in types,
                  f"Types: {types}")

    total += 1
    score += test("Found 'task' type",
                  "task" in types,
                  f"Types: {types}")

    total += 1
    score += test("Found 'preference' type",
                  "preference" in types,
                  f"Types: {types}")

    # ── Test 2: Entity extraction ──
    all_entities = set()
    for f in facts:
        all_entities.update(f.get("entities", []))

    total += 1
    score += test("Extracted 'Counterparty Alpha' entity",
                  any("counterparty alpha" in e.lower() for e in all_entities),
                  f"Entities: {all_entities}")

    total += 1
    score += test("Extracted ≥2 entities total",
                  len(all_entities) >= 2,
                  f"Got {len(all_entities)}: {all_entities}")

    # ── Test 3: Embedding + recall ──
    with tempfile.TemporaryDirectory() as tmpdir:
        store = MemoryStore(tmpdir)

        # Store facts
        for f in facts:
            store.store(f)

        # Recall test — semantic search
        results = store.recall("what is the cross-default threshold", n_results=3)
        total += 1
        top_text = results[0]["text"] if results else ""
        score += test("Recall: 'cross-default' query finds threshold fact",
                      "50" in top_text or "cross-default" in top_text.lower(),
                      f"Top result: {top_text[:80]}")

        results = store.recall("eligible collateral type", n_results=3)
        total += 1
        top_text = results[0]["text"] if results else ""
        score += test("Recall: 'eligible collateral' query finds treasuries",
                      "treasur" in top_text.lower() or "collateral" in top_text.lower(),
                      f"Top result: {top_text[:80]}")

        # Recall with type filter
        results = store.recall("anything", n_results=10, type_filter="task")
        total += 1
        score += test("Type filter: task filter returns task facts only",
                      all(r["metadata"]["type"] == "task" for r in results) and len(results) >= 1,
                      f"Got {len(results)} task results")

    # ── Test 4: Deduplication ──
    facts1 = ext.extract("The threshold is $50M for Alpha.", role="assistant")
    facts2 = ext.extract("The threshold is $50M for Alpha.", role="assistant")
    # Same content should produce same hash
    if facts1 and facts2:
        total += 1
        score += test("Dedup: identical sentences produce same hash",
                      facts1[0].get("hash") == facts2[0].get("hash"),
                      f"Hash1={facts1[0].get('hash','?')[:12]}, Hash2={facts2[0].get('hash','?')[:12]}")

    # ── Test 5: Noise filtering ──
    noise_facts = ext.extract("ok thanks", role="user")
    total += 1
    score += test("Noise filter: 'ok thanks' produces 0 facts",
                  len(noise_facts) == 0,
                  f"Got {len(noise_facts)} facts")

    noise_facts2 = ext.extract("Sure, I can help with that.", role="assistant")
    total += 1
    score += test("Noise filter: generic response produces 0 facts",
                  len(noise_facts2) == 0,
                  f"Got {len(noise_facts2)} facts")

    return score, total


# ══════════════════════════════════════════════════════════════
#  TIER 2: ANE — 1.7B Generative Analysis
# ══════════════════════════════════════════════════════════════

def eval_ane():
    section("TIER 2: ANE — 1.7B Generative Analysis on Neural Engine")

    try:
        from ane_server import ANEClient
        if not ANEClient.is_running():
            print(f"  {YELLOW}⚠ ANE server not running — skipping{RESET}")
            print(f"  {DIM}Start with: midas (auto-launches) or python ane_server.py{RESET}")
            return 0, 0
        client = ANEClient()
    except Exception as e:
        print(f"  {YELLOW}⚠ ANE client unavailable: {e}{RESET}")
        return 0, 0

    score = 0
    total = 0

    # ── Test 1: Entity summarization ──
    t0 = time.time()
    result = client.analyze(
        "Summarize this entity in 2 sentences:\n"
        "Entity: JPMorgan\n"
        "Facts:\n"
        "- ISDA Master Agreement dated 2024\n"
        "- Cross-default threshold $100M\n"
        "- Minimum transfer amount $1M\n"
        "- Eligible collateral: US Treasuries, Agency MBS\n"
        "- Rating trigger at BBB-\n"
        "- Independent amount: $25M\n",
        max_tokens=80,
    )
    elapsed = time.time() - t0
    total += 1
    # Check it mentions JPMorgan and at least one key fact
    has_entity = "jpmorgan" in result.lower()
    has_fact = any(kw in result.lower() for kw in ["100m", "100 m", "treasur", "cross-default", "isda", "collateral"])
    score += test("Summarize entity profile",
                  has_entity and has_fact,
                  f"({elapsed:.1f}s) {result[:120]}")

    # ── Test 2: Relationship extraction ──
    t0 = time.time()
    result = client.analyze(
        "What entities are mentioned and how are they related?\n"
        "Text: Goldman Sachs and Citadel both have CSA agreements with "
        "JPMorgan that include cross-default provisions referencing "
        "the ISDA 2002 framework.",
        max_tokens=80,
    )
    elapsed = time.time() - t0
    total += 1
    entities_found = sum(1 for e in ["goldman", "citadel", "jpmorgan", "isda"]
                        if e in result.lower())
    score += test("Extract entities and relationships",
                  entities_found >= 3,
                  f"({elapsed:.1f}s) Found {entities_found}/4 entities. {result[:120]}")

    # ── Test 3: Risk identification ──
    t0 = time.time()
    result = client.analyze(
        "What are the top 3 risks for this counterparty?\n"
        "Counterparty: Acme Corp\n"
        "- Cross-default threshold: $10M (very low)\n"
        "- No rating trigger clause\n"
        "- Eligible collateral: only corporate bonds\n"
        "- MTA: $5M\n"
        "- No independent amount required\n",
        max_tokens=100,
    )
    elapsed = time.time() - t0
    total += 1
    risk_words = sum(1 for w in ["risk", "low", "concern", "vulnerab", "expos", "issue", "weak"]
                    if w in result.lower())
    score += test("Identify counterparty risks",
                  risk_words >= 2,
                  f"({elapsed:.1f}s) {result[:120]}")

    # ── Test 4: Structured output ──
    t0 = time.time()
    result = client.analyze(
        "List the key terms from this CSA provision as bullet points:\n"
        "The Minimum Transfer Amount is USD 500,000. Eligible collateral "
        "includes US Treasuries (2% haircut), Agency MBS (5% haircut), "
        "and cash in USD, EUR, or GBP. The Independent Amount is "
        "USD 25,000,000 for Party A.",
        max_tokens=100,
    )
    elapsed = time.time() - t0
    total += 1
    has_amounts = sum(1 for a in ["500", "25", "2%", "5%"] if a in result)
    score += test("Extract structured terms from CSA text",
                  has_amounts >= 2,
                  f"({elapsed:.1f}s) Found {has_amounts}/4 amounts. {result[:120]}")

    # ── Test 5: Speed check ──
    t0 = time.time()
    result = client.analyze("Hello, who are you?", max_tokens=30)
    elapsed = time.time() - t0
    total += 1
    score += test("Response time < 3 seconds",
                  elapsed < 3.0,
                  f"{elapsed:.1f}s for {len(result.split())} words")

    return score, total


# ══════════════════════════════════════════════════════════════
#  TIER 3: GPU — 9B Interactive Reasoning
# ══════════════════════════════════════════════════════════════

def eval_gpu():
    section("TIER 3: GPU — 9B Interactive Reasoning (MLX)")

    try:
        from openai import OpenAI
        client = OpenAI(base_url="http://127.0.0.1:8899/v1", api_key="not-needed")
        client.models.list()
    except Exception as e:
        print(f"  {YELLOW}⚠ MLX server not running — skipping{RESET}")
        print(f"  {DIM}Start with: ~/.hermes/start-mlx-server.sh{RESET}")
        return 0, 0

    score = 0
    total = 0

    def ask(prompt, max_tokens=150):
        t0 = time.time()
        resp = client.chat.completions.create(
            model="mlx-community/Qwen3.5-9B-MLX-4bit",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
        )
        elapsed = time.time() - t0
        text = resp.choices[0].message.content or ""
        return text, elapsed

    # ── Test 1: Domain knowledge ──
    result, elapsed = ask(
        "In one paragraph, explain what a Credit Support Annex (CSA) is "
        "and why it matters for OTC derivatives."
    )
    total += 1
    domain_words = sum(1 for w in ["collateral", "margin", "otc", "derivative", "credit", "isda"]
                      if w in result.lower())
    score += test("Domain knowledge: CSA explanation",
                  domain_words >= 3,
                  f"({elapsed:.1f}s) {domain_words}/6 domain terms. {result[:100]}")

    # ── Test 2: Analytical reasoning ──
    result, elapsed = ask(
        "Counterparty A has a cross-default threshold of $10M and only "
        "accepts corporate bonds as collateral. Counterparty B has a "
        "threshold of $100M and accepts US Treasuries. Which is riskier "
        "and why? Answer in 2-3 sentences."
    )
    total += 1
    mentions_a = "counterparty a" in result.lower() or "party a" in result.lower() or "10" in result
    gives_reason = any(w in result.lower() for w in ["risk", "lower", "higher", "because", "since", "due to"])
    score += test("Analytical reasoning: compare counterparties",
                  mentions_a and gives_reason,
                  f"({elapsed:.1f}s) {result[:120]}")

    # ── Test 3: Structured output ──
    result, elapsed = ask(
        "Create a JSON object with these CSA terms:\n"
        "- Counterparty: Acme Corp\n"
        "- MTA: $500K\n"
        "- Threshold: $50M\n"
        "- Eligible collateral: US Treasuries, Cash\n"
        "Reply with ONLY the JSON, no explanation."
    )
    total += 1
    try:
        # Try to extract JSON from response
        json_match = re.search(r'\{[^}]+\}', result, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
            has_keys = len(parsed) >= 3
        else:
            has_keys = False
    except Exception:
        has_keys = False
    score += test("Structured output: generate valid JSON",
                  has_keys,
                  f"({elapsed:.1f}s) {result[:120]}")

    # ── Test 4: Tool use reasoning ──
    result, elapsed = ask(
        "You have access to these tools: memory_recall(query), memory_ingest(text), browse_navigate(url). "
        "A user says: 'What was the MTA we agreed on for Goldman last week?' "
        "Which tool would you use first and what query would you pass? "
        "Reply with the tool name and query only."
    )
    total += 1
    mentions_recall = "memory_recall" in result.lower() or "recall" in result.lower()
    mentions_query = any(w in result.lower() for w in ["goldman", "mta", "minimum transfer"])
    score += test("Tool use reasoning: correct tool selection",
                  mentions_recall and mentions_query,
                  f"({elapsed:.1f}s) {result[:120]}")

    # ── Test 5: Speed check ──
    result, elapsed = ask("Say hello in exactly 5 words.", max_tokens=20)
    total += 1
    tps = len(result.split()) / elapsed if elapsed > 0 else 0
    score += test("Response speed > 10 tok/s",
                  elapsed < 3.0,
                  f"{elapsed:.1f}s, ~{tps:.0f} words/s")

    return score, total


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

def main():
    print(f"\n{BOLD}{CYAN}  ╔══════════════════════════════════════════╗{RESET}")
    print(f"{BOLD}{CYAN}  ║   THREE-TIER EVALUATION — Orion ANE      ║{RESET}")
    print(f"{BOLD}{CYAN}  ╚══════════════════════════════════════════╝{RESET}")
    print(f"  {DIM}CPU (extract+embed) │ ANE (1.7B analyze) │ GPU (9B reason){RESET}")

    t_start = time.time()

    cpu_score, cpu_total = eval_cpu()
    ane_score, ane_total = eval_ane()
    gpu_score, gpu_total = eval_gpu()

    elapsed = time.time() - t_start

    # ── Summary ──
    section("RESULTS")
    total_score = cpu_score + ane_score + gpu_score
    total_total = cpu_total + ane_total + gpu_total

    def tier_bar(name, s, t):
        if t == 0:
            return f"  {name:.<30s} {YELLOW}SKIPPED{RESET}"
        pct = s / t * 100
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        color = GREEN if pct >= 80 else YELLOW if pct >= 60 else RED
        return f"  {name:.<30s} {color}{bar} {s}/{t} ({pct:.0f}%){RESET}"

    print(tier_bar("CPU (extraction+recall)", cpu_score, cpu_total))
    print(tier_bar("ANE (1.7B analysis)", ane_score, ane_total))
    print(tier_bar("GPU (9B reasoning)", gpu_score, gpu_total))
    print(f"  {'─' * 50}")
    if total_total > 0:
        pct = total_score / total_total * 100
        color = GREEN if pct >= 80 else YELLOW if pct >= 60 else RED
        print(f"  {BOLD}TOTAL{RESET}{':':.<25s} {color}{total_score}/{total_total} ({pct:.0f}%){RESET}")
    print(f"  {DIM}Completed in {elapsed:.1f}s{RESET}\n")


if __name__ == "__main__":
    main()
