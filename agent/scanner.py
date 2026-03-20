#!/usr/bin/env python3
"""
Scanner — Multi-source feed scanner for Midas agent
=====================================================

Two tiers of sources:
  Tier 1 (API):     Hacker News, RSS feeds — always available, no auth
  Tier 2 (Browser): X, Reddit — requires Chrome CDP, opportunistic

Writes candidate JSON files to vault/midas/scans/candidates/.
Reads verdict JSON files from vault/midas/scans/verdicts/ for self-calibration.

Usage:
    from scanner import Scanner
    s = Scanner()
    s.run_cycle()           # Scan all sources, write candidates
    s.read_verdicts()       # Read Claude's verdicts for learning
"""

import json
import os
import re
import time
import urllib.request
import urllib.error
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from typing import Optional

# ── Config ──────────────────────────────────────────────────────────────────

VAULT_PATH = "/Users/midas/Desktop/cowork/vault"
CANDIDATES_DIR = os.path.join(VAULT_PATH, "midas/scans/candidates")
VERDICTS_DIR = os.path.join(VAULT_PATH, "midas/scans/verdicts")

# Hacker News API
HN_API = "https://hacker-news.firebaseio.com/v0"
HN_TOP_STORIES = f"{HN_API}/topstories.json"
HN_ITEM = f"{HN_API}/item/{{id}}.json"
HN_MIN_SCORE = 30  # Minimum points to consider
HN_MAX_ITEMS = 50  # Check top N stories

# RSS feeds — high signal, no auth
RSS_FEEDS = {
    "huggingface": "https://huggingface.co/blog/feed.xml",
    "simon_willison": "https://simonwillison.net/atom/everything/",
    "interconnects": "https://www.interconnects.ai/feed",
    "mlx_blog": "https://ml-explore.github.io/mlx/feed.xml",
}

# Reddit subreddits (via old.reddit.com RSS — no auth needed!)
REDDIT_RSS = {
    "LocalLLaMA": "https://old.reddit.com/r/LocalLLaMA/top/.rss?t=day",
    "MachineLearning": "https://old.reddit.com/r/MachineLearning/top/.rss?t=day",
}

# Keywords ranked by relevance to the user's interests
PRIMARY_KEYWORDS = [
    "mlx", "apple silicon", "local llm", "on-device", "neural engine",
    "ane", "quantization", "gguf", "speculative decoding", "local inference",
    "phantom", "mcp", "tool use", "function calling",
    "coreml", "metal", "unified memory", "m4", "m5",
]
SECONDARY_KEYWORDS = [
    "isda", "derivatives", "collateral", "regulatory", "banking ai",
    "fintech", "compliance", "settlement", "netting",
]
INFRA_KEYWORDS = [
    "agent", "rag", "embedding", "vector", "chromadb", "obsidian",
    "ollama", "llama.cpp", "vllm", "lmstudio", "openai compatible",
    "claude", "anthropic", "agentic",
]

ALL_KEYWORDS = PRIMARY_KEYWORDS + SECONDARY_KEYWORDS + INFRA_KEYWORDS


def score_relevance(title: str, text: str = "") -> float:
    """Score how relevant an item is to the user's interests. 0-1 scale."""
    combined = (title + " " + text).lower()
    score = 0.0
    for kw in PRIMARY_KEYWORDS:
        if kw in combined:
            score += 0.15
    for kw in SECONDARY_KEYWORDS:
        if kw in combined:
            score += 0.12
    for kw in INFRA_KEYWORDS:
        if kw in combined:
            score += 0.08
    return min(score, 1.0)


def fetch_json(url: str, timeout: int = 10) -> Optional[dict]:
    """Fetch JSON from a URL, return None on failure."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "PhantomScanner/0.1"})
        resp = urllib.request.urlopen(req, timeout=timeout)
        return json.loads(resp.read())
    except Exception:
        return None


def fetch_text(url: str, timeout: int = 10) -> Optional[str]:
    """Fetch raw text from a URL, return None on failure."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "PhantomScanner/0.1"})
        resp = urllib.request.urlopen(req, timeout=timeout)
        return resp.read().decode("utf-8", errors="replace")
    except Exception:
        return None


class Scanner:
    """Multi-source feed scanner with two tiers: API (reliable) and Browser (opportunistic)."""

    def __init__(self, browser=None):
        """
        Args:
            browser: Optional BrowserBridge instance for Tier 2 (X, Reddit via CDP).
                     If None, browser sources are skipped.
        """
        self.browser = browser
        os.makedirs(CANDIDATES_DIR, exist_ok=True)
        os.makedirs(VERDICTS_DIR, exist_ok=True)

    # ── Tier 1: API Sources (always available) ───────────────────────────

    def scan_hackernews(self) -> dict:
        """Scan Hacker News top stories via public API."""
        try:
            story_ids = fetch_json(HN_TOP_STORIES)
            if not story_ids:
                return {"status": "error", "items": [], "error": "Failed to fetch HN top stories"}

            items = []
            for sid in story_ids[:HN_MAX_ITEMS]:
                story = fetch_json(HN_ITEM.format(id=sid))
                if not story or story.get("type") != "story":
                    continue

                title = story.get("title", "")
                score = story.get("score", 0)
                comments = story.get("descendants", 0)
                url = story.get("url", f"https://news.ycombinator.com/item?id={sid}")

                # Skip low-score items
                if score < HN_MIN_SCORE:
                    continue

                relevance = score_relevance(title)
                if relevance < 0.05:
                    continue

                items.append({
                    "id": f"hn_{sid}",
                    "title": title,
                    "url": url,
                    "source": "hackernews",
                    "score": score,
                    "comments": comments,
                    "relevance": round(relevance, 2),
                    "discovered_at": datetime.now().isoformat(),
                    "hn_url": f"https://news.ycombinator.com/item?id={sid}",
                })

            # Sort by relevance * log(score) for balanced ranking
            import math
            items.sort(key=lambda x: x["relevance"] * math.log(max(x["score"], 1) + 1), reverse=True)

            return {"status": "ok", "items": items[:15]}

        except Exception as e:
            return {"status": "error", "items": [], "error": str(e)}

    def scan_rss(self) -> dict:
        """Scan RSS/Atom feeds for relevant posts."""
        all_items = []
        errors = []

        for name, url in RSS_FEEDS.items():
            try:
                raw = fetch_text(url)
                if not raw:
                    errors.append(f"{name}: fetch failed")
                    continue

                root = ET.fromstring(raw)

                # Handle both RSS and Atom formats
                ns = {"atom": "http://www.w3.org/2005/Atom"}
                entries = root.findall(".//item") or root.findall(".//atom:entry", ns)

                for entry in entries[:10]:  # Cap per feed
                    # RSS format
                    title_el = entry.find("title") or entry.find("atom:title", ns)
                    link_el = entry.find("link") or entry.find("atom:link", ns)
                    desc_el = entry.find("description") or entry.find("atom:summary", ns) or entry.find("atom:content", ns)

                    title = title_el.text if title_el is not None and title_el.text else ""
                    if link_el is not None:
                        link = link_el.text or link_el.get("href", "")
                    else:
                        link = ""
                    desc = desc_el.text if desc_el is not None and desc_el.text else ""
                    # Strip HTML tags from description
                    desc = re.sub(r"<[^>]+>", " ", desc)[:300]

                    relevance = score_relevance(title, desc)
                    if relevance < 0.05:
                        continue

                    all_items.append({
                        "id": f"rss_{name}_{hash(title) % 100000}",
                        "title": title.strip(),
                        "url": link.strip(),
                        "source": f"rss:{name}",
                        "snippet": desc.strip(),
                        "relevance": round(relevance, 2),
                        "discovered_at": datetime.now().isoformat(),
                    })

            except Exception as e:
                errors.append(f"{name}: {e}")

        all_items.sort(key=lambda x: x["relevance"], reverse=True)

        result = {"status": "ok", "items": all_items[:15]}
        if errors:
            result["warnings"] = errors
        return result

    def scan_reddit_rss(self) -> dict:
        """Scan Reddit via public RSS feeds (no auth needed!)."""
        all_items = []
        errors = []

        for sub, url in REDDIT_RSS.items():
            try:
                raw = fetch_text(url)
                if not raw:
                    errors.append(f"r/{sub}: fetch failed")
                    continue

                root = ET.fromstring(raw)
                ns = {"atom": "http://www.w3.org/2005/Atom"}
                entries = root.findall(".//atom:entry", ns)

                for entry in entries[:15]:
                    title_el = entry.find("atom:title", ns)
                    link_el = entry.find("atom:link", ns)
                    content_el = entry.find("atom:content", ns)

                    title = title_el.text if title_el is not None and title_el.text else ""
                    link = link_el.get("href", "") if link_el is not None else ""
                    content = content_el.text if content_el is not None and content_el.text else ""
                    content = re.sub(r"<[^>]+>", " ", content)[:300]

                    # Extract score from content if present
                    score_match = re.search(r"(\d+)\s*points?", content)
                    score = int(score_match.group(1)) if score_match else 0

                    relevance = score_relevance(title, content)
                    if relevance < 0.05:
                        continue

                    all_items.append({
                        "id": f"reddit_{sub}_{hash(title) % 100000}",
                        "title": title.strip(),
                        "url": link.strip(),
                        "source": f"reddit:r/{sub}",
                        "score": score,
                        "snippet": content.strip()[:200],
                        "relevance": round(relevance, 2),
                        "discovered_at": datetime.now().isoformat(),
                    })

            except Exception as e:
                errors.append(f"r/{sub}: {e}")

        all_items.sort(key=lambda x: x["relevance"], reverse=True)
        result = {"status": "ok", "items": all_items[:15]}
        if errors:
            result["warnings"] = errors
        return result

    # ── Tier 2: Browser Sources (opportunistic) ─────────────────────────

    def _browser_available(self) -> bool:
        """Check if Chrome CDP is available for Tier 2 scanning."""
        if not self.browser:
            return False
        try:
            return self.browser.is_available()
        except Exception:
            return False

    def scan_x_feed(self) -> dict:
        """Scan X feed via Chrome CDP. Requires authenticated session."""
        if not self._browser_available():
            return {"status": "cdp_unavailable", "items": [], "error": "Chrome CDP not running"}

        try:
            result = self.browser.scan_x_feed(count=15)
            if result.get("auth_wall"):
                return {"status": "auth_wall", "items": [], "error": result.get("error", "Not logged into X")}
            if result.get("error"):
                return {"status": "error", "items": [], "error": result["error"]}

            items = []
            for tweet in result.get("tweets", []):
                text = tweet.get("text", "")
                relevance = score_relevance(text)

                items.append({
                    "id": f"x_{hash(text[:50]) % 100000}",
                    "title": text[:100],
                    "url": "",  # X doesn't expose URLs easily from feed scan
                    "source": "x_feed",
                    "author": tweet.get("handle", tweet.get("author", "")),
                    "snippet": text[:300],
                    "metrics": tweet.get("metrics", ""),
                    "links": tweet.get("links", []),
                    "relevance": round(relevance, 2),
                    "discovered_at": datetime.now().isoformat(),
                })

            items.sort(key=lambda x: x["relevance"], reverse=True)
            return {"status": "ok", "items": items[:10]}

        except Exception as e:
            return {"status": "error", "items": [], "error": str(e)}

    # ── Scan Cycle ──────────────────────────────────────────────────────

    def run_cycle(self) -> dict:
        """Run a full scan cycle across all sources. Returns the candidate file path."""
        timestamp = datetime.now()
        scan_id = timestamp.strftime("%Y-%m-%d_%H%M")

        sources = {}
        total = 0

        # Tier 1: Always run (no auth, no Chrome needed)
        for name, method in [
            ("hackernews", self.scan_hackernews),
            ("rss", self.scan_rss),
            ("reddit", self.scan_reddit_rss),
        ]:
            result = method()
            sources[name] = result
            total += len(result.get("items", []))

        # Tier 2: Only run if Chrome CDP is available
        x_result = self.scan_x_feed()
        sources["x_feed"] = x_result
        total += len(x_result.get("items", []))

        # Build summary
        by_source = {k: len(v.get("items", [])) for k, v in sources.items()}
        failed = [k for k, v in sources.items() if v.get("status") != "ok"]

        candidates = {
            "scan_id": scan_id,
            "timestamp": timestamp.isoformat(),
            "sources": sources,
            "summary": {
                "total_candidates": total,
                "by_source": by_source,
                "sources_failed": failed,
            },
        }

        # Write to vault
        filepath = os.path.join(CANDIDATES_DIR, f"{scan_id}.json")
        with open(filepath, "w") as f:
            json.dump(candidates, f, indent=2)

        return {"scan_id": scan_id, "filepath": filepath, "total": total, "failed": failed}

    # ── Verdict Reading (self-calibration) ──────────────────────────────

    def read_verdicts(self, since_days: int = 7) -> list:
        """Read recent verdict files from Claude's review."""
        verdicts = []
        cutoff = datetime.now() - timedelta(days=since_days)

        if not os.path.exists(VERDICTS_DIR):
            return []

        for fname in sorted(os.listdir(VERDICTS_DIR)):
            if not fname.endswith(".json"):
                continue
            fpath = os.path.join(VERDICTS_DIR, fname)
            try:
                with open(fpath) as f:
                    data = json.load(f)
                ts = data.get("timestamp", "")
                if ts and datetime.fromisoformat(ts) > cutoff:
                    verdicts.append(data)
            except Exception:
                continue

        return verdicts

    def get_calibration_stats(self) -> dict:
        """Analyze verdict history for self-calibration."""
        verdicts = self.read_verdicts()
        if not verdicts:
            return {"total_reviewed": 0, "signal_rate": 0, "notes": "No verdicts yet"}

        total = 0
        signal = 0
        noise = 0
        all_notes = []

        for v in verdicts:
            for item in v.get("verdicts", []):
                total += 1
                if item.get("verdict") == "signal":
                    signal += 1
                elif item.get("verdict") == "noise":
                    noise += 1
            notes = v.get("calibration_notes", {})
            if notes.get("suggested_filter_updates"):
                all_notes.extend(notes["suggested_filter_updates"])

        return {
            "total_reviewed": total,
            "signal": signal,
            "noise": noise,
            "signal_rate": round(signal / total, 2) if total > 0 else 0,
            "recent_calibration_notes": all_notes[-10:],
        }

    def get_unreviewed(self) -> list:
        """Get candidate files that haven't been reviewed yet."""
        # Get all candidate scan IDs
        candidate_ids = set()
        if os.path.exists(CANDIDATES_DIR):
            for f in os.listdir(CANDIDATES_DIR):
                if f.endswith(".json"):
                    candidate_ids.add(f.replace(".json", ""))

        # Get all reviewed scan IDs
        reviewed_ids = set()
        if os.path.exists(VERDICTS_DIR):
            for f in os.listdir(VERDICTS_DIR):
                if f.endswith(".json"):
                    try:
                        with open(os.path.join(VERDICTS_DIR, f)) as fh:
                            data = json.load(fh)
                            reviewed_ids.update(data.get("reviewed_scans", []))
                    except Exception:
                        continue

        unreviewed = sorted(candidate_ids - reviewed_ids)
        return unreviewed

    def get_latest_candidates(self, top_n: int = 10) -> list:
        """Get top candidates from most recent scan, ranked by relevance."""
        if not os.path.exists(CANDIDATES_DIR):
            return []

        files = sorted([f for f in os.listdir(CANDIDATES_DIR) if f.endswith(".json")])
        if not files:
            return []

        latest = os.path.join(CANDIDATES_DIR, files[-1])
        try:
            with open(latest) as f:
                data = json.load(f)
        except Exception:
            return []

        # Flatten all items across sources
        all_items = []
        for source_data in data.get("sources", {}).values():
            all_items.extend(source_data.get("items", []))

        all_items.sort(key=lambda x: x.get("relevance", 0), reverse=True)
        return all_items[:top_n]
