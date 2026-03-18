#!/usr/bin/env python3
"""
Browser Bridge — Chrome CDP access for Midas agent
=====================================================

Connects to Chrome via CDP (port 9222) for authenticated web access.
Uses the user's real Chrome session — logged into X, Reddit, Gmail, etc.

Requirements:
  - Chrome running with: --remote-debugging-port=9222 --user-data-dir=$HOME/.chrome-debug
  - Start with: ~/.hermes/start-chrome-debug.sh

Provides 5 tools:
  browse_navigate  — Open a URL
  browse_read      — Read page content (text extraction)
  browse_click     — Click an element by selector
  browse_type      — Type text into an input field
  browse_js        — Execute JavaScript in page context
"""

import json
import time
import urllib.request
import urllib.error


CDP_HOST = "127.0.0.1"
CDP_PORT = 9222


class BrowserBridge:
    """Lightweight CDP client — no dependencies beyond stdlib + pychrome."""

    def __init__(self):
        self._connected = False
        self._browser = None
        self._tab = None

    def is_available(self) -> bool:
        """Check if Chrome CDP is running."""
        try:
            resp = urllib.request.urlopen(
                f"http://{CDP_HOST}:{CDP_PORT}/json/version", timeout=1
            )
            return resp.status == 200
        except:
            return False

    def connect(self):
        """Connect to Chrome CDP."""
        import pychrome
        self._browser = pychrome.Browser(url=f"http://{CDP_HOST}:{CDP_PORT}")
        tabs = self._browser.list_tab()
        if tabs:
            self._tab = tabs[0]
            self._tab.start()
        else:
            self._tab = self._browser.new_tab()
            self._tab.start()
        self._connected = True

    def _ensure_tab(self):
        if not self._connected:
            self.connect()

    # Patterns that indicate an auth wall or access denied
    AUTH_WALL_TITLES = [
        "page not found", "not found", "sign in", "log in", "login",
        "access denied", "403", "401", "unauthorized", "authenticate",
    ]
    AUTH_WALL_URL_PATTERNS = [
        "/login", "/signin", "/auth", "/sso", "/oauth",
        "accounts.google.com", "appleid.apple.com",
    ]

    def navigate(self, url: str, wait: float = 2.0) -> dict:
        """Navigate to a URL and wait for load. Detects auth walls."""
        self._ensure_tab()
        try:
            self._tab.Page.navigate(url=url)
            time.sleep(wait)
            # Get page title
            result = self._tab.Runtime.evaluate(expression="document.title")
            title = result.get("result", {}).get("value", "")
            result2 = self._tab.Runtime.evaluate(expression="window.location.href")
            current_url = result2.get("result", {}).get("value", "")

            # Detect auth walls
            title_lower = title.lower()
            url_lower = current_url.lower()
            is_auth_wall = (
                any(p in title_lower for p in self.AUTH_WALL_TITLES)
                or any(p in url_lower for p in self.AUTH_WALL_URL_PATTERNS)
            )

            resp = {"status": "ok", "title": title, "url": current_url}
            if is_auth_wall:
                resp["auth_wall"] = True
                resp["warning"] = f"Likely auth wall or access denied. The user may need to log into this site in the debug Chrome (~/.chrome-debug). Do NOT retry — tell the user."
            return resp
        except Exception as e:
            return {"error": str(e)}

    def read_page(self, selector: str = "body", max_length: int = 5000) -> dict:
        """Read text content from the page or a specific selector."""
        self._ensure_tab()
        try:
            js = f"""
            (() => {{
                const el = document.querySelector('{selector}');
                if (!el) return JSON.stringify({{error: 'selector not found: {selector}'}});
                const text = el.innerText || el.textContent || '';
                return JSON.stringify({{
                    text: text.substring(0, {max_length}),
                    title: document.title,
                    url: window.location.href,
                    truncated: text.length > {max_length}
                }});
            }})()
            """
            result = self._tab.Runtime.evaluate(expression=js)
            val = result.get("result", {}).get("value", "{}")
            return json.loads(val)
        except Exception as e:
            return {"error": str(e)}

    def click(self, selector: str) -> dict:
        """Click an element by CSS selector."""
        self._ensure_tab()
        try:
            js = f"""
            (() => {{
                const el = document.querySelector('{selector}');
                if (!el) return JSON.stringify({{error: 'not found: {selector}'}});
                el.click();
                return JSON.stringify({{status: 'clicked', selector: '{selector}'}});
            }})()
            """
            result = self._tab.Runtime.evaluate(expression=js)
            val = result.get("result", {}).get("value", "{}")
            return json.loads(val)
        except Exception as e:
            return {"error": str(e)}

    def type_text(self, selector: str, text: str) -> dict:
        """Type text into an input field."""
        self._ensure_tab()
        try:
            # Escape quotes in text
            escaped = text.replace("\\", "\\\\").replace("'", "\\'").replace("\n", "\\n")
            js = f"""
            (() => {{
                const el = document.querySelector('{selector}');
                if (!el) return JSON.stringify({{error: 'not found: {selector}'}});
                el.focus();
                el.value = '{escaped}';
                el.dispatchEvent(new Event('input', {{bubbles: true}}));
                el.dispatchEvent(new Event('change', {{bubbles: true}}));
                return JSON.stringify({{status: 'typed', selector: '{selector}', length: {len(text)}}});
            }})()
            """
            result = self._tab.Runtime.evaluate(expression=js)
            val = result.get("result", {}).get("value", "{}")
            return json.loads(val)
        except Exception as e:
            return {"error": str(e)}

    def run_js(self, expression: str) -> dict:
        """Execute arbitrary JavaScript and return result."""
        self._ensure_tab()
        try:
            result = self._tab.Runtime.evaluate(expression=expression)
            val = result.get("result", {})
            if val.get("type") == "string":
                # Try to parse as JSON
                try:
                    return json.loads(val["value"])
                except:
                    return {"result": val["value"]}
            elif val.get("type") == "number":
                return {"result": val["value"]}
            elif val.get("type") == "boolean":
                return {"result": val["value"]}
            elif val.get("type") == "undefined":
                return {"result": None}
            else:
                return {"result": str(val.get("value", val))}
        except Exception as e:
            return {"error": str(e)}

    def search(self, query: str, max_results: int = 5) -> dict:
        """Google search with structured result extraction."""
        self._ensure_tab()
        try:
            # Navigate to Google search
            encoded = query.replace(" ", "+")
            self._tab.Page.navigate(url=f"https://www.google.com/search?q={encoded}")
            time.sleep(2.5)

            js = """
            (() => {
                const results = {snippets: [], featured: null, knowledge: null};

                // Featured snippet / answer box (stock prices, weather, conversions, etc.)
                const featured = document.querySelector('[data-featured-snippet]')
                    || document.querySelector('.IZ6rdc')
                    || document.querySelector('[data-attrid="wa:/description"]')
                    || document.querySelector('.ayqGOc');
                if (featured) {
                    results.featured = featured.innerText.substring(0, 500);
                }

                // Knowledge panel (right side)
                const kp = document.querySelector('[data-attrid]');
                if (kp) {
                    const kpText = kp.closest('.kp-wholepage, .osrp-blk, .Z1hOCe');
                    if (kpText) results.knowledge = kpText.innerText.substring(0, 800);
                }

                // Stock price specific
                const stockPrice = document.querySelector('[data-attrid="Price"] .IsqQVc')
                    || document.querySelector('[data-attrid*="rice"]')
                    || document.querySelector('.NprOob');
                if (stockPrice) {
                    results.featured = (results.featured || '') + ' PRICE: ' + stockPrice.innerText;
                }

                // Top search results
                const items = document.querySelectorAll('.tF2Cxc, .g');
                for (let i = 0; i < Math.min(items.length, """ + str(max_results) + """); i++) {
                    const item = items[i];
                    const title = item.querySelector('h3');
                    const desc = item.querySelector('.VwiC3b, .lEBKkf, [data-sncf]');
                    const link = item.querySelector('a');
                    if (title) {
                        results.snippets.push({
                            title: title.innerText,
                            description: desc ? desc.innerText.substring(0, 200) : '',
                            url: link ? link.href : ''
                        });
                    }
                }

                // Fallback: grab any prominent number/value on the page
                if (!results.featured) {
                    const bigNums = document.querySelectorAll('.BNeawe.iBp4i, .BNeawe.deIvCb, .LGOjhe, .DlyJkd');
                    if (bigNums.length > 0) {
                        results.featured = Array.from(bigNums).slice(0, 3).map(n => n.innerText).join(' | ');
                    }
                }

                return JSON.stringify(results);
            })()
            """
            result = self._tab.Runtime.evaluate(expression=js)
            val = result.get("result", {}).get("value", "{}")
            parsed = json.loads(val)
            parsed["query"] = query
            parsed["url"] = f"https://www.google.com/search?q={encoded}"
            return parsed
        except Exception as e:
            return {"error": str(e)}

    def scan_x_feed(self, count: int = 5) -> dict:
        """Navigate to X/Twitter, extract top tweets from the feed.

        Handles all the DOM complexity internally — tries multiple selectors,
        scrolls for content, extracts structured tweet data. One call, done.
        """
        self._ensure_tab()
        try:
            # Navigate to X home feed
            self._tab.Page.navigate(url="https://x.com/home")
            time.sleep(3)  # X is slow to hydrate

            # Check for auth wall
            result = self._tab.Runtime.evaluate(expression="document.title")
            title = result.get("result", {}).get("value", "")
            result2 = self._tab.Runtime.evaluate(expression="window.location.href")
            current_url = result2.get("result", {}).get("value", "")

            if any(p in title.lower() for p in self.AUTH_WALL_TITLES) or \
               any(p in current_url.lower() for p in self.AUTH_WALL_URL_PATTERNS):
                return {
                    "auth_wall": True,
                    "error": "Not logged into X. Log in at the debug Chrome (~/.chrome-debug) first."
                }

            # Scroll down a bit to load more tweets
            self._tab.Runtime.evaluate(expression="window.scrollBy(0, 800)")
            time.sleep(1)
            self._tab.Runtime.evaluate(expression="window.scrollBy(0, 800)")
            time.sleep(1)

            # Extract tweets with multiple selector strategies
            js = """
            (() => {
                const tweets = [];

                // Strategy 1: article[data-testid="tweet"] (most reliable)
                let articles = document.querySelectorAll('article[data-testid="tweet"]');

                // Strategy 2: fall back to any article inside timeline
                if (articles.length === 0) {
                    const timeline = document.querySelector('[aria-label="Timeline: Your Home Timeline"]')
                        || document.querySelector('[aria-label="Timeline"]')
                        || document.querySelector('main');
                    if (timeline) {
                        articles = timeline.querySelectorAll('article');
                    }
                }

                // Strategy 3: broadest fallback
                if (articles.length === 0) {
                    articles = document.querySelectorAll('article');
                }

                for (let i = 0; i < Math.min(articles.length, """ + str(count + 5) + """); i++) {
                    const article = articles[i];
                    const text = article.innerText || '';

                    // Skip ads and very short tweets
                    if (text.length < 20) continue;
                    if (text.toLowerCase().includes('promoted') && text.length < 200) continue;

                    // Extract structured data
                    const lines = text.split('\\n').filter(l => l.trim());

                    // Find author (usually first non-empty line or has @)
                    let author = '';
                    let handle = '';
                    let tweetText = '';
                    let metrics = '';

                    for (const line of lines) {
                        if (line.startsWith('@')) {
                            handle = line.split('·')[0].trim();
                        } else if (line.includes('@') && !handle) {
                            // Line might contain "Name @handle · time"
                            const atMatch = line.match(/@\\w+/);
                            if (atMatch) handle = atMatch[0];
                        }
                    }

                    // Author is usually the first line
                    if (lines.length > 0) author = lines[0];

                    // Tweet body: skip header lines, grab the meat
                    // Headers usually contain @, timestamps like "1h", "2d", "Mar 15"
                    let bodyStart = -1;
                    for (let j = 0; j < lines.length; j++) {
                        const l = lines[j];
                        // Skip if it looks like metadata
                        if (l.startsWith('@') || l.match(/^\\d+[hdms]$/) || l === '·' || l.length < 3) continue;
                        // Skip if it's just a number (metrics)
                        if (l.match(/^[\\d,.KMB]+$/)) continue;
                        // Skip the author name line
                        if (j === 0) continue;
                        // Found body
                        if (!l.startsWith('@') && l.length > 10) {
                            bodyStart = j;
                            break;
                        }
                    }

                    if (bodyStart >= 0) {
                        // Collect body lines until we hit metrics
                        const bodyLines = [];
                        for (let j = bodyStart; j < lines.length; j++) {
                            const l = lines[j];
                            // Stop at metrics row (reply/repost/like counts)
                            if (l.match(/^[\\d,.KMB]+$/) && j > bodyStart + 1) break;
                            if (['Reply', 'Repost', 'Like', 'Bookmark', 'Share'].includes(l)) break;
                            bodyLines.push(l);
                        }
                        tweetText = bodyLines.join(' ').substring(0, 300);
                    } else {
                        // Fallback: just grab the longest segment
                        tweetText = lines.filter(l => l.length > 20).slice(0, 3).join(' ').substring(0, 300);
                    }

                    // Extract engagement metrics
                    const metricEls = article.querySelectorAll('[data-testid="reply"], [data-testid="retweet"], [data-testid="like"]');
                    const metricTexts = [];
                    metricEls.forEach(el => {
                        const val = el.getAttribute('aria-label') || el.innerText;
                        if (val) metricTexts.push(val);
                    });
                    metrics = metricTexts.join(', ');

                    // Extract any links
                    const links = [];
                    article.querySelectorAll('a[href]').forEach(a => {
                        const href = a.href;
                        if (href && !href.includes('x.com') && !href.includes('twitter.com')
                            && !href.startsWith('javascript') && !href.includes('/hashtag/')) {
                            links.push(href);
                        }
                    });

                    if (tweetText.length > 10) {
                        tweets.push({
                            author: author.substring(0, 50),
                            handle: handle || '',
                            text: tweetText,
                            metrics: metrics.substring(0, 100),
                            links: links.slice(0, 3)
                        });
                    }
                }

                return JSON.stringify({
                    tweets: tweets.slice(0, """ + str(count) + """),
                    total_found: articles.length,
                    url: window.location.href
                });
            })()
            """
            result = self._tab.Runtime.evaluate(expression=js)
            val = result.get("result", {}).get("value", "{}")
            return json.loads(val)
        except Exception as e:
            return {"error": str(e)}

    def get_tabs(self) -> list:
        """List all open tabs."""
        try:
            resp = urllib.request.urlopen(
                f"http://{CDP_HOST}:{CDP_PORT}/json", timeout=2
            )
            tabs = json.loads(resp.read())
            return [{"title": t.get("title", ""), "url": t.get("url", ""), "id": t.get("id", "")} for t in tabs if t.get("type") == "page"]
        except:
            return []

    def switch_tab(self, tab_index: int = 0) -> dict:
        """Switch to a different tab by index."""
        try:
            import pychrome
            tabs = self._browser.list_tab()
            if tab_index < 0 or tab_index >= len(tabs):
                return {"error": f"tab index {tab_index} out of range (0-{len(tabs)-1})"}
            if self._tab:
                try:
                    self._tab.stop()
                except:
                    pass
            self._tab = tabs[tab_index]
            self._tab.start()
            result = self._tab.Runtime.evaluate(expression="JSON.stringify({title: document.title, url: window.location.href})")
            val = result.get("result", {}).get("value", "{}")
            return json.loads(val)
        except Exception as e:
            return {"error": str(e)}

    def disconnect(self):
        """Disconnect from Chrome."""
        if self._tab:
            try:
                self._tab.stop()
            except:
                pass
        self._connected = False


# ── Tool Definitions for Agent ──

BROWSER_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "browse_navigate",
            "description": "Navigate to a URL in Chrome. Uses the user's authenticated browser session — logged into X, Reddit, Gmail, etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to navigate to"},
                    "wait": {"type": "number", "description": "Seconds to wait for page load (default 2)", "default": 2}
                },
                "required": ["url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "browse_read",
            "description": "Read text content from the current page. Can target a specific CSS selector or read the whole body.",
            "parameters": {
                "type": "object",
                "properties": {
                    "selector": {"type": "string", "description": "CSS selector to read (default 'body')", "default": "body"},
                    "max_length": {"type": "integer", "description": "Max characters to return (default 5000)", "default": 5000}
                },
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "browse_click",
            "description": "Click an element on the page by CSS selector.",
            "parameters": {
                "type": "object",
                "properties": {
                    "selector": {"type": "string", "description": "CSS selector to click"}
                },
                "required": ["selector"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "browse_type",
            "description": "Type text into an input field on the page.",
            "parameters": {
                "type": "object",
                "properties": {
                    "selector": {"type": "string", "description": "CSS selector for the input field"},
                    "text": {"type": "string", "description": "Text to type"}
                },
                "required": ["selector", "text"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "browse_js",
            "description": "Execute JavaScript in the page context. For complex data extraction, DOM queries, or page interaction. Return data as JSON.stringify().",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "JavaScript expression to evaluate"}
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "browse_search",
            "description": "Google search — returns featured snippet (stock prices, weather, conversions, quick answers) plus top search results. Use this FIRST for any factual lookup before navigating to a specific site.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query (e.g. 'Tesla stock price', 'weather NYC', 'USD to EUR')"},
                    "max_results": {"type": "integer", "description": "Number of search results to return (default 5)", "default": 5}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "browse_x_feed",
            "description": "Scan X/Twitter feed and return top tweets with author, text, and engagement. One call — handles navigation, scrolling, and DOM extraction automatically. Use this instead of browse_navigate for X feed scanning.",
            "parameters": {
                "type": "object",
                "properties": {
                    "count": {"type": "integer", "description": "Number of tweets to return (default 5)", "default": 5}
                },
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "browse_tabs",
            "description": "List all open browser tabs with titles and URLs.",
            "parameters": {
                "type": "object",
                "properties": {},
            }
        }
    },
]
