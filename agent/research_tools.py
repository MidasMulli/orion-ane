"""
Midas Researcher v1 — Read-only tools for the agentic research loop.

Four tools, all sandboxed to ~/Desktop/cowork: grep, read_file, list_dir,
recall_memory. Each returns a string the 70B planner can append to its
context. No write surface. No shell injection — args are passed as argv
lists, never shell strings.

The tool loop driver lives in midas_ui.py /api/research; this module is
the tool implementations only, kept separate so it stays auditable.
"""

import os
import subprocess
from pathlib import Path

# Hard sandbox: every path resolved through here, escapes the cowork tree
# refuse. Symlinks resolved before the check so /tmp links can't smuggle
# reads of arbitrary FS regions.
COWORK_ROOT = Path("/Users/midas/Desktop/cowork").resolve()
MAX_FILE_BYTES = 200_000   # ~5K lines at 40 chars; truncate above
MAX_GREP_MATCHES = 200     # head_limit equivalent
MAX_LIST_ENTRIES = 300


def _safe_resolve(path_str: str) -> Path | None:
    """Resolve path_str against COWORK_ROOT. Reject anything outside."""
    if not path_str:
        return None
    p = Path(path_str)
    if not p.is_absolute():
        p = COWORK_ROOT / p
    try:
        resolved = p.resolve()
    except (OSError, RuntimeError):
        return None
    try:
        resolved.relative_to(COWORK_ROOT)
    except ValueError:
        return None
    return resolved


def tool_grep(pattern: str, glob: str | None = None, path: str | None = None) -> str:
    """ripgrep wrapper. Returns file:line:content matches, capped."""
    if not pattern:
        return "ERROR: grep requires a pattern"
    search_root = _safe_resolve(path) if path else COWORK_ROOT
    if search_root is None:
        return f"ERROR: path '{path}' is outside the sandbox"
    cmd = ["rg", "--no-heading", "--line-number", "--color", "never",
           "--max-count", "20", "-S"]
    if glob:
        cmd.extend(["--glob", glob])
    cmd.append(pattern)
    cmd.append(str(search_root))
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
    except subprocess.TimeoutExpired:
        return "ERROR: grep timed out (20s)"
    except FileNotFoundError:
        return "ERROR: ripgrep (rg) not on PATH"
    lines = (out.stdout or "").splitlines()
    if not lines:
        return f"NO MATCHES for /{pattern}/" + (f" in {glob}" if glob else "")
    if len(lines) > MAX_GREP_MATCHES:
        truncated = len(lines) - MAX_GREP_MATCHES
        lines = lines[:MAX_GREP_MATCHES]
        lines.append(f"... ({truncated} more matches truncated; refine pattern or glob)")
    # Strip the absolute prefix to keep output compact
    prefix = str(COWORK_ROOT) + "/"
    return "\n".join(l.replace(prefix, "") for l in lines)


def tool_read_file(path: str, offset: int = 1, limit: int = 200) -> str:
    """Read a file from the sandbox. offset is 1-indexed, limit is line count."""
    p = _safe_resolve(path)
    if p is None:
        return f"ERROR: path '{path}' is outside the sandbox or invalid"
    if not p.exists():
        return f"ERROR: file '{path}' does not exist"
    if not p.is_file():
        return f"ERROR: '{path}' is not a regular file"
    try:
        size = p.stat().st_size
    except OSError as e:
        return f"ERROR: stat failed: {e}"
    if size > MAX_FILE_BYTES * 5:
        return f"ERROR: file is {size} bytes — too large to read directly. Use grep with a specific pattern."
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        return f"ERROR: read failed: {e}"
    all_lines = text.splitlines()
    total = len(all_lines)
    start = max(1, int(offset)) - 1
    end = min(total, start + max(1, int(limit)))
    chunk = all_lines[start:end]
    rel = str(p.relative_to(COWORK_ROOT))
    header = f"{rel} (lines {start+1}-{end} of {total})"
    numbered = "\n".join(f"{start+i+1}\t{line}" for i, line in enumerate(chunk))
    body = numbered if len(numbered) <= MAX_FILE_BYTES else numbered[:MAX_FILE_BYTES] + "\n... (truncated)"
    return f"{header}\n{body}"


def tool_list_dir(path: str = ".") -> str:
    """List a directory. Marks files vs subdirs, capped."""
    p = _safe_resolve(path)
    if p is None:
        return f"ERROR: path '{path}' is outside the sandbox"
    if not p.exists():
        return f"ERROR: directory '{path}' does not exist"
    if not p.is_dir():
        return f"ERROR: '{path}' is not a directory"
    try:
        entries = sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
    except OSError as e:
        return f"ERROR: list failed: {e}"
    if len(entries) > MAX_LIST_ENTRIES:
        truncated = len(entries) - MAX_LIST_ENTRIES
        entries = entries[:MAX_LIST_ENTRIES]
        tail = f"\n... ({truncated} more entries truncated)"
    else:
        tail = ""
    rel = str(p.relative_to(COWORK_ROOT)) or "."
    lines = [f"{rel}/ ({len(entries)} entries shown)"]
    for e in entries:
        try:
            if e.is_dir():
                lines.append(f"  {e.name}/")
            else:
                sz = e.stat().st_size
                lines.append(f"  {e.name}  ({sz} B)")
        except OSError:
            lines.append(f"  {e.name}  (stat failed)")
    return "\n".join(lines) + tail


def tool_recall_memory(memory_bridge, query: str, k: int = 8) -> str:
    """Wrap MemoryBridge.recall into a tool result string."""
    if not query:
        return "ERROR: recall_memory requires a query"
    try:
        result = memory_bridge.recall(query, n_results=int(k))
    except Exception as e:
        return f"ERROR: recall failed: {type(e).__name__}: {e}"
    if isinstance(result, dict) and result.get("error"):
        return f"ERROR: {result['error']}"
    items = (result or {}).get("results", []) if isinstance(result, dict) else []
    if not items:
        return f"NO MEMORIES for /{query}/"
    lines = [f"Recalled {len(items)} memories for /{query}/:"]
    for i, m in enumerate(items, 1):
        score = m.get("score", m.get("similarity", 0.0))
        text = (m.get("text") or m.get("content") or "").strip().replace("\n", " ")
        if len(text) > 280:
            text = text[:277] + "..."
        lines.append(f"  [{i}] (score={score:.2f}) {text}")
    return "\n".join(lines)


# Tool dispatch table — used by the loop driver in midas_ui.py
def dispatch(tool_name: str, args: dict, memory_bridge=None) -> str:
    """Dispatch a tool call. Returns a string the 70B can read."""
    if tool_name == "grep":
        return tool_grep(args.get("pattern", ""),
                         args.get("glob"),
                         args.get("path"))
    if tool_name == "read_file":
        return tool_read_file(args.get("path", ""),
                              args.get("offset", 1),
                              args.get("limit", 200))
    if tool_name == "list_dir":
        return tool_list_dir(args.get("path", "."))
    if tool_name == "recall_memory":
        if memory_bridge is None:
            return "ERROR: memory bridge not wired"
        return tool_recall_memory(memory_bridge,
                                  args.get("query", ""),
                                  args.get("k", 8))
    return f"ERROR: unknown tool '{tool_name}'. Valid: grep, read_file, list_dir, recall_memory"


# Tool catalog for the system prompt — keep in sync with dispatch()
TOOL_CATALOG = """\
You have four read-only tools. Call exactly one per turn by emitting a single
JSON object on a line by itself, prefixed with TOOL_CALL:.

  TOOL_CALL: {"tool": "grep", "args": {"pattern": "...", "glob": "*.py", "path": "ane-compiler"}}
  TOOL_CALL: {"tool": "read_file", "args": {"path": "ane-compiler/build_8b_q8.py", "offset": 1, "limit": 100}}
  TOOL_CALL: {"tool": "list_dir", "args": {"path": "ane-compiler"}}
  TOOL_CALL: {"tool": "recall_memory", "args": {"query": "8B dispatch fusion", "k": 8}}

Tool details:
  grep        — ripgrep over /Users/midas/Desktop/cowork. Args: pattern (required),
                glob (optional file filter like "*.py"), path (optional subdir).
                Returns up to 200 file:line matches.
  read_file   — read a file from the sandbox. Args: path (required), offset (1-indexed
                line, default 1), limit (lines, default 200). Returns numbered lines.
  list_dir    — list a directory. Args: path (default "."). Returns names with sizes.
  recall_memory — query the Subconscious memory store. Args: query (required),
                k (top-k, default 8). Returns scored memories.

When you have enough evidence to answer, emit a single line:
  FINAL_REPORT:
followed by your final report. The loop ends when FINAL_REPORT is emitted or
the iteration cap is reached.

Cite file:line for every code claim. If you cannot verify a claim, say so
explicitly — do not guess. Memory recall is for priors and context; for any
claim about current code state, you MUST verify with grep or read_file.
"""
