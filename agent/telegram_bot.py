#!/usr/bin/env python3
"""
Midas Telegram Bot — chat with Midas from anywhere.

Uses the v2 deterministic router: keyword routing → code-constructed tool calls → LLM text synthesis.
No OpenAI function calling. Same architecture as the terminal agent.

Connects to the local MLX server, memory daemon, scanner, and vault.
Runs as a launchd service. No port forwarding needed — Telegram handles the transport.

Usage:
    python3 telegram_bot.py                     # Run with token from env
    MIDAS_TG_TOKEN=xxx python3 telegram_bot.py  # Explicit token
"""

import os
import sys
import json
import logging
import re
from datetime import datetime

# Telegram
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
from telegram.constants import ParseMode, ChatAction

# Agent v2 modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from router import route, layer1_route
from tool_executor import execute as execute_tool, set_memory, set_browser
from synthesizer import synthesize
from memory_bridge import MemoryBridge
from feedback_loop import log_decision

# ── Config ──────────────────────────────────────────────────────────────────

TOKEN = os.environ.get("MIDAS_TG_TOKEN", "")
MLX_BASE_URL = os.environ.get("MLX_BASE_URL", "http://127.0.0.1:8899/v1")
MLX_MODEL = os.environ.get("MLX_MODEL", "mlx-community/Llama-3.3-70B-Instruct-3bit")
VAULT_PATH = "/Users/midas/Desktop/cowork/vault"
ALLOWED_USERS = []  # Empty = allow all. Add Telegram user IDs to restrict.

MAX_HISTORY = 20  # Messages to keep in context per chat

# ── Logging ─────────────────────────────────────────────────────────────────

logging.basicConfig(
    format="%(asctime)s [midas-tg] %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger("midas.telegram")

# Suppress noisy libraries
for name in ("httpx", "httpcore", "openai", "telegram.ext", "telegram"):
    logging.getLogger(name).setLevel(logging.WARNING)

# ── LLM Client ──────────────────────────────────────────────────────────────

import urllib.request


def llm_fn(messages, max_tokens=500, temperature=0.3):
    """Call the four-path MLX server."""
    payload = {
        "model": MLX_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "repetition_penalty": 1.15,
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{MLX_BASE_URL}/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    resp = urllib.request.urlopen(req, timeout=90)
    r = json.loads(resp.read())
    return r["choices"][0]["message"].get("content", "") or ""


def llm_classify(prompt, max_tokens=8, temperature=0.0):
    """Single-word LLM classification for Layer 2 routing."""
    return llm_fn(
        [{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
    )


# ── Memory Bridge (shared with agent_v2) ────────────────────────────────────

memory = MemoryBridge()

# ── Response Cleaning ───────────────────────────────────────────────────────

def _clean_response(text):
    """Strip think tags, special tokens, chat template artifacts, repetition."""
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'<think>.*$', '', text, flags=re.DOTALL)
    text = re.sub(r'</?think>', '', text)
    for special in ['<|endoftext|>', '<|im_end|>', '<|im_start|>']:
        text = text.replace(special, '')
    text = re.sub(r'\n(user|assistant|system)\s*$', '', text)

    # Truncate repetition loops: if any 8+ word phrase repeats 3+ times, cut at second occurrence
    words = text.split()
    for phrase_len in range(8, 4, -1):
        for i in range(len(words) - phrase_len):
            phrase = ' '.join(words[i:i + phrase_len])
            rest = ' '.join(words[i + phrase_len:])
            count = rest.count(phrase)
            if count >= 2:
                first_end = text.find(phrase) + len(phrase)
                second_start = text.find(phrase, first_end)
                if second_start > 0:
                    text = text[:second_start].rstrip()
                    break
        else:
            continue
        break

    return text.strip()


# ── Chat History ────────────────────────────────────────────────────────────

chat_histories: dict[int, list] = {}


def get_history(chat_id: int) -> list:
    if chat_id not in chat_histories:
        chat_histories[chat_id] = []
    return chat_histories[chat_id]


def trim_history(chat_id: int):
    hist = get_history(chat_id)
    if len(hist) > MAX_HISTORY:
        chat_histories[chat_id] = hist[-MAX_HISTORY:]


# ── Core Chat Function (v2 pipeline) ────────────────────────────────────────

def chat_with_midas(chat_id: int, user_message: str) -> str:
    """Route → Execute → Synthesize. Same pipeline as terminal agent_v2."""
    history = get_history(chat_id)

    # Auto-ingest user messages to memory
    memory.ingest("user", user_message)

    # Layer 1+2: Route
    l1_result = layer1_route(user_message)
    tool_name, tool_args = route(user_message, llm_fn=llm_classify)
    l2_category = tool_name if not l1_result else None
    log_decision(user_message, l1_result, l2_category, tool_name, tool_args)
    log.info("Route: '%s' -> %s", user_message[:60], tool_name)

    # Layer 3: Execute tool (if not conversation)
    tool_result = None
    if tool_name != "conversation":
        tool_result = execute_tool(tool_name, tool_args)
        log.info("Tool result: %s (%d chars)", tool_name, len(str(tool_result)))

    # Short-circuit: if tool returned an error, return it directly (don't let 9B hallucinate around it)
    if isinstance(tool_result, dict) and "error" in tool_result:
        text = tool_result["error"]
        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": text})
        trim_history(chat_id)
        return text

    # Layer 4: Synthesize response
    try:
        text = synthesize(
            llm_fn, history, user_message,
            tool_name=tool_name if tool_result else None,
            tool_args=tool_args if tool_result else None,
            tool_result=tool_result,
            max_tokens=400,  # Shorter for mobile
            temperature=0.3,
        )
    except Exception as e:
        log.error("Synthesis failed: %s", e)
        text = f"LLM error: {e}"

    text = _clean_response(text)

    # Update history
    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": text})
    trim_history(chat_id)

    return text


# ── Telegram Handlers ───────────────────────────────────────────────────────

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    log.info("New chat: %s (%d)", user.first_name, user.id)
    stats = memory.stats()
    total = stats.get("total_memories", "?") if isinstance(stats, dict) else "?"
    await update.message.reply_text(
        f"Midas online. {total} memories loaded.\n\n"
        "Running on your Mac — same brain, same memory, same tools.\n"
        "Ask me anything or say /scan to see what the scanner found."
    )


async def cmd_scan(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Quick scan digest."""
    await update.message.chat.send_action(ChatAction.TYPING)
    response = chat_with_midas(update.effective_chat.id, "What did the scanner find? Give me the top items.")
    await _send_long(update, response)


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """System status."""
    stats = memory.stats()
    total = stats.get("total_memories", "?") if isinstance(stats, dict) else "?"

    services = []
    for name, pid_file in [("enricher", "/tmp/phantom-enricher.pid"), ("scanner", "/tmp/phantom-scanner.pid")]:
        try:
            with open(pid_file) as f:
                pid = int(f.read().strip())
            os.kill(pid, 0)
            services.append(f"{name}: running")
        except:
            services.append(f"{name}: offline")

    # MLX check
    try:
        urllib.request.urlopen(f"{MLX_BASE_URL}/models", timeout=5)
        services.append("mlx: online")
    except:
        services.append("mlx: offline")

    text = f"**Midas Status**\nMemories: {total}\n" + "\n".join(services)
    await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)


async def cmd_clear(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Clear chat history."""
    chat_histories[update.effective_chat.id] = []
    await update.message.reply_text("History cleared.")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle regular text messages."""
    if not update.message or not update.message.text:
        return

    if ALLOWED_USERS and update.effective_user.id not in ALLOWED_USERS:
        await update.message.reply_text("Unauthorized.")
        return

    await update.message.chat.send_action(ChatAction.TYPING)
    response = chat_with_midas(update.effective_chat.id, update.message.text)
    await _send_long(update, response)


async def _send_long(update: Update, text: str):
    """Send a message, splitting if too long for Telegram's 4096 char limit."""
    if not text:
        text = "(empty response)"

    for i in range(0, len(text), 4000):
        chunk = text[i:i + 4000]
        try:
            await update.message.reply_text(chunk, parse_mode=ParseMode.MARKDOWN)
        except Exception:
            await update.message.reply_text(chunk)


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    log.info("Starting Midas Telegram Bot (v2 — deterministic router)...")

    # Suppress library noise during boot
    import warnings
    warnings.filterwarnings("ignore")
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Boot memory (no enricher — runs via launchd) and wire into tool_executor
    memory.start(enable_enricher=False)
    set_memory(memory)

    # Test MLX connection
    try:
        urllib.request.urlopen(f"{MLX_BASE_URL}/models", timeout=5)
        log.info("MLX server connected")
    except Exception as e:
        log.error("MLX server offline: %s", e)
        log.error("Start with: ~/.hermes/start-mlx-server.sh")
        return

    # Build and run the bot
    app = ApplicationBuilder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("scan", cmd_scan))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("clear", cmd_clear))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    log.info("Bot ready — listening for messages")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
