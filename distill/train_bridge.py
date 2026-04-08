#!/usr/bin/env python3
"""Bridge between training process output and dashboard WebSocket.

Tails the training output file and sends parsed metrics to the dashboard
via WebSocket on port 8421.

Usage:
  python train_bridge.py --logfile /path/to/training.log
  # Or pipe directly:
  ./train ... 2>&1 | python train_bridge.py --pipe
"""

import asyncio
import json
import re
import sys
import argparse
import time

try:
    import aiohttp
    from aiohttp import web
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

# Regex patterns for parsing training output
RE_STEP = re.compile(r'step\s+(\d+)\s+loss=([\d.nan]+)\s+lr=([\d.e+-]+)\s+([\d.]+)ms/step\s+x\[([-\d.inf]+),([-\d.inf]+)\]\s+dy\[([-\d.e+nan]+),([-\d.e+nan]+)\]')
RE_DISTILL = re.compile(r'\[distill\]\s+CE=([\d.nan]+)\s+KL=([\d.nan]+)\s+combined=([\d.nan]+)\s+\(seq=(\d+)\)')
RE_GRAD = re.compile(r'grad_norm=([\d.naninf]+)\s+attn=([\d.naninf]+)\s+ffn=([\d.naninf]+)\s+embed=([\d.naninf]+)')
RE_SKIP = re.compile(r'SKIP bwd|WARNING.*skipping')
RE_SDPA = re.compile(r'L0 sdpa_bwd.*dq\|=([\d.nan]+).*dk\|=([\d.nan]+).*dv\|=([\d.nan]+)')

clients = set()
metrics_history = []

def parse_float(s):
    try:
        return float(s)
    except (ValueError, TypeError):
        return None

def parse_line(line):
    """Parse a training output line into a metric event."""
    line = line.strip()
    if not line:
        return None

    m = RE_STEP.search(line)
    if m:
        return {
            'type': 'step',
            'step': int(m.group(1)),
            'loss': parse_float(m.group(2)),
            'lr': parse_float(m.group(3)),
            'ms_per_step': parse_float(m.group(4)),
            'x_min': parse_float(m.group(5)),
            'x_max': parse_float(m.group(6)),
            'dy_min': parse_float(m.group(7)),
            'dy_max': parse_float(m.group(8)),
            'time': time.time()
        }

    m = RE_DISTILL.search(line)
    if m:
        return {
            'type': 'distill',
            'ce_loss': parse_float(m.group(1)),
            'kl_loss': parse_float(m.group(2)),
            'combined': parse_float(m.group(3)),
            'seq_idx': int(m.group(4)),
            'time': time.time()
        }

    m = RE_GRAD.search(line)
    if m:
        return {
            'type': 'gradients',
            'grad_norm': parse_float(m.group(1)),
            'attn_norm': parse_float(m.group(2)),
            'ffn_norm': parse_float(m.group(3)),
            'embed_norm': parse_float(m.group(4)),
            'time': time.time()
        }

    m = RE_SDPA.search(line)
    if m:
        return {
            'type': 'sdpa',
            'dq_max': parse_float(m.group(1)),
            'dk_max': parse_float(m.group(2)),
            'dv_max': parse_float(m.group(3)),
            'time': time.time()
        }

    if RE_SKIP.search(line):
        return {
            'type': 'skip',
            'message': line,
            'time': time.time()
        }

    if 'ckpt saved' in line:
        return {
            'type': 'checkpoint',
            'message': line,
            'time': time.time()
        }

    return None


async def ws_handler(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    clients.add(ws)
    print(f"  Dashboard connected ({len(clients)} clients)")

    # Send history
    for m in metrics_history[-100:]:
        await ws.send_json(m)

    try:
        async for msg in ws:
            pass  # We don't expect messages from dashboard
    finally:
        clients.discard(ws)
        print(f"  Dashboard disconnected ({len(clients)} clients)")
    return ws


last_distill = {}  # Cache last distill event to merge with step

async def broadcast(event):
    global clients, last_distill
    metrics_history.append(event)

    # Convert to dashboard-compatible format
    dashboard_msgs = []
    if event['type'] == 'distill':
        last_distill = event  # Cache for merging with next step
    elif event['type'] == 'step':
        # Merge with cached distill event
        step_msg = {
            'type': 'step',
            'step': event['step'],
            'loss': event.get('loss'),
            'ms': event.get('ms_per_step'),
        }
        if last_distill:
            step_msg['ce'] = last_distill.get('ce_loss')
            step_msg['kl'] = last_distill.get('kl_loss')
        dashboard_msgs.append(step_msg)
        # Also send timing
        timing_msg = {
            'type': 'timing',
            'lr': event.get('lr'),
            'x_range': [event.get('x_min', 0), event.get('x_max', 0)],
            'dy_range': [event.get('dy_min', 0), event.get('dy_max', 0)],
        }
        dashboard_msgs.append(timing_msg)
    elif event['type'] == 'gradients':
        dashboard_msgs.append({
            'type': 'timing',
            'grad_norm': event.get('grad_norm'),
        })
    elif event['type'] == 'skip':
        dashboard_msgs.append({
            'type': 'log',
            'msg': event.get('message', 'Step skipped'),
            'level': 'warn'
        })
    elif event['type'] == 'checkpoint':
        dashboard_msgs.append({
            'type': 'log',
            'msg': event.get('message', 'Checkpoint saved'),
            'level': 'highlight'
        })

    if not dashboard_msgs:
        dashboard_msgs = [event]  # Fallback: send raw

    dead = set()
    for ws in clients:
        for msg in dashboard_msgs:
            try:
                await ws.send_json(msg)
            except:
                dead.add(ws)
                break
    clients -= dead


async def tail_file(path):
    """Tail a file and broadcast parsed events."""
    print(f"  Tailing: {path}")
    pos = 0
    while True:
        try:
            with open(path, 'r') as f:
                f.seek(pos)
                new_data = f.read()
                if new_data:
                    pos = f.tell()
                    for line in new_data.split('\n'):
                        event = parse_line(line)
                        if event:
                            await broadcast(event)
        except FileNotFoundError:
            pass
        await asyncio.sleep(0.5)


async def read_pipe():
    """Read from stdin and broadcast parsed events."""
    print("  Reading from stdin pipe...")
    loop = asyncio.get_event_loop()
    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    await loop.connect_read_pipe(lambda: protocol, sys.stdin)

    while True:
        line = await reader.readline()
        if not line:
            break
        text = line.decode('utf-8', errors='replace')
        event = parse_line(text)
        if event:
            await broadcast(event)
        # Also print to stdout for terminal monitoring
        print(text, end='', flush=True)


async def main_server(args):
    if not HAS_AIOHTTP:
        print("ERROR: pip install aiohttp (required for WebSocket bridge)")
        sys.exit(1)

    app = web.Application()
    app.router.add_get('/ws', ws_handler)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', 8421)
    await site.start()
    print(f"\n  Training bridge running on ws://localhost:8421")
    print(f"  Open dashboard.html in browser to visualize\n")

    if args.pipe:
        await read_pipe()
    elif args.logfile:
        await tail_file(args.logfile)
    else:
        print("  Waiting for --logfile or --pipe...")
        while True:
            await asyncio.sleep(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--logfile', help='Path to training log file to tail')
    parser.add_argument('--pipe', action='store_true', help='Read from stdin pipe')
    args = parser.parse_args()

    asyncio.run(main_server(args))
