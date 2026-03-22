#!/usr/bin/env python3
"""
Automated Swarm Test Harness v2
================================
Sends tasks to the swarm dashboard via WebSocket, monitors events,
and waits for completion. Includes a session analyzer.

Usage:
  1. Start the swarm server normally
  2. Run: python tests/test_swarm_sessions.py
  3. Analyze: python tests/test_swarm_sessions.py --analyze
"""

import asyncio
import json
import sys
import time
import glob
import os
from pathlib import Path

try:
    import websockets
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "websockets", "-q"])
    import websockets

# ─── Configuration ───────────────────────────────────────────────────────────

WS_URL = "ws://localhost:8080/ws"
API_BASE = "http://localhost:8080"
LOGS_DIR = "/tmp/screener-mvp/.swarm_logs"

TEST_TASKS = [
    {
        "name": "T1: Order system",
        "message": (
            "Build an order tracking system with SQLAlchemy models and "
            "Pydantic schemas. Users can place orders with symbol, quantity, "
            "order type (market/limit), and price. Add CRUD routes under /orders."
        ),
    },
    {
        "name": "T2: Order history CSV",
        "message": (
            "Add a CSV export for order history. "
            "GET /orders/export should return all orders as a downloadable CSV."
        ),
    },
    {
        "name": "T3: Order analytics",
        "message": (
            "Add an analytics endpoint at GET /orders/analytics that returns "
            "total orders, orders by type, and most traded symbols as JSON."
        ),
    },
]

SESSION_TIMEOUT = 120
PAUSE_BETWEEN_SESSIONS = 20


# ─── WebSocket Test Runner ───────────────────────────────────────────────────

async def run_session(task: dict, session_num: int) -> dict:
    """Run a single test session via WebSocket."""
    result = {
        "task": task["name"],
        "session": session_num,
        "started_at": time.time(),
        "events": [],
        "status": "unknown",
        "agents_used": set(),
        "files_written": [],
        "error": None,
    }

    print(f"\n{'='*60}")
    print(f"  SESSION {session_num}: {task['name']}")
    print(f"{'='*60}")

    try:
        async with websockets.connect(WS_URL, ping_interval=20) as ws:
            # Drain initial event buffer
            print(f"  ⏳ Connected, draining event buffer...")
            try:
                while True:
                    msg = await asyncio.wait_for(ws.recv(), timeout=2.0)
            except asyncio.TimeoutError:
                pass

            # Resume all agents (in case paused from previous session)
            await ws.send(json.dumps({"action": "resume_all"}))
            await asyncio.sleep(2)

            # Drain resume events
            try:
                while True:
                    msg = await asyncio.wait_for(ws.recv(), timeout=1.0)
            except asyncio.TimeoutError:
                pass

            # Send the chat message
            print(f"  📤 Sending task: {task['message'][:80]}...")
            await ws.send(json.dumps({
                "action": "chat",
                "message": task["message"],
            }))

            # Monitor events
            # WS event format: {"source": str, "type": str, "data": dict, "timestamp": float}
            start = time.time()
            agent_completed = set()
            agents_dispatched = set()
            last_activity = time.time()

            while time.time() - start < SESSION_TIMEOUT:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
                    event = json.loads(raw)
                    last_activity = time.time()

                    # Dashboard WS uses "type" not "event_type"
                    etype = event.get("type", "")
                    source = event.get("source", "")
                    data = event.get("data", {})

                    # Track dispatches
                    if "dispatch" in etype.lower() or "task_dispatch" in etype.lower():
                        agent = data.get("agent", "")
                        if agent:
                            agents_dispatched.add(agent)
                            result["agents_used"].add(agent)
                            print(f"  🚀 Dispatched: {agent}")

                    # Track agent status changes
                    if "status" in etype.lower() or "AGENT_STATUS" in etype:
                        agent = data.get("agent", source)
                        status = data.get("status", "")
                        if "completed" in str(status).lower() or "completed" in etype.lower():
                            agent_completed.add(agent)
                            print(f"  ✅ Completed: {agent}")

                    # Track completions from orchestrator
                    if "completed" in etype.lower() or "task_done" in etype.lower():
                        agent = data.get("agent", source)
                        if agent:
                            agent_completed.add(agent)
                            print(f"  ✅ Task done: {agent}")

                    # Track file writes
                    if "file_write" in etype.lower() or "file" in etype.lower():
                        fp = data.get("file", data.get("path", ""))
                        if fp:
                            result["files_written"].append(fp)
                            print(f"  📝 File: {fp}")

                    # Track chat responses
                    if etype == "chat_response":
                        resp = data.get("response", "")[:80]
                        print(f"  💬 Manager: {resp}")

                    # Track session end
                    if "paused" in etype.lower() or "session" in etype.lower():
                        if "all_agents_paused" in etype or "session_logs" in str(data):
                            result["status"] = "completed"
                            print(f"  🏁 Session ended (all agents paused)")
                            break

                    # Check all dispatched agents done
                    if agents_dispatched and agent_completed >= agents_dispatched:
                        print(f"  🏁 All dispatched agents completed!")
                        result["status"] = "completed"
                        await asyncio.sleep(5)  # Let final events flush
                        break

                except asyncio.TimeoutError:
                    elapsed = time.time() - start
                    idle = time.time() - last_activity

                    # If idle for 30s after dispatch, probably done
                    if agents_dispatched and idle > 30:
                        result["status"] = "idle_completed"
                        print(f"  🏁 Idle 30s after dispatch — assuming done")
                        break

                    if not agents_dispatched and elapsed > 90:
                        result["status"] = "no_dispatch"
                        print(f"  ❌ No dispatch detected after 90s")
                        break

                    if elapsed > SESSION_TIMEOUT:
                        result["status"] = "timeout"
                        break

                    # Print progress every 15s
                    if int(elapsed) % 15 == 0 and int(elapsed) > 0:
                        print(f"  ⏳ {elapsed:.0f}s... dispatched={list(agents_dispatched)} done={list(agent_completed)}")

            # Pause all and wait for full wind-down
            print(f"  ⏸  Pausing all agents...")
            await ws.send(json.dumps({"action": "pause_all"}))
            await asyncio.sleep(8)  # Wait longer for agents to fully stop

    except Exception as exc:
        result["error"] = str(exc)
        result["status"] = "error"
        print(f"  ❌ Error: {exc}")

    result["ended_at"] = time.time()
    result["duration"] = result["ended_at"] - result["started_at"]
    result["agents_used"] = list(result["agents_used"])

    print(f"  📊 Status: {result['status']} | Duration: {result['duration']:.1f}s")
    print(f"     Agents: {result['agents_used']} | Files: {len(result['files_written'])}")

    return result


async def run_all_sessions():
    """Run all test sessions sequentially."""
    print("\n" + "🧪 " * 20)
    print("  SWARM AUTOMATED TEST HARNESS v2")
    print("🧪 " * 20)

    import urllib.request
    try:
        urllib.request.urlopen(f"{API_BASE}/api/status", timeout=3)
        print("\n✅ Server is running at localhost:8080")
    except Exception as exc:
        print(f"\n❌ Server not reachable: {exc}")
        print("   Start the swarm first, then re-run this script.")
        return

    existing_logs = set(glob.glob(f"{LOGS_DIR}/session_replay_*.jsonl"))
    print(f"📁 Existing session logs: {len(existing_logs)}")

    results = []
    for i, task in enumerate(TEST_TASKS, 1):
        result = await run_session(task, i)
        results.append(result)

        if i < len(TEST_TASKS):
            print(f"\n  ⏳ Waiting {PAUSE_BETWEEN_SESSIONS}s before next session...")
            await asyncio.sleep(PAUSE_BETWEEN_SESSIONS)

    new_logs = set(glob.glob(f"{LOGS_DIR}/session_replay_*.jsonl")) - existing_logs
    print(f"\n📁 New session logs created: {len(new_logs)}")
    for log in sorted(new_logs):
        size = os.path.getsize(log)
        print(f"   {os.path.basename(log)} ({size:,} bytes)")

    print(f"\n{'='*60}")
    print("  TEST SUMMARY")
    print(f"{'='*60}")
    for r in results:
        emoji = {"completed": "✅", "idle_completed": "✅", "timeout": "⏰", "error": "❌", "no_dispatch": "🚫"}.get(r["status"], "❓")
        print(f"  {emoji} {r['task']}: {r['status']} ({r['duration']:.1f}s) agents={r['agents_used']} files={len(r['files_written'])}")

    results_file = f"{LOGS_DIR}/test_results_{int(time.time())}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n📄 Results saved to: {results_file}")
    print(f"🔍 To analyze: python {__file__} --analyze")


# ─── Session Analyzer ───────────────────────────────────────────────────────

def analyze_sessions():
    """Analyze all session replay files."""
    print("\n" + "📊 " * 20)
    print("  SESSION REPLAY ANALYSIS")
    print("📊 " * 20)

    log_files = sorted(glob.glob(f"{LOGS_DIR}/session_replay_*.jsonl"), key=os.path.getmtime)
    if not log_files:
        print("No session replay files found!")
        return

    log_files = log_files[-5:]
    print(f"\nAnalyzing {len(log_files)} most recent sessions:\n")

    all_sessions = []

    for log_file in log_files:
        size = os.path.getsize(log_file)
        if size == 0:
            print(f"⚠️  {os.path.basename(log_file)}: EMPTY (0 bytes) — agents never started")
            continue

        with open(log_file) as f:
            events = [json.loads(line.strip()) for line in f if line.strip()]

        if not events:
            continue

        session = {
            "file": os.path.basename(log_file),
            "agents": {},
            "total_tokens": 0,
            "total_writes": 0,
            "total_reads": 0,
            "total_tools": 0,
            "total_blocked": 0,
        }

        for e in events:
            agent = e.get("agent", "")
            et = e.get("event_type", "")

            if agent and agent != "manager":
                if agent not in session["agents"]:
                    session["agents"][agent] = {
                        "tokens": 0, "reads": 0, "writes": 0,
                        "tools": 0, "blocked": 0, "iters": 0,
                        "tool_breakdown": {},
                    }
                a = session["agents"][agent]

                if et == "llm_call":
                    a["tokens"] = e.get("cumulative_tokens", a["tokens"])
                    a["iters"] = max(a["iters"], e.get("iteration", 0))
                elif et == "tool_execution":
                    a["tools"] += 1
                    session["total_tools"] += 1
                    tn = e.get("tool_name", "")
                    a["tool_breakdown"][tn] = a["tool_breakdown"].get(tn, 0) + 1
                    if tn == "read_file": a["reads"] += 1; session["total_reads"] += 1
                    if tn == "write_file" and e.get("success", True):
                        a["writes"] += 1; session["total_writes"] += 1
                    if not e.get("success", True):
                        a["blocked"] += 1; session["total_blocked"] += 1

            if et == "session_summary":
                session["total_tokens"] = e.get("total_tokens", 0)

        if not session["total_tokens"]:
            session["total_tokens"] = sum(a["tokens"] for a in session["agents"].values())

        all_sessions.append(session)

        print(f"{'─'*60}")
        print(f"📄 {session['file']}")
        print(f"   Tokens: {session['total_tokens']:,} | R={session['total_reads']} W={session['total_writes']} | Tools: {session['total_tools']} | Blocked: {session['total_blocked']}")

        for name, a in sorted(session["agents"].items()):
            tw = max(a["writes"], 0.1)
            print(f"   └─ {name}: {a['tokens']:,}tok {a['iters']}itr R={a['reads']} W={a['writes']} blocked={a['blocked']} tok/w={a['tokens']/tw:,.0f}")

    if len(all_sessions) > 1:
        print(f"\n{'='*60}")
        print("  CROSS-SESSION COMPARISON")
        print(f"{'='*60}")
        print(f"{'Session':<45} {'Tokens':>7} {'R':>3} {'W':>3} {'Blk':>4} {'Tok/W':>7}")
        print(f"{'─'*45} {'─'*7} {'─'*3} {'─'*3} {'─'*4} {'─'*7}")

        for s in all_sessions:
            tw = max(s["total_writes"], 0.1)
            print(f"{s['file']:<45} {s['total_tokens']:>7,} {s['total_reads']:>3} {s['total_writes']:>3} {s['total_blocked']:>4} {s['total_tokens']/tw:>7,.0f}")

        total_tok = sum(s["total_tokens"] for s in all_sessions)
        total_w = sum(s["total_writes"] for s in all_sessions)
        total_b = sum(s["total_blocked"] for s in all_sessions)
        print(f"{'TOTAL':<45} {total_tok:>7,} {'':>3} {total_w:>3} {total_b:>4} {total_tok/max(total_w,0.1):>7,.0f}")


if __name__ == "__main__":
    if "--analyze" in sys.argv:
        analyze_sessions()
    else:
        asyncio.run(run_all_sessions())
