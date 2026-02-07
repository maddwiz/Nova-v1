from __future__ import annotations

import json
import random
import uuid
from datetime import datetime, timezone, timedelta


def _iso(ts: datetime) -> str:
    return ts.astimezone(timezone.utc).isoformat(timespec="milliseconds")


RAW_THINK_LINES = [
    "thinking: evaluating next step",
    "note: storing intermediate result",
    "analysis: tool output seems relevant",
    "decision: refine query based on results",
    "memory: add key insight to scratchpad",
    "trace: step completed successfully",
    "warning: minor mismatch in expected format",
]


def mixed_tool_trace(steps: int = 1200, seed: int = 7) -> str:
    """
    Bursty mixed-tool trace (more realistic):
      - many RAW lines
      - tool calls happen in bursts
      - search -> open/click followups sometimes

    This creates tool locality so block skipping can shine.
    """
    rng = random.Random(seed)
    t0 = datetime(2026, 1, 20, 12, 0, 0, tzinfo=timezone.utc)

    queries = [
        "NVIDIA Blackwell specs",
        "latest OpenAI API pricing",
        "zstd dictionary training example",
        "DGX Spark firmware update notes",
        "SnapKV vs PyramidKV paper",
    ]
    urls = [
        "turn0search0",
        "turn0fetch1",
        "turn1view0",
        "https://example.com/docs",
    ]

    lines = []
    cur = t0

    def emit_raw(k: int):
        nonlocal cur
        for _ in range(k):
            cur = cur + timedelta(milliseconds=rng.randint(40, 250))
            lines.append(f"[{_iso(cur)}] {rng.choice(RAW_THINK_LINES)}")

    def emit_tool(tool: str, payload: dict):
        nonlocal cur
        rid = str(uuid.uuid4())
        cur = cur + timedelta(milliseconds=rng.randint(40, 250))
        j = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
        lines.append(f"[{_iso(cur)}] tool_call::{tool} rid={rid} payload={j}")

    i = 0
    while i < steps:
        # 1) Think chunk (raw lines)
        emit_raw(rng.randint(2, 8))
        i += 1

        # 2) Decide next burst type
        burst_roll = rng.random()

        # A) Web research burst (search + open/click followups)
        if burst_roll < 0.65:
            emit_tool("web.search_query", {"q": rng.choice(queries), "recency": rng.choice([7, 30, 90])})
            # followups 0-2
            for _ in range(rng.randint(0, 2)):
                emit_raw(rng.randint(1, 4))
                if rng.random() < 0.55:
                    emit_tool("web.open", {"ref_id": rng.choice(urls), "lineno": rng.choice([0, 40, 120])})
                else:
                    emit_tool("web.click", {"ref_id": rng.choice(urls), "id": rng.choice([1, 3, 7, 12, 18])})

            # rare screenshot
            if rng.random() < 0.08:
                emit_raw(rng.randint(1, 3))
                emit_tool("web.screenshot", {"ref_id": "turn1view0", "pageno": rng.choice([0, 1, 2, 3])})

        # B) Finance burst
        elif burst_roll < 0.82:
            emit_tool("finance", {"ticker": rng.choice(["NVDA", "AAPL", "MSFT", "TSLA"]), "type": "equity", "market": "USA"})

        # C) Weather burst
        else:
            emit_tool("weather", {"location": rng.choice(["Denver, CO", "San Francisco, CA", "Austin, TX"]), "duration": 7})

        emit_raw(rng.randint(1, 6))
        i += 1

    return "\n".join(lines[:steps])
