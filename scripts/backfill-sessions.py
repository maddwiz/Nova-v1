#!/usr/bin/env python3
"""Backfill all existing sessions into Nova's searchable memory.

Finds all .jsonl session files across Claude Code and OpenClaw,
parses them, and ingests the content into SQLite FTS5 for keyword search.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_src = str(Path(__file__).resolve().parents[1] / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

from c3ae.config import Config
from c3ae.memory_spine.spine import MemorySpine

WATCH_DIRS = [
    Path.home() / ".openclaw" / "agents",
    Path.home() / ".claude" / "projects",
]
MIN_SIZE = 2000


def main():
    data_dir = Path.home() / "Nova-v1" / "data"
    config = Config()
    config.data_dir = data_dir
    config.ensure_dirs()
    spine = MemorySpine(config)

    # Find all sessions
    sessions = []
    for d in WATCH_DIRS:
        if d.exists():
            for f in d.rglob("*.jsonl"):
                if f.stat().st_size >= MIN_SIZE:
                    sessions.append(f)

    print(f"Found {len(sessions)} session files to ingest")
    total_chunks = 0
    start = time.time()

    for i, session_file in enumerate(sessions, 1):
        try:
            result = spine.ingest_session(session_file)
            n = result["chunks_ingested"]
            total_chunks += n
            print(f"  [{i}/{len(sessions)}] {session_file.name}: "
                  f"{n} chunks ({result['roles']})")
        except Exception as e:
            print(f"  [{i}/{len(sessions)}] ERROR {session_file.name}: {e}")

    elapsed = time.time() - start
    db_chunks = spine.sqlite.count_chunks()
    print(f"\nDone in {elapsed:.1f}s")
    print(f"  Sessions ingested: {len(sessions)}")
    print(f"  Total chunks: {total_chunks}")
    print(f"  DB total chunks: {db_chunks}")

    spine.sqlite.close()


if __name__ == "__main__":
    main()
