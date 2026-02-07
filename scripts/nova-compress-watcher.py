#!/usr/bin/env python3
"""Nova Session Compression Watcher — watches for new sessions and compresses them.

Monitors the OpenClaw and Claude Code session directories for new/modified .jsonl files
and compresses them through the cognitive dedup pipeline.

Runs as a systemd user service for persistent operation.
"""
from __future__ import annotations

import os
import sys
import time
import json
import hashlib
from pathlib import Path

# Ensure imports
_src = str(Path(__file__).resolve().parents[1] / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

from c3ae.config import Config
from c3ae.memory_spine.spine import MemorySpine

# Directories to watch
WATCH_DIRS = [
    Path.home() / ".openclaw" / "agents",
    Path.home() / ".claude" / "projects",
]

# Where to store compressed blobs
OUTPUT_DIR = Path.home() / "Nova-v1" / "data" / "compressed"

# State file to track what's been compressed
STATE_FILE = Path.home() / "Nova-v1" / "data" / "compress-state.json"

# Minimum file size to bother compressing
MIN_SIZE = 2000

# How often to scan (seconds)
SCAN_INTERVAL = 300  # 5 minutes


def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {}


def save_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2))


def file_hash(path: Path) -> str:
    """Quick hash of file size + mtime for change detection."""
    st = path.stat()
    return hashlib.md5(f"{st.st_size}:{st.st_mtime}".encode()).hexdigest()


def find_sessions() -> list[Path]:
    """Find all .jsonl session files."""
    sessions = []
    for watch_dir in WATCH_DIRS:
        if watch_dir.exists():
            for f in watch_dir.rglob("*.jsonl"):
                if f.stat().st_size >= MIN_SIZE:
                    sessions.append(f)
    return sessions


def compress_session(spine: MemorySpine, session_file: Path) -> dict | None:
    """Compress a single session file. Returns result dict or None on error."""
    try:
        data = session_file.read_bytes()
        session_id = session_file.stem
        result = spine.compress_session(data, session_id=session_id)

        # Save compressed blob
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = OUTPUT_DIR / f"{session_id}.ucog"
        out_path.write_bytes(result["blob"])

        ratio = result["original_size"] / max(1, result["compressed_size"])
        return {
            "file": str(session_file),
            "original_size": result["original_size"],
            "compressed_size": result["compressed_size"],
            "ratio": round(ratio, 1),
            "output": str(out_path),
        }
    except Exception as e:
        print(f"  ERROR compressing {session_file.name}: {e}", file=sys.stderr)
        return None


def run_once(spine: MemorySpine) -> int:
    """Scan for new/changed sessions and compress them. Returns count compressed."""
    state = load_state()
    sessions = find_sessions()
    compressed = 0

    for session_file in sessions:
        fpath = str(session_file)
        fhash = file_hash(session_file)

        # Skip if already compressed with same hash
        if fpath in state and state[fpath] == fhash:
            continue

        result = compress_session(spine, session_file)
        if result:
            state[fpath] = fhash
            compressed += 1
            print(f"  Compressed {session_file.name}: "
                  f"{result['original_size']:,} -> {result['compressed_size']:,} "
                  f"({result['ratio']}x)")

    if compressed:
        save_state(state)

    return compressed


def main():
    daemon = "--daemon" in sys.argv

    # Init spine
    data_dir = Path.home() / "Nova-v1" / "data"
    config = Config()
    config.data_dir = data_dir
    config.ensure_dirs()
    spine = MemorySpine(config)

    print(f"Nova compression watcher started")
    print(f"  Watching: {[str(d) for d in WATCH_DIRS]}")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Mode: {'daemon' if daemon else 'one-shot'}")

    if daemon:
        while True:
            try:
                count = run_once(spine)
                if count:
                    cs = spine.cogstore.stats()
                    print(f"  [{time.strftime('%H:%M')}] Compressed {count} sessions, "
                          f"cogstore: {cs.get('unique_chunks', 0)} chunks")
            except Exception as e:
                print(f"  Error in scan: {e}", file=sys.stderr)
            time.sleep(SCAN_INTERVAL)
    else:
        count = run_once(spine)
        cs = spine.cogstore.stats()
        print(f"\nDone: {count} sessions compressed, "
              f"cogstore: {cs.get('unique_chunks', 0)} chunks")

    spine.sqlite.close()


if __name__ == "__main__":
    main()
