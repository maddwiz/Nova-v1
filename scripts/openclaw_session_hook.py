#!/usr/bin/env python3
"""OpenClaw post-session hook — compress session transcript via C3/Ae MemorySpine.

Called by OpenClaw after each agent session ends. Reads the session JSONL,
compresses it via cognitive deduplication, and stores the compressed blob.

Usage:
    python openclaw_session_hook.py <session_jsonl_path> [--output-dir <dir>]

Environment:
    C3AE_DATA_DIR: Override C3/Ae data directory (default: c3ae/data/)
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Ensure packages are importable (monorepo src/ layout)
_src = str(Path(__file__).resolve().parents[1] / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

from c3ae.config import Config
from c3ae.memory_spine.spine import MemorySpine


def main() -> None:
    parser = argparse.ArgumentParser(description="Compress an OpenClaw session transcript")
    parser.add_argument("session_path", help="Path to session JSONL file")
    parser.add_argument("--output-dir", "-o", default="", help="Output directory for compressed blob")
    args = parser.parse_args()

    session_path = Path(args.session_path)
    if not session_path.exists():
        print(f"Error: {session_path} not found", file=sys.stderr)
        sys.exit(1)

    # Init spine
    config = Config()
    data_dir = os.environ.get("C3AE_DATA_DIR")
    if data_dir:
        config.data_dir = Path(data_dir)
    spine = MemorySpine(config)

    # Read and compress
    data = session_path.read_bytes()
    session_id = session_path.stem
    result = spine.compress_session(data, session_id=session_id)

    ratio = result["original_size"] / max(1, result["compressed_size"])
    print(f"Compressed {session_path.name}: "
          f"{result['original_size']:,} -> {result['compressed_size']:,} bytes "
          f"({ratio:.1f}x)")

    # Save compressed blob
    out_dir = Path(args.output_dir) if args.output_dir else session_path.parent
    out_path = out_dir / f"{session_id}.ucog"
    out_path.write_bytes(result["blob"])
    print(f"Saved: {out_path}")

    # Report anomalies
    stats = result["stats"]
    if "anomaly_alert" in stats:
        a = stats["anomaly_alert"]
        print(f"ANOMALY: {a['type']} (z={a['z_score']:.2f})")

    spine.sqlite.close()


if __name__ == "__main__":
    main()
