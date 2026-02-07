#!/usr/bin/env python3
"""Nova Memory Server — FastAPI service for memory recall and search.

Runs the C3/Ae memory API on localhost:8420. This is how agents (OpenClaw,
Claude Code, etc.) query Nova's memory.

Usage:
    python scripts/nova-memory-server.py              # Run on default port
    python scripts/nova-memory-server.py --port 8420  # Custom port
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure src is on path
_src = str(Path(__file__).resolve().parents[1] / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

import argparse
import uvicorn
from c3ae.api.routes import create_app


def main():
    parser = argparse.ArgumentParser(description="Nova Memory Server")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address")
    parser.add_argument("--port", type=int, default=8420, help="Port")
    parser.add_argument("--data-dir", default=str(Path.home() / "Nova-v1" / "data"),
                        help="Data directory")
    args = parser.parse_args()

    app = create_app(data_dir=args.data_dir)

    print(f"Nova Memory Server starting on {args.host}:{args.port}")
    print(f"  Data: {args.data_dir}")
    print(f"  Endpoints:")
    print(f"    POST /api/v1/memory/recall   - Search memories")
    print(f"    POST /api/v1/memory/search   - Hybrid search")
    print(f"    POST /api/v1/memory/ingest   - Ingest text")
    print(f"    POST /api/v1/sessions/ingest - Ingest session file")
    print(f"    GET  /api/v1/sessions/list   - List sessions")
    print(f"    GET  /api/v1/status/full     - Full status")
    print(f"    GET  /api/v1/health          - Health check")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
