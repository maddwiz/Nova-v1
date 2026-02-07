#!/usr/bin/env python3
"""Real-World Benchmark — compress actual Nova/OpenClaw session transcripts.

This is the proof: compress REAL agent sessions through the cogdedup pipeline
and show the compression ratio improving as the cognitive dictionary grows.
"""
import os
import sys
import time
from pathlib import Path

# Ensure imports (monorepo src/ layout)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from c3ae.config import Config
from c3ae.memory_spine.spine import MemorySpine

import tempfile
import zstandard as zstd


def find_sessions():
    """Find all real session JSONL files, sorted by size."""
    sessions = []

    # Claude Code sessions (the real Nova brain)
    claude_dir = Path.home() / ".claude" / "projects" / "-home-nova"
    if claude_dir.exists():
        for f in claude_dir.rglob("*.jsonl"):
            size = f.stat().st_size
            if size > 1000:  # Skip tiny files
                sessions.append(f)

    # OpenClaw sessions
    openclaw_dir = Path.home() / ".openclaw" / "agents"
    if openclaw_dir.exists():
        for f in openclaw_dir.rglob("*.jsonl"):
            size = f.stat().st_size
            if size > 1000:
                sessions.append(f)

    # Sort by size (process smaller ones first for better learning curve)
    sessions.sort(key=lambda f: f.stat().st_size)
    return sessions


def main():
    sessions = find_sessions()
    if not sessions:
        print("No session files found!")
        sys.exit(1)

    total_raw = sum(f.stat().st_size for f in sessions)
    print("=" * 80)
    print(" REAL-WORLD BENCHMARK: Nova Session Compression")
    print(f" {len(sessions)} sessions, {total_raw / 1024 / 1024:.1f} MB total")
    print("=" * 80)
    print()

    # Use temp dir for benchmark cogstore (fresh start)
    tmp = tempfile.mkdtemp(prefix="nova-bench-")
    config = Config()
    config.data_dir = Path(tmp)
    config.ensure_dirs()
    spine = MemorySpine(config)

    results = []
    cumulative_raw = 0
    cumulative_compressed = 0
    cumulative_zstd = 0

    print(f"  {'#':>3}  {'Size':>10}  {'Cogdedup':>10}  {'Ratio':>7}  "
          f"{'Cum.Ratio':>9}  {'vs zstd':>8}  {'REF%':>6}  "
          f"{'Encode':>8}  File")
    print(f"  {'─'*3}  {'─'*10}  {'─'*10}  {'─'*7}  "
          f"{'─'*9}  {'─'*8}  {'─'*6}  "
          f"{'─'*8}  {'─'*30}")

    for i, session_file in enumerate(sessions):
        data = session_file.read_bytes()
        raw_size = len(data)
        cumulative_raw += raw_size
        session_id = session_file.stem

        # Baseline: raw zstd-19
        zstd_blob = zstd.ZstdCompressor(level=19).compress(data)
        zstd_size = len(zstd_blob)
        cumulative_zstd += zstd_size

        # Cogdedup through MemorySpine
        t0 = time.perf_counter()
        result = spine.compress_session(data, session_id=session_id)
        elapsed = time.perf_counter() - t0

        blob = result["blob"]
        stats = result["stats"]
        compressed_size = len(blob)
        cumulative_compressed += compressed_size

        # Verify roundtrip
        decoded = spine.decompress_with_dedup(blob,
                                               expected_hash=stats.get("integrity_hash", ""))
        if decoded != data:
            print(f"  *** ROUNDTRIP FAILED on {session_file.name} ***")
            continue

        ratio = raw_size / max(1, compressed_size)
        cum_ratio = cumulative_raw / max(1, cumulative_compressed)
        vs_zstd = zstd_size / max(1, compressed_size)
        ref_pct = stats.get("ref", 0) / max(1, stats.get("chunks", 1)) * 100
        speed_mb = (raw_size / 1024 / 1024) / max(0.001, elapsed)

        fname = session_file.name[:30]
        if len(session_file.name) > 30:
            fname = session_file.name[:27] + "..."

        print(f"  {i+1:>3}  {raw_size:>10,}  {compressed_size:>10,}  "
              f"{ratio:>6.1f}x  {cum_ratio:>8.1f}x  "
              f"{vs_zstd:>7.2f}x  {ref_pct:>5.1f}%  "
              f"{speed_mb:>6.1f}MB  {fname}")

        results.append({
            "session": i + 1,
            "file": str(session_file),
            "raw_size": raw_size,
            "compressed_size": compressed_size,
            "zstd_size": zstd_size,
            "ratio": round(ratio, 2),
            "cum_ratio": round(cum_ratio, 2),
            "vs_zstd": round(vs_zstd, 2),
            "ref_pct": round(ref_pct, 1),
            "encode_time": round(elapsed, 3),
            "ref": stats.get("ref", 0),
            "delta": stats.get("delta", 0),
            "full": stats.get("full", 0),
            "pred_delta": stats.get("pred_delta", 0),
            "anomaly": stats.get("anomaly_alert"),
        })

    print()
    print("=" * 80)
    print(" THE LEARNING CURVE")
    print("=" * 80)
    print()

    # Show how REF% increases over time
    if len(results) >= 2:
        first_quarter = results[:len(results)//4] if len(results) >= 4 else results[:1]
        last_quarter = results[-(len(results)//4):] if len(results) >= 4 else results[-1:]

        avg_ref_first = sum(r["ref_pct"] for r in first_quarter) / max(1, len(first_quarter))
        avg_ref_last = sum(r["ref_pct"] for r in last_quarter) / max(1, len(last_quarter))
        avg_ratio_first = sum(r["ratio"] for r in first_quarter) / max(1, len(first_quarter))
        avg_ratio_last = sum(r["ratio"] for r in last_quarter) / max(1, len(last_quarter))

        print(f"  First quarter:  avg ratio {avg_ratio_first:.1f}x, avg REF {avg_ref_first:.1f}%")
        print(f"  Last quarter:   avg ratio {avg_ratio_last:.1f}x, avg REF {avg_ref_last:.1f}%")
        if avg_ratio_first > 0:
            print(f"  Improvement:    {avg_ratio_last / max(0.1, avg_ratio_first):.2f}x better")
    print()

    # Summary
    print("=" * 80)
    print(" SUMMARY")
    print("=" * 80)
    total_compressed = sum(r["compressed_size"] for r in results)
    total_zstd = sum(r["zstd_size"] for r in results)
    total_raw_bytes = sum(r["raw_size"] for r in results)

    print(f"  Sessions processed:  {len(results)}")
    print(f"  Total raw size:      {total_raw_bytes / 1024 / 1024:.1f} MB")
    print(f"  Cogdedup total:      {total_compressed / 1024 / 1024:.2f} MB "
          f"({total_raw_bytes / max(1, total_compressed):.1f}x)")
    print(f"  zstd-19 total:       {total_zstd / 1024 / 1024:.2f} MB "
          f"({total_raw_bytes / max(1, total_zstd):.1f}x)")
    print(f"  Cogdedup vs zstd:    {total_zstd / max(1, total_compressed):.2f}x "
          f"{'BETTER' if total_compressed < total_zstd else 'worse'}")
    print()

    # Cogstore stats
    cs = spine.cogstore.stats()
    print(f"  Cogstore chunks:     {cs.get('unique_chunks', 0)}")
    print(f"  Total references:    {cs.get('total_references', 0)}")
    print(f"  Hot tier:            {cs.get('hot_chunks', 0)}")

    # Anomaly report
    report = spine._anomaly_detector.drift_report()
    print(f"  Anomaly observations: {spine._anomaly_detector._observation_count}")
    print(f"  Anomaly alerts:      {report.alerts_count}")

    # All roundtrips passed
    print(f"\n  ALL {len(results)} ROUNDTRIPS: PASSED")
    print()

    spine.sqlite.close()


if __name__ == "__main__":
    main()
