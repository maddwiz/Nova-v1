#!/usr/bin/env python3
"""Cost-Per-Retrieval Killer Metric — shows compression ROI over time.

Upgrade #13: The key metric for enterprise adoption is not compression
ratio alone, but *cost per retrieval* — the amortized cost of storing
and retrieving a memory/session, which should DECREASE as the cogstore
learns more patterns.

Formula:
    cost_per_retrieval = (storage_cost + compute_cost) / retrievals

Where:
    storage_cost = compressed_bytes * $/byte/month
    compute_cost = encode_time * $/cpu-second + decode_time * $/cpu-second
    retrievals = number of times data has been retrieved (ref_count proxy)

As the cogstore grows:
    - More REFs → smaller compressed_bytes → lower storage_cost
    - Better predictions → faster encode → lower compute_cost
    - More retrievals → better amortization → lower cost_per_retrieval

This benchmark simulates 500 agent sessions and tracks how
cost_per_retrieval evolves — the "killer chart" for the paper.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Reuse the proven data generator from cogdedup session benchmark
from bench_cogdedup_sessions import generate_session

from usc.cogdedup.codec import cogdedup_encode, cogdedup_decode
from usc.cogdedup.store import MemoryCogStore
from usc.cogdedup.predictor import PredictiveCompressor


# Pricing assumptions (AWS-like, per month)
PRICE_PER_GB_MONTH = 0.023        # S3 standard
PRICE_PER_CPU_SECOND = 0.0000166  # Lambda-like pricing

NUM_SESSIONS = 500


def main():
    session_size_kb = 20
    total_mb = NUM_SESSIONS * session_size_kb / 1024
    print("=" * 80)
    print(" COST-PER-RETRIEVAL KILLER METRIC")
    print(f" {NUM_SESSIONS} sessions x {session_size_kb}KB = {total_mb:.1f} MB total")
    print("=" * 80)
    print()

    store = MemoryCogStore()
    predictor = PredictiveCompressor(store)

    results = []
    total_raw_bytes = 0
    total_compressed_bytes = 0
    total_encode_time = 0.0
    total_decode_time = 0.0

    for i in range(NUM_SESSIONS):
        session_data = generate_session(i, size_kb=session_size_kb)
        raw_size = len(session_data)
        total_raw_bytes += raw_size

        # Encode
        t0 = time.perf_counter()
        blob, stats = cogdedup_encode(session_data, store, predictor=predictor)
        t_encode = time.perf_counter() - t0

        # Decode (verify + time)
        t1 = time.perf_counter()
        decoded = cogdedup_decode(blob, store, predictor=predictor)
        t_decode = time.perf_counter() - t1

        assert decoded == session_data, f"Roundtrip failed at session {i}"

        compressed_size = len(blob)
        total_compressed_bytes += compressed_size
        total_encode_time += t_encode
        total_decode_time += t_decode

        # Retrievals = total ref_counts across store
        store_stats = store.stats()
        total_retrievals = store_stats["total_references"]

        # Calculate cost metrics
        storage_cost_monthly = (total_compressed_bytes / 1e9) * PRICE_PER_GB_MONTH
        compute_cost = (total_encode_time + total_decode_time) * PRICE_PER_CPU_SECOND
        total_cost = storage_cost_monthly + compute_cost

        cost_per_retrieval = total_cost / max(1, total_retrievals)
        ratio = total_raw_bytes / max(1, total_compressed_bytes)

        result = {
            "session": i + 1,
            "cumulative_ratio": round(ratio, 2),
            "session_ratio": round(raw_size / max(1, compressed_size), 2),
            "storage_cost_monthly_usd": round(storage_cost_monthly, 8),
            "compute_cost_usd": round(compute_cost, 8),
            "total_cost_usd": round(total_cost, 8),
            "total_retrievals": total_retrievals,
            "cost_per_retrieval_usd": round(cost_per_retrieval, 12),
            "cost_per_1k_retrievals_usd": round(cost_per_retrieval * 1000, 8),
            "ref_pct": round(stats["ref"] / max(1, stats["chunks"]) * 100, 1),
            "ref": stats["ref"],
            "delta": stats["delta"],
            "full": stats["full"],
            "pred_delta": stats.get("pred_delta", 0),
        }
        results.append(result)

        # Print at key milestones
        if i + 1 in [1, 2, 5, 10, 25, 50, 100, 200, 300, 400, 500]:
            print(f"  Session {i+1:>4}/{NUM_SESSIONS}: "
                  f"ratio={ratio:>6.1f}x  "
                  f"$/1K-ret=${result['cost_per_1k_retrievals_usd']:.6f}  "
                  f"refs={total_retrievals:>6}  "
                  f"storage=${storage_cost_monthly:.6f}/mo  "
                  f"REF={result['ref_pct']:>5.1f}%")

    print()
    print("=" * 80)
    print(" THE KILLER CHART: Cost Per 1K Retrievals Over Time")
    print("=" * 80)
    print()
    print(f"  {'Session':>8}  {'Ratio':>8}  {'$/1K-Ret':>12}  "
          f"{'Retrievals':>12}  {'Storage$/mo':>12}  {'REF%':>6}")
    print(f"  {'-'*8}  {'-'*8}  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*6}")

    milestones = [1, 5, 10, 25, 50, 100, 200, 500]
    for ms in milestones:
        if ms <= len(results):
            r = results[ms - 1]
            print(f"  {r['session']:>8}  {r['cumulative_ratio']:>7.1f}x  "
                  f"${r['cost_per_1k_retrievals_usd']:>10.6f}  "
                  f"{r['total_retrievals']:>12}  "
                  f"${r['storage_cost_monthly_usd']:>10.6f}  "
                  f"{r['ref_pct']:>5.1f}%")

    print()

    # Summary
    first = results[0]
    last = results[-1]
    improvement = first["cost_per_1k_retrievals_usd"] / max(1e-15, last["cost_per_1k_retrievals_usd"])

    print("=" * 80)
    print(" SUMMARY")
    print("=" * 80)
    print(f"  First session $/1K-ret:  ${first['cost_per_1k_retrievals_usd']:.6f}")
    print(f"  Last session  $/1K-ret:  ${last['cost_per_1k_retrievals_usd']:.6f}")
    print(f"  Cost reduction:          {improvement:.1f}x cheaper over {NUM_SESSIONS} sessions")
    print(f"  Final compression ratio: {last['cumulative_ratio']:.1f}x")
    print(f"  Total storage cost/mo:   ${last['storage_cost_monthly_usd']:.6f}")
    print(f"  Total retrievals:        {last['total_retrievals']}")
    print()

    # Save results
    out_dir = os.path.join(os.path.dirname(__file__), "..", "results", "cost_per_retrieval")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "cost_per_retrieval.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {out_file}")


if __name__ == "__main__":
    main()
