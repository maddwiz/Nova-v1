import time
from typing import List

from usc.mem.hdfs_templates_v0 import HDFSTemplateBank, parse_hdfs_lines
from usc.mem.tpl_pf1_recall_v1 import build_tpl_pf1_blob as build_pf1
from usc.mem.tpl_pfq1_query_v1 import build_pfq1_blob, build_pfq1_index, query_keywords
from usc.mem.tpl_fast_query_v1 import query_fast_pf1


def read_first_n_lines(path: str, n: int) -> List[str]:
    out = []
    with open(path, "r", errors="replace") as f:
        for _ in range(n):
            ln = f.readline()
            if not ln:
                break
            out.append(ln.rstrip("\n"))
    return out


def pretty(n: int) -> str:
    if n >= 1_000_000:
        return f"{n/1_000_000:.2f} MB"
    if n >= 1_000:
        return f"{n/1_000:.2f} KB"
    return f"{n} B"


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--lines", type=int, default=200000)
    p.add_argument("--log", default="data/loghub/HDFS.log")
    p.add_argument("--tpl", default="data/loghub/preprocessed/HDFS.log_templates.csv")
    p.add_argument("--packet_events", type=int, default=32768)
    p.add_argument("--limit", type=int, default=25)
    p.add_argument("--query", default="IOException receiveBlock")
    args = p.parse_args()

    lines = read_first_n_lines(args.log, args.lines)
    raw_text = "\n".join(lines) + "\n"
    raw_bytes = raw_text.encode("utf-8", errors="replace")

    bank = HDFSTemplateBank.from_csv(args.tpl)
    events, unknown = parse_hdfs_lines(lines, bank)

    with open(args.tpl, "r", errors="replace") as f:
        tpl_text = f.read()

    print("STREAM_BENCH58 — FAST PF1 template-routed query vs PFQ1 bloom scan")
    print(f"lines: {len(lines)}")
    print(f"RAW: {pretty(len(raw_bytes))}")
    print(f"packet_events: {args.packet_events}")
    print(f"query: {args.query!r}")
    print("-" * 60)

    # Build PF1
    t0 = time.perf_counter()
    pf1_blob, _m1 = build_pf1(events, unknown, tpl_text, packet_events=args.packet_events, zstd_level=10)
    t_pf1 = (time.perf_counter() - t0) * 1000.0

    # Build PFQ1
    t0 = time.perf_counter()
    pfq1_blob, _m2 = build_pfq1_blob(events, unknown, tpl_text, packet_events=args.packet_events, zstd_level=10, bloom_bits=8192, bloom_k=4)
    t_pfq1 = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    pfq1_idx = build_pfq1_index(pfq1_blob)
    t_pfq1_idx = (time.perf_counter() - t0) * 1000.0

    print(f"PF1  blob: {pretty(len(pf1_blob))}  ratio={len(raw_bytes)/len(pf1_blob):.2f}x  build={t_pf1:.2f} ms")
    print(f"PFQ1 blob: {pretty(len(pfq1_blob))}  ratio={len(raw_bytes)/len(pfq1_blob):.2f}x  build={t_pfq1:.2f} ms  index={t_pfq1_idx:.2f} ms")
    print("-" * 60)

    # FAST query on PF1
    t0 = time.perf_counter()
    fast_hits, cands = query_fast_pf1(pf1_blob, args.query, limit=args.limit)
    t_fast = (time.perf_counter() - t0) * 1000.0

    # PFQ1 query
    t0 = time.perf_counter()
    pfq1_hits = query_keywords(pfq1_idx, pfq1_blob, args.query, limit=args.limit, require_all_terms=True)
    t_pfq1_q = (time.perf_counter() - t0) * 1000.0

    print(f"FAST PF1: hits={len(fast_hits)}  time={t_fast:.2f} ms  candidates={cands}")
    if fast_hits:
        print("  sample(FAST):", fast_hits[0][:140])

    print(f"PFQ1 scan: hits={len(pfq1_hits)}  time={t_pfq1_q:.2f} ms")
    if pfq1_hits:
        print("  sample(PFQ1):", pfq1_hits[0][:140])

    print("-" * 60)
    print("DONE ✅")


if __name__ == "__main__":
    main()
