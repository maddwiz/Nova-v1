import time
from typing import List

from usc.mem.hdfs_templates_v0 import HDFSTemplateBank, parse_hdfs_lines
from usc.mem.tpl_pf1_recall_v1 import build_tpl_pf1_blob as build_pf1
from usc.mem.tpl_pfq1_query_v1 import build_pfq1_blob
from usc.mem.tpl_query_router_v1 import query_router_v1


def read_first_n_lines(path: str, n: int) -> List[str]:
    out = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
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

    with open(args.tpl, "r", encoding="utf-8", errors="replace") as f:
        tpl_text = f.read()

    print("STREAM_BENCH59 — USC Query Router (FAST → PFQ1 fallback)")
    print(f"lines: {len(lines)}")
    print(f"RAW: {pretty(len(raw_bytes))}")
    print(f"events: {len(events)}  unknown: {len(unknown)}")
    print(f"packet_events: {args.packet_events}")
    print(f"query: {args.query!r}")
    print("-" * 60)

    t0 = time.perf_counter()
    pf1_blob, _m1 = build_pf1(
        events, unknown, tpl_text,
        packet_events=args.packet_events,
        zstd_level=10
    )
    t_pf1 = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    pfq1_blob, _m2 = build_pfq1_blob(
        events, unknown, tpl_text,
        packet_events=args.packet_events,
        zstd_level=10,
        bloom_bits=8192,
        bloom_k=4
    )
    t_pfq1 = (time.perf_counter() - t0) * 1000.0

    print(f"PF1  blob: {pretty(len(pf1_blob))}  build={t_pf1:.2f} ms")
    print(f"PFQ1 blob: {pretty(len(pfq1_blob))}  build={t_pfq1:.2f} ms")
    print("-" * 60)

    t0 = time.perf_counter()
    hits, mode = query_router_v1(pf1_blob, pfq1_blob, args.query, limit=args.limit)
    dt = (time.perf_counter() - t0) * 1000.0

    print(f"router_mode={mode}  hits={len(hits)}  time={dt:.2f} ms")
    if hits:
        print("sample:")
        for h in hits[: min(len(hits), 3)]:
            print("  ", h[:140])

    print("-" * 60)
    print("DONE ✅")


if __name__ == "__main__":
    main()
