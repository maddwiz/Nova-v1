import time
from typing import List

from usc.mem.hdfs_templates_v0 import HDFSTemplateBank, parse_hdfs_lines
from usc.mem.tpl_pf1_recall_v0 import build_tpl_pf1_blob, recall_event_id


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
    p.add_argument("--packet_events", type=int, default=4096)
    p.add_argument("--limit", type=int, default=25)
    args = p.parse_args()

    lines = read_first_n_lines(args.log, args.lines)
    raw_text = "\n".join(lines) + "\n"
    raw_bytes = raw_text.encode("utf-8", errors="replace")

    bank = HDFSTemplateBank.from_csv(args.tpl)
    events, unknown = parse_hdfs_lines(lines, bank)

    with open(args.tpl, "r", errors="replace") as f:
        tpl_text = f.read()

    print("STREAM_BENCH54 — HDFS Template PF1 Recall")
    print(f"lines: {len(lines)}")
    print(f"RAW: {pretty(len(raw_bytes))}")
    print(f"packet_events: {args.packet_events}")
    print("-" * 60)

    t0 = time.perf_counter()
    blob, meta = build_tpl_pf1_blob(
        events=events,
        unknown_lines=unknown,
        template_csv_text=tpl_text,
        packet_events=args.packet_events,
        zstd_level=10,
        bloom_bits=4096,
        bloom_k=3,
    )
    t_build = (time.perf_counter() - t0) * 1000.0

    print(f"PF1 blob: {pretty(len(blob))}  ratio={len(raw_bytes)/len(blob):.2f}x  build_time={t_build:.2f} ms  packets={meta.packet_count}")
    print("-" * 60)

    test_eids = [1, 3, 6, 14]
    for eid in test_eids:
        t0 = time.perf_counter()
        hits = recall_event_id(blob, eid, limit=args.limit)
        dt = (time.perf_counter() - t0) * 1000.0
        print(f"recall E{eid:<2}  hits={len(hits):<3}  time={dt:8.2f} ms")
        if hits:
            print("  sample:", hits[0][:120])

    print("-" * 60)
    print("DONE ✅")


if __name__ == "__main__":
    main()
