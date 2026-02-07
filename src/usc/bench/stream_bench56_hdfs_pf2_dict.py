import time
from typing import List

from usc.mem.hdfs_templates_v0 import HDFSTemplateBank, parse_hdfs_lines
from usc.mem.tpl_pf1_recall_v1 import build_tpl_pf1_blob as build_pf1, build_pf1_index, recall_event_id_index
from usc.mem.tpl_pf1_recall_v2_dict import build_tpl_pf2_blob, build_pf2_index, recall_event_id_pf2


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
    args = p.parse_args()

    lines = read_first_n_lines(args.log, args.lines)
    raw_text = "\n".join(lines) + "\n"
    raw_bytes = raw_text.encode("utf-8", errors="replace")

    bank = HDFSTemplateBank.from_csv(args.tpl)
    events, unknown = parse_hdfs_lines(lines, bank)

    with open(args.tpl, "r", errors="replace") as f:
        tpl_text = f.read()

    print("STREAM_BENCH56 — PF1 selective decode vs PF2 shared zstd dict")
    print(f"lines: {len(lines)}")
    print(f"RAW: {pretty(len(raw_bytes))}")
    print(f"packet_events: {args.packet_events}")
    print("-" * 60)

    t0 = time.perf_counter()
    blob1, meta1 = build_pf1(events, unknown, tpl_text, packet_events=args.packet_events, zstd_level=10)
    t_build1 = (time.perf_counter() - t0) * 1000

    idx1 = build_pf1_index(blob1)

    t0 = time.perf_counter()
    blob2, meta2 = build_tpl_pf2_blob(events, unknown, tpl_text, packet_events=args.packet_events, zstd_level=10)
    t_build2 = (time.perf_counter() - t0) * 1000

    idx2 = build_pf2_index(blob2)

    print(f"PF1 blob: {pretty(len(blob1))}  ratio={len(raw_bytes)/len(blob1):.2f}x  build={t_build1:.2f} ms  packets={meta1.packet_count}")
    print(f"PF2 blob: {pretty(len(blob2))}  ratio={len(raw_bytes)/len(blob2):.2f}x  build={t_build2:.2f} ms  packets={meta2.packet_count}  dict={pretty(meta2.dict_bytes)}")
    print("-" * 60)

    for eid in [3, 6, 14]:
        t0 = time.perf_counter()
        h1 = recall_event_id_index(idx1, blob1, eid, limit=args.limit)
        dt1 = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        h2 = recall_event_id_pf2(idx2, blob2, eid, limit=args.limit)
        dt2 = (time.perf_counter() - t0) * 1000

        print(f"E{eid:<2} PF1 hits={len(h1):<3} time={dt1:8.2f} ms   |   PF2 hits={len(h2):<3} time={dt2:8.2f} ms")
        if h2:
            print("  sample(PF2):", h2[0][:120])

    print("-" * 60)
    print("DONE ✅")


if __name__ == "__main__":
    main()
