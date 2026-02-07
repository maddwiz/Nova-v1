import time
from typing import List

from usc.mem.hdfs_templates_v0 import HDFSTemplateBank, parse_hdfs_lines
from usc.mem.tpl_pf1_recall_v0 import build_tpl_pf1_blob as build_v0, recall_event_id as recall_v0
from usc.mem.tpl_pf1_recall_v1 import build_tpl_pf1_blob as build_v1, build_pf1_index, recall_event_id_index


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

    print("STREAM_BENCH55 — PF1 v0 bloom vs PF1 v1 exact EventID sets (CACHED INDEX)")
    print(f"lines: {len(lines)}")
    print(f"RAW: {pretty(len(raw_bytes))}")
    print(f"packet_events(v1): {args.packet_events}")
    print("-" * 60)

    # build v0
    t0 = time.perf_counter()
    blob0, meta0 = build_v0(events, unknown, tpl_text, packet_events=4096, zstd_level=10, bloom_bits=4096, bloom_k=3)
    t_build0 = (time.perf_counter() - t0) * 1000

    # build v1
    t0 = time.perf_counter()
    blob1, meta1 = build_v1(events, unknown, tpl_text, packet_events=args.packet_events, zstd_level=10)
    t_build1 = (time.perf_counter() - t0) * 1000

    # cache index for v1
    t0 = time.perf_counter()
    idx1 = build_pf1_index(blob1)
    t_index = (time.perf_counter() - t0) * 1000

    print(f"PF1 v0 blob: {pretty(len(blob0))}  ratio={len(raw_bytes)/len(blob0):.2f}x  build={t_build0:.2f} ms  packets={meta0.packet_count}")
    print(f"PF1 v1 blob: {pretty(len(blob1))}  ratio={len(raw_bytes)/len(blob1):.2f}x  build={t_build1:.2f} ms  packets={meta1.packet_count}  index={t_index:.2f} ms")
    print("-" * 60)

    test_eids = [3, 6, 14]
    for eid in test_eids:
        t0 = time.perf_counter()
        h0 = recall_v0(blob0, eid, limit=args.limit)
        dt0 = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        h1 = recall_event_id_index(idx1, blob1, eid, limit=args.limit)
        dt1 = (time.perf_counter() - t0) * 1000

        print(f"E{eid:<2}  v0 hits={len(h0):<3} time={dt0:8.2f} ms   |   v1 hits={len(h1):<3} time={dt1:8.2f} ms")
        if h1:
            print("  sample(v1):", h1[0][:120])

    print("-" * 60)
    print("DONE ✅")


if __name__ == "__main__":
    main()
