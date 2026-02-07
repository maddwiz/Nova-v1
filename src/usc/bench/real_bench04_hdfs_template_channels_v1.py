import gzip
import time
from typing import List

try:
    import zstandard as zstd
except Exception:
    zstd = None

from usc.mem.hdfs_templates_v0 import HDFSTemplateBank, parse_hdfs_lines
from usc.api.hdfs_template_codec_v0 import encode_and_compress
from usc.api.hdfs_template_codec_v1_channels import encode_and_compress_v1


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
    p.add_argument("--lines", type=int, default=100000)
    p.add_argument("--log", default="data/loghub/HDFS.log")
    p.add_argument("--tpl", default="data/loghub/preprocessed/HDFS.log_templates.csv")
    args = p.parse_args()

    lines = read_first_n_lines(args.log, args.lines)
    raw_text = "\n".join(lines) + "\n"
    raw_bytes = raw_text.encode("utf-8", errors="replace")

    print("REAL_BENCH04 — HDFS Template Mode (V0 vs V1 Channels)")
    print("log:", args.log)
    print("tpl:", args.tpl)
    print("lines:", len(lines))
    print("RAW:", pretty(len(raw_bytes)))
    print("------------------------------------------------------------")

    # Baselines
    t0 = time.perf_counter()
    gz = gzip.compress(raw_bytes, compresslevel=9)
    t1 = time.perf_counter()
    print(f"gzip-9      out={pretty(len(gz)):>10s}  ratio={len(raw_bytes)/len(gz):6.2f}x  time={(t1-t0)*1000:8.2f} ms")

    if zstd is not None:
        t0 = time.perf_counter()
        zs = zstd.ZstdCompressor(level=10).compress(raw_bytes)
        t1 = time.perf_counter()
        print(f"zstd-10     out={pretty(len(zs)):>10s}  ratio={len(raw_bytes)/len(zs):6.2f}x  time={(t1-t0)*1000:8.2f} ms")
    else:
        print("zstd-10     (skipped; install zstandard)")

    # Template parse
    bank = HDFSTemplateBank.load_from_csv(args.tpl)
    events, unknown = parse_hdfs_lines(lines, bank)
    ev_pairs = [(e.event_id, e.params) for e in events]

    # V0
    t0 = time.perf_counter()
    comp0, meta0 = encode_and_compress(ev_pairs, unknown, zstd_level=10)
    t1 = time.perf_counter()
    print("------------------------------------------------------------")
    print(f"HDFS-TPLv0  out={pretty(len(comp0)):>10s}  ratio={len(raw_bytes)/len(comp0):6.2f}x  time={(t1-t0)*1000:8.2f} ms")
    print(f"  structured_raw={pretty(meta0.raw_structured_bytes)}  events={meta0.event_count}  unknown={meta0.unknown_count}")

    # V1
    t0 = time.perf_counter()
    comp1, meta1 = encode_and_compress_v1(ev_pairs, unknown, zstd_level=10)
    t1 = time.perf_counter()
    print(f"HDFS-TPLv1  out={pretty(len(comp1)):>10s}  ratio={len(raw_bytes)/len(comp1):6.2f}x  time={(t1-t0)*1000:8.2f} ms")
    print(f"  structured_raw={pretty(meta1.raw_structured_bytes)}  events={meta1.event_count}  unknown={meta1.unknown_count}  channels={meta1.channel_count}")

    print("------------------------------------------------------------")
    print("DONE ✅")


if __name__ == "__main__":
    main()
