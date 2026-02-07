import gzip
import time
from typing import List

try:
    import zstandard as zstd
except Exception:
    zstd = None

from usc.mem.hdfs_templates_v0 import load_hdfs_template_bank, parse_hdfs_lines
from usc.api.hdfs_template_codec_v1_channels_mask import encode_and_compress_v1m
from usc.api.hdfs_template_codec_v1m_bundle import bundle_encode_and_compress_v1m


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
    args = p.parse_args()

    lines = read_first_n_lines(args.log, args.lines)
    raw_text = "\n".join(lines) + "\n"
    raw_bytes = raw_text.encode("utf-8", errors="replace")

    bank = load_hdfs_template_bank(args.tpl)
    events, unknown = parse_hdfs_lines(lines, bank)

    # read template csv as text for bundling
    with open(args.tpl, "r", errors="replace") as f:
        tpl_text = f.read()

    print("REAL_BENCH07 — HDFS V1M vs V1M-BUNDLE (self contained)")
    print(f"lines: {args.lines}")
    print(f"RAW: {pretty(len(raw_bytes))}")
    print("-" * 60)

    # baselines
    t0 = time.perf_counter()
    gz = gzip.compress(raw_bytes, compresslevel=9)
    tgz = (time.perf_counter() - t0) * 1000
    print(f"gzip-9   out={pretty(len(gz)):>9}  ratio={len(raw_bytes)/len(gz):6.2f}x  time={tgz:8.2f} ms")

    if zstd is not None:
        cctx = zstd.ZstdCompressor(level=10)
        t0 = time.perf_counter()
        z = cctx.compress(raw_bytes)
        tz = (time.perf_counter() - t0) * 1000
        print(f"zstd-10  out={pretty(len(z)):>9}  ratio={len(raw_bytes)/len(z):6.2f}x  time={tz:8.2f} ms")

    print("-" * 60)

    t0 = time.perf_counter()
    blob1, meta1 = encode_and_compress_v1m(events, unknown, zstd_level=10)
    t1 = (time.perf_counter() - t0) * 1000
    print(f"TPLv1M   out={pretty(len(blob1)):>9}  ratio={len(raw_bytes)/len(blob1):6.2f}x  time={t1:8.2f} ms  structured_raw={pretty(meta1.raw_structured_bytes)}")

    t0 = time.perf_counter()
    blob2, meta2 = bundle_encode_and_compress_v1m(events, unknown, tpl_text, zstd_level=10)
    t2 = (time.perf_counter() - t0) * 1000
    print(f"BUNDLE   out={pretty(len(blob2)):>9}  ratio={len(raw_bytes)/len(blob2):6.2f}x  time={t2:8.2f} ms  bundle_bytes={pretty(meta2.bundle_bytes)}")

    overhead = len(blob2) - len(blob1)
    print("-" * 60)
    print(f"overhead: {pretty(overhead)}  ({(overhead/len(blob1))*100:.3f}% vs tpl-only)")
    print("DONE ✅")


if __name__ == "__main__":
    main()
