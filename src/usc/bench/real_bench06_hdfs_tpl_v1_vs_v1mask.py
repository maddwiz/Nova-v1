import gzip
import time
from typing import List

try:
    import zstandard as zstd
except Exception:
    zstd = None

from usc.mem.hdfs_templates_v0 import HDFSTemplateBank, parse_hdfs_lines
from usc.api.hdfs_template_codec_v1_channels import encode_and_compress_v1
from usc.api.hdfs_template_codec_v1_channels_mask import encode_and_compress_v1m


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

    print("REAL_BENCH06 — HDFS Template Channels (V1 vs V1-MASK)")
    print(f"lines: {len(lines)}")
    print(f"RAW: {pretty(len(raw_bytes))}")
    print("------------------------------------------------------------")

    # baselines
    t0 = time.perf_counter()
    gz = gzip.compress(raw_bytes, compresslevel=9)
    t_gz = (time.perf_counter() - t0) * 1000.0

    zs = None
    t_zs = 0.0
    if zstd is not None:
        t0 = time.perf_counter()
        zs = zstd.ZstdCompressor(level=10).compress(raw_bytes)
        t_zs = (time.perf_counter() - t0) * 1000.0

    # parse templates
    bank = HDFSTemplateBank.load_from_csv(args.tpl)
    events, unknown = parse_hdfs_lines(lines, bank)
    ev_pairs = [(e.event_id, e.params) for e in events]

    def ratio(out_len: int) -> float:
        return len(raw_bytes) / max(1, out_len)

    # V1
    t0 = time.perf_counter()
    out1, meta1 = encode_and_compress_v1(ev_pairs, unknown, zstd_level=10)
    t1 = (time.perf_counter() - t0) * 1000.0

    # V1-MASK
    t0 = time.perf_counter()
    outm, metam = encode_and_compress_v1m(ev_pairs, unknown, zstd_level=10)
    tm = (time.perf_counter() - t0) * 1000.0

    # print results
    print(f"gzip-9   out={pretty(len(gz)):>9}  ratio={ratio(len(gz)):6.2f}x  time={t_gz:8.2f} ms")
    if zs is not None:
        print(f"zstd-10  out={pretty(len(zs)):>9}  ratio={ratio(len(zs)):6.2f}x  time={t_zs:8.2f} ms")
    print("------------------------------------------------------------")
    print(f"TPLv1    out={pretty(len(out1)):>9}  ratio={ratio(len(out1)):6.2f}x  time={t1:8.2f} ms  structured_raw={pretty(meta1.raw_structured_bytes)}")
    print(f"TPLv1M   out={pretty(len(outm)):>9}  ratio={ratio(len(outm)):6.2f}x  time={tm:8.2f} ms  structured_raw={pretty(metam.raw_structured_bytes)}")
    print("------------------------------------------------------------")
    print("DONE ✅")


if __name__ == "__main__":
    main()
