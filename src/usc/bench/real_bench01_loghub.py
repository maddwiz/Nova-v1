import os
import time
import gzip
from dataclasses import dataclass
from typing import Optional

try:
    import zstandard as zstd
except Exception:
    zstd = None

from usc.mem.sas_dict_token_v1 import build_sas_packets_from_text
from usc.api.odc2_sharded_v0 import odc2s_encode_packets
from usc.api.odc2_pf0_v0 import pf0_encode_packets


@dataclass
class BenchResult:
    name: str
    bytes_out: int
    ms: float


def read_first_n_lines(path: str, n: int) -> str:
    lines = []
    with open(path, "r", errors="replace") as f:
        for _ in range(n):
            line = f.readline()
            if not line:
                break
            lines.append(line.rstrip("\n"))
    return "\n".join(lines) + "\n"


def run_gzip(raw: bytes) -> BenchResult:
    t0 = time.perf_counter()
    out = gzip.compress(raw, compresslevel=9)
    ms = (time.perf_counter() - t0) * 1000.0
    return BenchResult("gzip-9", len(out), ms)


def run_zstd(raw: bytes) -> Optional[BenchResult]:
    if zstd is None:
        return None
    t0 = time.perf_counter()
    cctx = zstd.ZstdCompressor(level=10)
    out = cctx.compress(raw)
    ms = (time.perf_counter() - t0) * 1000.0
    return BenchResult("zstd-10", len(out), ms)


def run_odc2s(text: str, lines_per_packet: int) -> BenchResult:
    packets = build_sas_packets_from_text(text, max_lines_per_packet=lines_per_packet, tok_top_k=0)
    t0 = time.perf_counter()
    blob, meta = odc2s_encode_packets(
        packets,
        group_size=2,
        dict_target_size=8192,
        zstd_level=10,
        sample_blocks=64,
    )
    ms = (time.perf_counter() - t0) * 1000.0
    return BenchResult(f"ODC2S(L{lines_per_packet})", len(blob), ms)


def run_usc_best(text: str) -> BenchResult:
    """
    This is the tuned version (the "toy winner" style).
    We keep it deterministic and lossless, but push the knobs:
      - bigger trained dict target
      - more samples
      - higher zstd level
      - larger group size
      - use L50 for best structural capture
    """
    packets = build_sas_packets_from_text(text, max_lines_per_packet=50, tok_top_k=0)
    t0 = time.perf_counter()
    blob, meta = odc2s_encode_packets(
        packets,
        group_size=4,
        dict_target_size=16384,
        zstd_level=19,
        sample_blocks=256,
    )
    ms = (time.perf_counter() - t0) * 1000.0
    return BenchResult("USC_BEST(ODC2S-L50)", len(blob), ms)


def run_pf0(text: str, lines_per_packet: int) -> BenchResult:
    packets = build_sas_packets_from_text(text, max_lines_per_packet=lines_per_packet, tok_top_k=0)
    t0 = time.perf_counter()
    blob, meta = pf0_encode_packets(packets, group_size=2, zstd_level=10)
    ms = (time.perf_counter() - t0) * 1000.0
    return BenchResult(f"PF0(L{lines_per_packet})", len(blob), ms)


def pretty(bytes_out: int) -> str:
    if bytes_out >= 1_000_000:
        return f"{bytes_out/1_000_000:.2f} MB"
    if bytes_out >= 1_000:
        return f"{bytes_out/1_000:.2f} KB"
    return f"{bytes_out} B"


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--file", required=True, help="Path to a real log file")
    p.add_argument("--lines", type=int, default=20000, help="How many lines to read")
    args = p.parse_args()

    if not os.path.exists(args.file):
        raise SystemExit(f"File not found: {args.file}")

    print("REAL_BENCH01 — LogHub real logs (size + encode time)")
    print("file:", args.file)
    print("lines:", args.lines)
    print("------------------------------------------------------------")

    text = read_first_n_lines(args.file, args.lines)
    raw = text.encode("utf-8", errors="replace")

    print("RAW:", pretty(len(raw)))
    print("------------------------------------------------------------")

    results = []
    results.append(run_gzip(raw))

    zr = run_zstd(raw)
    if zr:
        results.append(zr)
    else:
        print("NOTE: zstandard not installed, skipping zstd")
        print("Install with: pip install zstandard")

    # Our "best" tuned mode (the one that was winning toy benches)
    results.append(run_usc_best(text))

    # Standard grids
    for lp in [10, 25, 50]:
        results.append(run_odc2s(text, lp))
        results.append(run_pf0(text, lp))

    results.sort(key=lambda r: r.bytes_out)
    best = results[0].bytes_out

    print("RESULTS (sorted by smallest output):")
    for r in results:
        ratio = len(raw) / max(r.bytes_out, 1)
        rel = r.bytes_out / best
        print(
            f"{r.name:22s}  out={pretty(r.bytes_out):>10s}  "
            f"ratio={ratio:6.2f}x  time={r.ms:8.2f} ms  vs_best={rel:5.2f}x"
        )

    print("------------------------------------------------------------")
    print("DONE ✅")


if __name__ == "__main__":
    main()
