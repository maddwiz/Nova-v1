from usc.bench.datasets_real_agent_trace import real_agent_trace
from usc.bench.metrics import gzip_compress
import zstandard as zstd

from usc.api.codec_odc import (
    build_v3b_packets_from_text,
    odc_encode_packets,
)

from usc.api.codec_odc2_indexed import (
    odc2_encode_packets,
)


def _ratio(raw: int, comp: int) -> float:
    return raw / max(1, comp)


def run():
    loops = 900
    text = real_agent_trace(loops=loops, seed=7)
    raw = text.encode("utf-8")

    gz = gzip_compress(raw)

    # zstd baseline (plain, no dict)
    zc = zstd.ZstdCompressor(level=10)
    zstd_plain = zc.compress(raw)

    # build USC packets once
    packets = build_v3b_packets_from_text(
        text,
        max_lines_per_chunk=60,
        window_chunks=1,
        level=10,
    )

    # ODC1 (best ratio, no selective decode)
    blob1, meta1 = odc_encode_packets(
        packets,
        level=10,
        dict_target_size=8192,
        sample_chunk_size=1024,
    )

    print("USC Bench24 — ODC1 vs ODC2 sweep (group_size tradeoff)")
    print("------------------------------------------------------------")
    print(f"RAW bytes        : {len(raw)}")
    print(f"GZIP bytes       : {len(gz):>7} ({_ratio(len(raw), len(gz)):.2f}x)")
    print(f"ZSTD plain bytes : {len(zstd_plain):>7} ({_ratio(len(raw), len(zstd_plain)):.2f}x)")
    print("------------------------------------------------------------")
    print(f"ODC1 blob bytes  : {len(blob1):>7} ({_ratio(len(raw), len(blob1)):.2f}x) dict={meta1.dict_bytes} packets={meta1.packets}")
    print("------------------------------------------------------------")

    for gs in [2, 4, 8, 16]:
        blob2, meta2 = odc2_encode_packets(
            packets,
            level=10,
            dict_target_size=8192,
            sample_chunk_size=1024,
            group_size=gs,
        )
        print(
            f"ODC2 gs={gs:<2} blob={len(blob2):>7}  "
            f"ratio={_ratio(len(raw), len(blob2)):.2f}x  "
            f"blocks={meta2.block_count:<3}  "
            f"dict={meta2.dict_bytes:<5}  "
            f"mode={meta2.used_mode}"
        )

    print("------------------------------------------------------------")


if __name__ == "__main__":
    run()
