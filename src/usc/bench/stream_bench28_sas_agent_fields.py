from usc.bench.datasets_real_agent_trace import real_agent_trace
from usc.bench.metrics import gzip_compress
import zstandard as zstd

from usc.api.codec_odc import build_v3b_packets_from_text
from usc.api.codec_odc2_indexed import odc2_encode_packets

from usc.mem.sas_agent_fields_v0 import build_sas_packets_from_text


def _ratio(raw: int, comp: int) -> float:
    return raw / max(1, comp)


def run():
    loops = 900
    text = real_agent_trace(loops=loops, seed=7)
    raw = text.encode("utf-8")

    gz = gzip_compress(raw)
    zstd_plain = zstd.ZstdCompressor(level=10).compress(raw)

    # Baseline: USC v3b -> ODC2
    packets_v3b = build_v3b_packets_from_text(
        text,
        max_lines_per_chunk=60,
        window_chunks=1,
        level=10,
    )
    blob_v3b, meta_v3b = odc2_encode_packets(
        packets_v3b,
        level=10,
        dict_target_size=8192,
        sample_chunk_size=1024,
        group_size=8,
    )

    # New: SAS v2 -> ODC2
    packets_sas = build_sas_packets_from_text(
        text,
        max_lines_per_packet=60,
    )
    blob_sas, meta_sas = odc2_encode_packets(
        packets_sas,
        level=10,
        dict_target_size=8192,
        sample_chunk_size=1024,
        group_size=8,
    )

    print("USC Bench28 — SAS v2 (absolute base timestamp) vs v3b (ODC2 gs=8)")
    print("------------------------------------------------------------")
    print("RAW bytes            :", len(raw))
    print("GZIP bytes           :", f"{len(gz):7d}", f"({_ratio(len(raw), len(gz)):.2f}x)")
    print("ZSTD plain bytes     :", f"{len(zstd_plain):7d}", f"({_ratio(len(raw), len(zstd_plain)):.2f}x)")
    print("------------------------------------------------------------")
    print("ODC2 v3b blob bytes  :", f"{len(blob_v3b):7d}", f"({_ratio(len(raw), len(blob_v3b)):.2f}x)",
          f"blocks={meta_v3b.block_count} dict={meta_v3b.dict_bytes} mode={meta_v3b.used_mode}")
    print("ODC2 SAS blob bytes  :", f"{len(blob_sas):7d}", f"({_ratio(len(raw), len(blob_sas)):.2f}x)",
          f"blocks={meta_sas.block_count} dict={meta_sas.dict_bytes} mode={meta_sas.used_mode}")
    print("------------------------------------------------------------")
    print("Packets v3b:", len(packets_v3b))
    print("Packets SAS:", len(packets_sas))
    print("------------------------------------------------------------")


if __name__ == "__main__":
    run()
