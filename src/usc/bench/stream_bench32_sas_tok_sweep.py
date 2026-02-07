from usc.bench.datasets_real_agent_trace import real_agent_trace
import zstandard as zstd

from usc.api.codec_odc2_indexed import odc2_encode_packets
from usc.mem.sas_dict_token_v1 import build_sas_packets_from_text


def run():
    loops = 900
    text = real_agent_trace(loops=loops, seed=7)
    raw = text.encode("utf-8")

    print("USC Bench32 — SAS DictToken v1 tok_top_k sweep (header split)")
    print("RAW bytes:", len(raw))
    print("------------------------------------------------------------")

    for k in [0, 16, 32, 64, 128, 256]:
        packets = build_sas_packets_from_text(
            text,
            max_lines_per_packet=60,
            tok_top_k=k,
        )
        header = packets[0]
        data = packets[1:]

        header_comp = zstd.ZstdCompressor(level=6).compress(header)
        blob_data, meta = odc2_encode_packets(
            data,
            level=10,
            dict_target_size=8192,
            sample_chunk_size=1024,
            group_size=8,
        )
        total = len(header_comp) + len(blob_data)
        ratio = len(raw) / max(1, total)

        print(f"k={k:3d} | total={total:6d} bytes | ratio={ratio:5.2f}x | header_zstd={len(header_comp):5d} | data_dict={meta.dict_bytes:4d}")

    print("------------------------------------------------------------")


if __name__ == "__main__":
    run()
