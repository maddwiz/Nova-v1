import time

from usc.bench.datasets_mixed_tool_trace import mixed_tool_trace
from usc.mem.sas_dict_token_v1 import build_sas_packets_from_text
from usc.mem.sas_keyword_index_v0 import build_keyword_index
from usc.mem.block_bloom_index_v0 import build_block_bloom_index, query_blocks_for_keywords
from usc.api.odc2_sharded_v0 import odc2s_encode_packets, odc2s_decode_selected_blocks
from usc.api.odc2s_bloom_footer_v0 import append_block_bloom_footer, read_block_bloom_footer


def run():
    steps = 1200
    text = mixed_tool_trace(steps=steps, seed=7)

    packets = build_sas_packets_from_text(text, max_lines_per_packet=10, tok_top_k=0)

    blob, meta = odc2s_encode_packets(
        packets,
        group_size=2,
        dict_target_size=8192,
        zstd_level=10,
        sample_blocks=64,
    )

    kwi = build_keyword_index(packets, m_bits=2048, k_hashes=4, prefix_len=5)
    bbi = build_block_bloom_index(
        packet_blooms=kwi.packet_blooms,
        m_bits=kwi.m_bits,
        k_hashes=kwi.k_hashes,
        group_size=2,
    )

    blob2 = append_block_bloom_footer(blob, bbi, block_count=meta.block_count)

    print("Bench43 — Blob-only block selection via footer")
    print("Original blob bytes:", len(blob))
    print("Blob+footer bytes:", len(blob2))
    print("Blocks:", meta.block_count)
    print("------------------------------------------------------------")

    bbi2 = read_block_bloom_footer(blob2)
    if bbi2 is None:
        raise RuntimeError("Footer not found!")

    tests = [
        {"tool:web.screenshot", "kv:ref_id=turn1view0"},
        {"tool:finance", "kv:ticker=nvda"},
        {"tool:web.open", "n:lineno=120"},
        {"tool:web.click", "n:id=12"},
        {"tool:weather", "v:denver, co"},
    ]

    for kws in tests:
        ta = time.perf_counter()
        block_ids = query_blocks_for_keywords(bbi2, kws, require_all=True)
        tb = time.perf_counter()

        pct = 100.0 * len(block_ids) / max(1, meta.block_count)
        label = "+".join(sorted(kws))

        print(
            f"AND | {label:55s} | sel_blocks={len(block_ids):3d}/{meta.block_count:3d} ({pct:5.1f}%) | time_ms={(tb-ta)*1000:6.2f}"
        )

    print("------------------------------------------------------------")
    print("Decode-only sanity check (decode first selected block set):")
    print("------------------------------------------------------------")

    kws = {"tool:finance", "kv:ticker=nvda"}
    block_ids = query_blocks_for_keywords(bbi2, kws, require_all=True)
    packets_part, _ = odc2s_decode_selected_blocks(blob2[:len(blob)], block_ids=block_ids)  # decode base blob region only
    print("decoded packets:", len(packets_part))


if __name__ == "__main__":
    run()
