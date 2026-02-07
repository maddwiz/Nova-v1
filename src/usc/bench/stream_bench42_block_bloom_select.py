import time

from usc.bench.datasets_mixed_tool_trace import mixed_tool_trace
from usc.mem.sas_dict_token_v1 import build_sas_packets_from_text
from usc.mem.sas_keyword_index_v0 import build_keyword_index
from usc.mem.block_bloom_index_v0 import build_block_bloom_index, query_blocks_for_keywords
from usc.mem.usc_recall_v0 import recall_from_odc2s
from usc.api.odc2_sharded_v0 import odc2s_encode_packets


def run():
    steps = 1200
    text = mixed_tool_trace(steps=steps, seed=7)

    max_lines = 10
    group_size = 2

    packets = build_sas_packets_from_text(
        text,
        max_lines_per_packet=max_lines,
        tok_top_k=0,
    )

    blob, meta = odc2s_encode_packets(
        packets,
        group_size=group_size,
        dict_target_size=8192,
        zstd_level=10,
        sample_blocks=64,
    )

    kwi = build_keyword_index(
        packets,
        m_bits=2048,
        k_hashes=4,
        include_tool_names=True,
        include_raw_lines=True,
        include_payload_fields=True,
        enable_stem=True,
        prefix_len=5,
    )

    # Build block blooms from packet blooms
    bbi = build_block_bloom_index(
        packet_blooms=kwi.packet_blooms,
        m_bits=kwi.m_bits,
        k_hashes=kwi.k_hashes,
        group_size=group_size,
    )

    tests = [
        {"tool:web.screenshot", "kv:ref_id=turn1view0"},
        {"tool:finance", "kv:ticker=nvda"},
        {"tool:web.open", "n:lineno=120"},
        {"tool:web.click", "n:id=12"},
        {"tool:weather", "v:denver, co"},
    ]

    print("USC Bench42 — Block Bloom selection (no packet scan)")
    print(f"Preset: max_lines={max_lines} | group={group_size}")
    print("Packets:", len(packets), "| Blocks:", meta.block_count, "| Bytes:", len(blob))
    print("------------------------------------------------------------")

    for kws in tests:
        ta = time.perf_counter()
        block_ids = query_blocks_for_keywords(bbi, kws, require_all=True)
        tb = time.perf_counter()

        pct = 100.0 * len(block_ids) / max(1, meta.block_count)
        label = "+".join(sorted(kws))

        print(
            f"AND | {label:55s} | sel_blocks={len(block_ids):3d}/{meta.block_count:3d} ({pct:5.1f}%) | time_ms={(tb-ta)*1000:6.2f}"
        )

    print("------------------------------------------------------------")
    print("Now run recall (normal path) for comparison:")
    print("------------------------------------------------------------")

    for kws in tests:
        ta = time.perf_counter()
        res = recall_from_odc2s(
            blob=blob,
            packets_all=packets,
            keywords=kws,
            kwi=kwi,
            group_size=group_size,
            require_all=True,
        )
        tb = time.perf_counter()

        pct = 100.0 * res.selected_blocks / max(1, res.total_blocks)
        label = "+".join(sorted(kws))
        print(
            f"AND | {label:55s} | sel_blocks={res.selected_blocks:3d}/{res.total_blocks:3d} ({pct:5.1f}%) | hits={len(res.matched_lines):3d} | time_ms={(tb-ta)*1000:6.2f}"
        )

    print("------------------------------------------------------------")


if __name__ == "__main__":
    run()
