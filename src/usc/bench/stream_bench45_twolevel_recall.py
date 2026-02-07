import time

from usc.bench.datasets_mixed_tool_trace import mixed_tool_trace
from usc.mem.sas_dict_token_v1 import build_sas_packets_from_text
from usc.mem.sas_keyword_index_v0 import build_keyword_index
from usc.mem.block_bloom_index_v0 import build_block_bloom_index
from usc.api.odc2_sharded_v0 import odc2s_encode_packets
from usc.api.odc2s_bloom_footer_v0 import append_block_bloom_footer

from usc.mem.usc_recall_v0 import recall_from_odc2s
from usc.mem.usc_smart_recall_v0 import smart_recall_from_blob
from usc.mem.usc_smart_recall_v1_twolevel import smart_recall_twolevel_from_blob


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
    bbi = build_block_bloom_index(kwi.packet_blooms, kwi.m_bits, kwi.k_hashes, group_size=2)

    blob2 = append_block_bloom_footer(blob, bbi, block_count=meta.block_count, compress=True)

    tests = [
        {"tool:finance", "kv:ticker=nvda"},
        {"tool:web.screenshot", "kv:ref_id=turn1view0"},
        {"tool:web.open", "n:lineno=120"},
        {"tool:web.click", "n:id=12"},
        {"tool:weather", "v:denver, co"},
    ]

    print("Bench45 — normal vs smart-v0 vs smart-v1(two-level)")
    print("Blocks:", meta.block_count, "| blob bytes:", len(blob2))
    print("------------------------------------------------------------")

    for kws in tests:
        label = "+".join(sorted(kws))

        t0 = time.perf_counter()
        r0 = recall_from_odc2s(blob2, packets, kws, kwi=kwi, group_size=2, require_all=True)
        t1 = time.perf_counter()

        t2 = time.perf_counter()
        r1 = smart_recall_from_blob(blob2, packets, kws, kwi=kwi, require_all=True)
        t3 = time.perf_counter()

        t4 = time.perf_counter()
        r2 = smart_recall_twolevel_from_blob(blob2, packets, kws, kwi=kwi, require_all=True)
        t5 = time.perf_counter()

        print(
            f"{label:55s} | normal_ms={(t1-t0)*1000:6.2f} hits={len(r0.matched_lines):3d}"
            f" | smart0_ms={(t3-t2)*1000:6.2f} hits={len(r1.matched_lines):3d}"
            f" | smart1_ms={(t5-t4)*1000:6.2f} hits={len(r2.matched_lines):3d}"
        )

    print("------------------------------------------------------------")


if __name__ == "__main__":
    run()
