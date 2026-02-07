import time

from usc.bench.datasets_mixed_tool_trace import mixed_tool_trace
from usc.mem.sas_dict_token_v1 import build_sas_packets_from_text
from usc.mem.sas_keyword_index_v0 import build_keyword_index, query_packets_for_keywords
from usc.mem.block_bloom_index_v0 import build_block_bloom_index
from usc.api.odc2_sharded_v0 import odc2s_encode_packets
from usc.api.odc2s_bloom_footer_v0 import append_block_bloom_footer

from usc.mem.usc_recall_v0 import recall_from_odc2s, recall_from_packets_decoded
from usc.mem.usc_smart_recall_v1_twolevel import smart_recall_twolevel_from_blob

from usc.api.odc2_pf0_v0 import pf0_encode_packets, pf0_decode_packet_indices


def run():
    steps = 2400
    text = mixed_tool_trace(steps=steps, seed=7)

    packets = build_sas_packets_from_text(text, max_lines_per_packet=10, tok_top_k=0)

    # Keyword index once
    kwi = build_keyword_index(packets, m_bits=2048, k_hashes=4, prefix_len=5)

    # === ODC2S baseline blob (with footer blooms)
    blob, meta = odc2s_encode_packets(
        packets,
        group_size=2,
        dict_target_size=8192,
        zstd_level=10,
        sample_blocks=64,
    )

    bbi = build_block_bloom_index(kwi.packet_blooms, kwi.m_bits, kwi.k_hashes, group_size=2)
    blob2 = append_block_bloom_footer(blob, bbi, block_count=meta.block_count, compress=True)

    # === PF0 blob (packet-framed)
    pf0_blob, pf0_meta = pf0_encode_packets(packets, group_size=2, zstd_level=10)

    tests = [
        {"tool:finance", "kv:ticker=nvda"},
        {"tool:web.screenshot", "kv:ref_id=turn1view0"},
        {"tool:web.open", "n:lineno=120"},
        {"tool:web.click", "n:id=12"},
        {"tool:weather", "v:denver, co"},
    ]

    print("Bench46 — PF0 targeted packet decode vs ODC2S recalls")
    print("ODC2S blocks:", meta.block_count, "| blob bytes:", len(blob2))
    print("PF0   blocks:", pf0_meta.block_count, "| blob bytes:", len(pf0_blob))
    print("------------------------------------------------------------")

    for kws in tests:
        label = "+".join(sorted(kws))

        # ODC2S normal recall
        t0 = time.perf_counter()
        r0 = recall_from_odc2s(blob2, packets, kws, kwi=kwi, group_size=2, require_all=True)
        t1 = time.perf_counter()

        # ODC2S two-level recall
        t2 = time.perf_counter()
        r1 = smart_recall_twolevel_from_blob(blob2, packets, kws, kwi=kwi, require_all=True)
        t3 = time.perf_counter()

        # PF0 targeted packet decode:
        t4 = time.perf_counter()
        pkt_ids = query_packets_for_keywords(kwi, packets, kws, enable_stem=True, prefix_len=5, require_all=True)
        want = set(pkt_ids)
        want.add(0)  # dict always

        decoded = pf0_decode_packet_indices(pf0_blob, want)

        # ensure dict packet is first
        packets_part = [packets[0]] + [p for p in decoded if p != packets[0]]

        r2 = recall_from_packets_decoded(
            packets_part=packets_part,
            keywords=kws,
            require_all=True,
            prefix_len=5,
            selected_packets=max(0, len(want) - 1),
            selected_blocks=0,
            total_blocks=pf0_meta.block_count,
        )
        t5 = time.perf_counter()

        print(
            f"{label:55s} | normal_ms={(t1-t0)*1000:6.2f} hits={len(r0.matched_lines):3d}"
            f" | smart_ms={(t3-t2)*1000:6.2f} hits={len(r1.matched_lines):3d}"
            f" | pf0_ms={(t5-t4)*1000:6.2f} hits={len(r2.matched_lines):3d} pkts={len(want):3d}"
        )

    print("------------------------------------------------------------")


if __name__ == "__main__":
    run()
