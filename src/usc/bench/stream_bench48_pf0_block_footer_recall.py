import time

from usc.bench.datasets_mixed_tool_trace import mixed_tool_trace
from usc.mem.sas_dict_token_v1 import build_sas_packets_from_text
from usc.mem.sas_keyword_index_v0 import build_keyword_index
from usc.mem.block_bloom_index_v0 import build_block_bloom_index

from usc.api.odc2_pf0_v0 import pf0_encode_packets
from usc.api.pf0_block_bloom_footer_v0 import PF0BlockBloomFooter, append_pf0_block_bloom_footer
from usc.mem.pf0_block_recall_v0 import pf0_block_smart_recall_from_blob


def run():
    steps = 2400
    text = mixed_tool_trace(steps=steps, seed=7)

    packets = build_sas_packets_from_text(text, max_lines_per_packet=10, tok_top_k=0)

    prefix_len = 5
    group_size = 2

    # ✅ stronger packet blooms -> stronger block blooms
    kwi = build_keyword_index(packets, m_bits=2048, k_hashes=4, prefix_len=prefix_len)

    pf0_blob, pf0_meta = pf0_encode_packets(packets, group_size=group_size, zstd_level=10)

    bbi = build_block_bloom_index(kwi.packet_blooms, kwi.m_bits, kwi.k_hashes, group_size=group_size)

    footer = PF0BlockBloomFooter(
        m_bits=kwi.m_bits,
        k_hashes=kwi.k_hashes,
        prefix_len=prefix_len,
        group_size=group_size,
        block_count=len(bbi.block_blooms),
        block_blooms=bbi.block_blooms,
    )

    pf0_blob2 = append_pf0_block_bloom_footer(pf0_blob, footer, compress=True)

    tests = [
        {"tool:finance", "kv:ticker=nvda"},
        {"tool:web.screenshot", "kv:ref_id=turn1view0"},
        {"tool:web.open", "n:lineno=120"},
        {"tool:web.click", "n:id=12"},
        {"tool:weather", "v:denver, co"},
    ]

    print("Bench48 — PF0 BLOCK footer recall (block bloom -> decode block packets)")
    print("PF0 blocks:", pf0_meta.block_count, "| raw bytes:", len(pf0_blob), "| +BLOCKfooter bytes:", len(pf0_blob2))
    print("------------------------------------------------------------")

    for kws in tests:
        label = "+".join(sorted(kws))
        t0 = time.perf_counter()
        r = pf0_block_smart_recall_from_blob(pf0_blob2, packets, kws, require_all=True)
        t1 = time.perf_counter()

        print(
            f"{label:55s} | pf0_block_ms={(t1-t0)*1000:6.2f} hits={len(r.matched_lines):3d}"
            f" sel_pkts={r.selected_packets:3d} sel_blocks={r.selected_blocks:3d}"
        )

    print("------------------------------------------------------------")


if __name__ == "__main__":
    run()
