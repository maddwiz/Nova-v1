import time

from usc.bench.datasets_mixed_tool_trace import mixed_tool_trace
from usc.mem.sas_dict_token_v1 import build_sas_packets_from_text
from usc.mem.sas_keyword_index_v0 import build_keyword_index
from usc.mem.block_bloom_index_v0 import build_block_bloom_index

from usc.api.odc2_pf0_v0 import pf0_encode_packets
from usc.api.pf0_twolevel_footer_v0 import PF0TwoLevelFooter, append_pf0_twolevel_footer

from usc.mem.pf0_twolevel_recall_v0 import (
    build_pf0_twolevel_ctx_from_blob,
    pf0_twolevel_smart_recall_from_blob,
    pf0_twolevel_smart_recall_cached,
)


def run():
    steps = 2400
    text = mixed_tool_trace(steps=steps, seed=7)

    packets = build_sas_packets_from_text(text, max_lines_per_packet=10, tok_top_k=0)

    prefix_len = 5
    group_size = 2

    kwi = build_keyword_index(packets, m_bits=1024, k_hashes=4, prefix_len=prefix_len)
    bbi = build_block_bloom_index(kwi.packet_blooms, kwi.m_bits, kwi.k_hashes, group_size=group_size)

    pf0_blob, pf0_meta = pf0_encode_packets(packets, group_size=group_size, zstd_level=10)

    footer = PF0TwoLevelFooter(
        m_bits=kwi.m_bits,
        k_hashes=kwi.k_hashes,
        prefix_len=prefix_len,
        group_size=group_size,
        packet_count=len(packets),
        block_count=len(bbi.block_blooms),
        block_blooms=bbi.block_blooms,
        packet_blooms=kwi.packet_blooms,
    )

    pf0_blob2 = append_pf0_twolevel_footer(pf0_blob, footer, compress=True)

    tests = [
        {"tool:finance", "kv:ticker=nvda"},
        {"tool:web.screenshot", "kv:ref_id=turn1view0"},
        {"tool:web.open", "n:lineno=120"},
        {"tool:web.click", "n:id=12"},
        {"tool:weather", "v:denver, co"},
    ]

    print("Bench50 — PF0 two-level cached vs uncached speed")
    print("PF0 blocks:", pf0_meta.block_count, "| raw bytes:", len(pf0_blob), "| +twolevel bytes:", len(pf0_blob2))
    print("------------------------------------------------------------")

    # build ctx once
    ctx = build_pf0_twolevel_ctx_from_blob(pf0_blob2)
    assert ctx is not None, "two-level footer missing"

    # warmup
    for kws in tests:
        _ = pf0_twolevel_smart_recall_cached(ctx, pf0_blob2, packets, kws, require_all=True)

    reps = 200

    # uncached timing
    t0 = time.perf_counter()
    hits_u = 0
    for _ in range(reps):
        for kws in tests:
            r = pf0_twolevel_smart_recall_from_blob(pf0_blob2, packets, kws, require_all=True)
            hits_u += len(r.matched_lines)
    t1 = time.perf_counter()

    # cached timing
    t2 = time.perf_counter()
    hits_c = 0
    for _ in range(reps):
        for kws in tests:
            r = pf0_twolevel_smart_recall_cached(ctx, pf0_blob2, packets, kws, require_all=True)
            hits_c += len(r.matched_lines)
    t3 = time.perf_counter()

    unc_ms = (t1 - t0) * 1000.0
    cached_ms = (t3 - t2) * 1000.0

    print(f"UNCACHED total_ms={unc_ms:8.2f}  (reps={reps} x {len(tests)} queries) hits={hits_u}")
    print(f"CACHED   total_ms={cached_ms:8.2f}  (reps={reps} x {len(tests)} queries) hits={hits_c}")
    print("------------------------------------------------------------")

    print(f"Speedup = {unc_ms / max(cached_ms, 1e-9):.2f}x")


if __name__ == "__main__":
    run()
