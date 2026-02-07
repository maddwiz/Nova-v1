import time

from usc.bench.datasets_mixed_tool_trace import mixed_tool_trace
from usc.mem.sas_dict_token_v1 import build_sas_packets_from_text
from usc.mem.sas_keyword_index_v0 import build_keyword_index
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

    tests = [
        ({"tool:web.screenshot", "k:ref_id", "v:turn1view0"}, True),
        ({"tool:web.screenshot", "kv:ref_id=turn1view0"}, True),
        ({"tool:web.screenshot", "k:pageno", "n:pageno=2"}, True),

        ({"tool:finance", "k:ticker", "v:nvda"}, True),
        ({"tool:finance", "kv:ticker=nvda"}, True),

        ({"tool:web.click", "k:id", "n:id=12"}, True),
        ({"tool:web.open", "k:lineno", "n:lineno=120"}, True),

        ({"tool:weather", "k:location", "v:denver, co"}, True),
    ]

    print("USC Bench41 — Payload field query recall (k:/v:/kv:/n: tokens)")
    print(f"Preset: max_lines={max_lines} | group={group_size}")
    print("Packets:", len(packets), "| Blocks:", meta.block_count, "| Bytes:", len(blob))
    print("------------------------------------------------------------")

    for kws, require_all in tests:
        ta = time.perf_counter()
        res = recall_from_odc2s(
            blob=blob,
            packets_all=packets,
            keywords=kws,
            kwi=kwi,
            group_size=group_size,
            require_all=require_all,
        )
        tb = time.perf_counter()

        pct = 100.0 * res.selected_blocks / max(1, res.total_blocks)
        label = "+".join(sorted(kws))
        mode = "AND" if require_all else "OR"
        print(
            f"{mode:3s} | {label:70s} | sel_blocks={res.selected_blocks:3d}/{res.total_blocks:3d} ({pct:5.1f}%) | "
            f"sel_pkts={res.selected_packets:3d} | hits={len(res.matched_lines):3d} | time_ms={(tb-ta)*1000:6.2f}"
        )

    print("------------------------------------------------------------")


if __name__ == "__main__":
    run()
