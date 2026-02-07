from usc.bench.datasets_real_agent_trace import real_agent_trace
from usc.mem.sas_dict_token_v1 import build_sas_packets_from_text, _decode_dict_packet
from usc.mem.sas_index_v0 import build_index

from usc.api.odc2_sharded_v0 import (
    odc2s_encode_packets,
    packet_indices_to_block_ids,
)


def run():
    loops = 900
    text = real_agent_trace(loops=loops, seed=7)

    max_lines = 10
    group_size = 2

    packets = build_sas_packets_from_text(
        text,
        max_lines_per_packet=max_lines,
        tok_top_k=0,
    )
    d = _decode_dict_packet(packets[0])
    idx = build_index(packets)

    blob, meta = odc2s_encode_packets(
        packets,
        group_size=group_size,
        dict_target_size=8192,
        zstd_level=10,
        sample_blocks=64,
    )

    tools_to_test = [
        "web.search_query",
        "web.open",
        "web.click",
        "web.screenshot",
        "finance",
        "weather",
    ]

    print("USC Bench36 — Tool selectivity (% blocks needed)")
    print(f"Preset: max_lines={max_lines} | group={group_size}")
    print("Packets:", len(packets), "| Blocks:", meta.block_count, "| Bytes:", len(blob))
    print("------------------------------------------------------------")

    for tool in tools_to_test:
        tid = d.tool_to_id.get(tool, 0)
        if tid == 0:
            print(f"{tool:16s} | tool_id=0 (not present)")
            continue

        want_packet_indices = set(idx.tool_to_packets.get(tid, []))
        block_ids = packet_indices_to_block_ids(want_packet_indices, meta.group_size)

        sel = len(block_ids)
        total = meta.block_count
        pct = 100.0 * sel / max(1, total)

        print(
            f"{tool:16s} | tool_id={tid:2d} | wanted_pkts={len(want_packet_indices):3d} | "
            f"sel_blocks={sel:3d}/{total:3d} ({pct:5.1f}%)"
        )

    print("------------------------------------------------------------")


if __name__ == "__main__":
    run()
