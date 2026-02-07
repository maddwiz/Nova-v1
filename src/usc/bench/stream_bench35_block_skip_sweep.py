import time

from usc.bench.datasets_real_agent_trace import real_agent_trace
from usc.mem.sas_dict_token_v1 import build_sas_packets_from_text, _decode_dict_packet
from usc.mem.sas_index_v0 import build_index

from usc.api.odc2_sharded_v0 import (
    odc2s_encode_packets,
    odc2s_decode_selected_blocks,
    packet_indices_to_block_ids,
)


def run():
    loops = 900
    text = real_agent_trace(loops=loops, seed=7)

    want_tool = "web.search_query"

    print("USC Bench35 — Block-skip sweep (packet size + block group_size)")
    print("Tool:", want_tool)
    print("------------------------------------------------------------")

    for max_lines in [10, 15, 25, 60]:
        # Build SAS packets
        packets = build_sas_packets_from_text(
            text,
            max_lines_per_packet=max_lines,
            tok_top_k=0,
        )
        d = _decode_dict_packet(packets[0])

        want_tool_id = d.tool_to_id.get(want_tool, 0)
        if want_tool_id == 0:
            print("ERROR: tool not found:", want_tool)
            return

        idx = build_index(packets)
        want_packet_indices = set(idx.tool_to_packets.get(want_tool_id, []))

        for group_size in [1, 2, 4, 8]:
            # Encode sharded
            t0 = time.perf_counter()
            blob, meta = odc2s_encode_packets(
                packets,
                group_size=group_size,
                dict_target_size=8192,
                zstd_level=10,
                sample_blocks=64,
            )
            t1 = time.perf_counter()

            block_ids = packet_indices_to_block_ids(want_packet_indices, meta.group_size)
            sel = len(block_ids)
            total = meta.block_count
            pct = (100.0 * sel / max(1, total))

            # Decode only selected blocks (just to time it)
            t2 = time.perf_counter()
            _pkts_part, _ = odc2s_decode_selected_blocks(blob, block_ids=block_ids)
            t3 = time.perf_counter()

            print(
                f"max_lines={max_lines:2d} | group={group_size:2d} | packets={len(packets):3d} | "
                f"wanted_pkts={len(want_packet_indices):3d} | blocks={total:3d} | sel={sel:3d} ({pct:5.1f}%) | "
                f"enc_ms={(t1-t0)*1000:5.2f} | dec_sel_ms={(t3-t2)*1000:5.2f} | bytes={len(blob):6d}"
            )

        print("------------------------------------------------------------")


if __name__ == "__main__":
    run()
