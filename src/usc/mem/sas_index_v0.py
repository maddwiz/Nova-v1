from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from usc.mem.sas_dict_token_v1 import (
    MAGIC_SASD,
    MAGIC_SASA,
    _uvarint_decode,
    _decode_dict_packet,
    _decode_data_packet,
)


@dataclass
class SASIndex:
    """
    packet_tools maps:
      packet_index -> set(tool_id) present in that packet
    tool_to_packets maps:
      tool_id -> list(packet_index) where tool_id appears
    """
    packet_tools: List[Set[int]]
    tool_to_packets: Dict[int, List[int]]
    total_packets: int


def build_index(packets: List[bytes]) -> SASIndex:
    """
    packets[0] is dict packet
    packets[1:] are data packets

    We scan each data packet and record which tool_ids appear.
    """
    if not packets:
        return SASIndex(packet_tools=[], tool_to_packets={}, total_packets=0)

    if not packets[0].startswith(MAGIC_SASD):
        raise ValueError("Not a SAS dict-token v1 stream")

    packet_tools: List[Set[int]] = []
    tool_to_packets: Dict[int, List[int]] = {}

    # data packets start at index 1
    for pi in range(1, len(packets)):
        pkt = packets[pi]
        if not pkt.startswith(MAGIC_SASA):
            # skip unknown packet types
            packet_tools.append(set())
            continue

        entries = _decode_data_packet(pkt)
        tools_here: Set[int] = set()

        for (_dt_us, _ts_style, tool_id, _rid16, _raw_body, _payload_tok) in entries:
            # tool_id 0 = raw line
            if tool_id != 0:
                tools_here.add(int(tool_id))

        packet_tools.append(tools_here)

        for tid in tools_here:
            tool_to_packets.setdefault(tid, []).append(pi)

    return SASIndex(packet_tools=packet_tools, tool_to_packets=tool_to_packets, total_packets=len(packets))


def selective_decode_lines(
    packets: List[bytes],
    include_tools: Optional[Set[str]] = None,
    include_raw_lines: bool = False,
) -> List[str]:
    """
    Selectively decode only lines matching tool calls in include_tools.
    If include_raw_lines=True, raw (non-tool) lines are included too.

    include_tools is a set of tool NAMES, e.g. {"web.search_query"}.
    """
    if not packets:
        return []

    # decode dict to get tool_id -> tool_name mapping
    d = _decode_dict_packet(packets[0])
    want_all_tools = (include_tools is None)

    out: List[str] = []

    # scan packets
    for pi in range(1, len(packets)):
        pkt = packets[pi]
        if not pkt.startswith(MAGIC_SASA):
            continue

        entries = _decode_data_packet(pkt)

        for (_dt_us, _ts_style, tool_id, rid16, raw_body, payload_tok) in entries:
            if tool_id == 0:
                if include_raw_lines:
                    out.append(raw_body.decode("utf-8", errors="replace"))
                continue

            # tool id -> tool name
            tid = int(tool_id)
            tool_name = "UNKNOWN"
            if 1 <= tid <= len(d.id_to_tool):
                tool_name = d.id_to_tool[tid - 1]

            if (not want_all_tools) and (tool_name not in include_tools):
                continue

            # Reconstruct tool_call line minimally (no timestamp rebuild needed for filtering)
            rid = rid16.hex()
            rid_fmt = f"{rid[0:8]}-{rid[8:12]}-{rid[12:16]}-{rid[16:20]}-{rid[20:32]}"
            payload = payload_tok.decode("utf-8", errors="replace")

            out.append(f"tool_call::{tool_name} rid={rid_fmt} payload={payload}")

    return out


def packets_for_tools(
    packets: List[bytes],
    tool_names: Set[str],
) -> List[bytes]:
    """
    Return only dict packet + data packets that contain any requested tools.
    This is for "partial stream decode" or partial transport.
    """
    if not packets:
        return []
    d = _decode_dict_packet(packets[0])

    # map wanted tool names -> ids
    want_ids: Set[int] = set()
    for name in tool_names:
        tid = d.tool_to_id.get(name)
        if tid:
            want_ids.add(int(tid))

    if not want_ids:
        return [packets[0]]  # dict only

    idx = build_index(packets)

    selected_packet_indices: Set[int] = set()
    for tid in want_ids:
        for pi in idx.tool_to_packets.get(tid, []):
            selected_packet_indices.add(pi)

    out = [packets[0]]
    for pi in sorted(selected_packet_indices):
        out.append(packets[pi])
    return out
