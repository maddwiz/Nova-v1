from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

# v3AUTO v3 = ZERO overhead:
# - DICT packet is exactly v3b DICT packet (no wrapper)
# - DATA packet is exactly chosen subcodec DATA packet (no wrapper)

from usc.mem.stream_proto_canz_v3b import (
    StreamStateV3B,
    build_dict_state_from_chunks as build_v3b,
    encode_dict_packet as dict_v3b,
    apply_dict_packet as apply_v3b,
    encode_data_packet as data_v3b,
)

from usc.mem.stream_proto_canz_v3d_drain3 import (
    StreamStateV3D,
    build_dict_state_from_chunks as build_v3d6,
    encode_data_packet as data_v3d6,
)


@dataclass
class StreamStateV3AUTO:
    st_v3b: StreamStateV3B = field(default_factory=StreamStateV3B)
    st_v3d6: StreamStateV3D = field(default_factory=StreamStateV3D)


def build_dict_state_from_chunks(chunks: List[str], state: StreamStateV3AUTO) -> None:
    # build both encoder-side, but we only transmit v3b dict (v3d runs dictless)
    build_v3b(chunks, state=state.st_v3b)
    build_v3d6(chunks, state=state.st_v3d6)


def encode_dict_packet(state: StreamStateV3AUTO, level: int = 10) -> bytes:
    # ZERO overhead: dict packet == v3b dict packet
    return dict_v3b(state.st_v3b, level=level)


def apply_dict_packet(packet: bytes, state: StreamStateV3AUTO) -> None:
    # Apply v3b dict as usual
    apply_v3b(packet, state=state.st_v3b)

    # v3d6 dictless reset: it will learn templates/strings from its own data packets
    state.st_v3d6.templates = []
    state.st_v3d6.temp_index = {}
    state.st_v3d6.mtf = []
    state.st_v3d6.strings = []
    state.st_v3d6.str_index = {}
    state.st_v3d6.str_mtf = []
    state.st_v3d6.miner = None


def encode_data_packet(chunks: List[str], state: StreamStateV3AUTO, level: int = 10) -> bytes:
    # Compare sub-packet sizes and return the smaller packet directly.
    pkt_b = data_v3b(chunks, state.st_v3b, level=level)
    pkt_d = data_v3d6(chunks, state.st_v3d6, level=level)

    return pkt_b if len(pkt_b) <= len(pkt_d) else pkt_d
