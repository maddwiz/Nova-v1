from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from usc.mem.stream_proto_canz_v3b import (
    StreamStateV3B,
    build_dict_state_from_chunks as build_v3b,
    encode_dict_packet as dict_v3b,
    apply_dict_packet as apply_v3b,
    encode_data_packet as data_v3b,
)

from usc.mem.stream_proto_canz_v3b_selfcontained import (
    StreamStateV3BSC,
    encode_data_packet as data_sc,
)

# Modes
MODE_V3B = "v3b"
MODE_V3BSC = "v3bSC"


@dataclass
class AutoSessionResult:
    mode: str
    dict_bytes: int
    data_bytes: int
    total_bytes: int


def _windows(items: List[str], win: int):
    for i in range(0, len(items), win):
        yield items[i:i + win]


def estimate_v3b_total(chunks: List[str], window_chunks: int, level: int = 10) -> AutoSessionResult:
    # build dict over whole session
    st_build = StreamStateV3B()
    build_v3b(chunks, state=st_build)
    pkt_dict = dict_v3b(st_build, level=level)

    # apply dict to sender state
    st_send = StreamStateV3B()
    apply_v3b(pkt_dict, state=st_send)

    data_total = 0
    for w in _windows(chunks, window_chunks):
        pkt = data_v3b(w, st_send, level=level)
        data_total += len(pkt)

    total = len(pkt_dict) + data_total
    return AutoSessionResult(mode=MODE_V3B, dict_bytes=len(pkt_dict), data_bytes=data_total, total_bytes=total)


def estimate_v3bsc_total(chunks: List[str], window_chunks: int, level: int = 10) -> AutoSessionResult:
    st_sc = StreamStateV3BSC()

    data_total = 0
    for w in _windows(chunks, window_chunks):
        pkt = data_sc(w, st_sc, level=level)
        data_total += len(pkt)

    total = data_total
    return AutoSessionResult(mode=MODE_V3BSC, dict_bytes=0, data_bytes=data_total, total_bytes=total)


def choose_best_session(chunks: List[str], window_chunks: int, level: int = 10) -> AutoSessionResult:
    a = estimate_v3b_total(chunks, window_chunks=window_chunks, level=level)
    b = estimate_v3bsc_total(chunks, window_chunks=window_chunks, level=level)
    return a if a.total_bytes <= b.total_bytes else b


def encode_session_packets(chunks: List[str], window_chunks: int, level: int = 10) -> Tuple[str, List[bytes]]:
    """
    Returns:
      mode, packets

    mode == "v3b"   -> packets = [DICT, DATA1, DATA2, ...]
    mode == "v3bSC" -> packets = [DATA1, DATA2, ...]  (each self-contained)
    """
    best = choose_best_session(chunks, window_chunks=window_chunks, level=level)

    if best.mode == MODE_V3B:
        st_build = StreamStateV3B()
        build_v3b(chunks, state=st_build)
        pkt_dict = dict_v3b(st_build, level=level)

        st_send = StreamStateV3B()
        apply_v3b(pkt_dict, state=st_send)

        packets = [pkt_dict]
        for w in _windows(chunks, window_chunks):
            packets.append(data_v3b(w, st_send, level=level))
        return MODE_V3B, packets

    st_sc = StreamStateV3BSC()
    packets = []
    for w in _windows(chunks, window_chunks):
        packets.append(data_sc(w, st_sc, level=level))
    return MODE_V3BSC, packets
