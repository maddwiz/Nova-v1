from __future__ import annotations

from dataclasses import dataclass
from typing import List, Set

from usc.api.pf0_bloom_footer_v0 import read_pf0_bloom_footer
from usc.api.odc2_pf0_v0 import pf0_decode_packet_indices
from usc.mem.sas_keyword_index_v0 import query_packets_for_keywords
from usc.mem.usc_recall_v0 import recall_from_packets_decoded, RecallResult


@dataclass
class _KwiLite:
    """
    Minimal SASKeywordIndex-compatible object for query_packets_for_keywords().
    We reuse the existing bloom query logic (same hashes/tokenization),
    which fixes the false-negative issue.
    """
    m_bits: int
    k_hashes: int
    prefix_len: int
    packet_blooms: List[bytes]


def pf0_smart_recall_from_blob(
    pf0_blob: bytes,
    packets_all: List[bytes],
    keywords: Set[str],
    require_all: bool = True,
) -> RecallResult:
    footer = read_pf0_bloom_footer(pf0_blob)

    # If no footer: brute decode all (fallback)
    if footer is None:
        decoded_all = pf0_decode_packet_indices(pf0_blob, set(range(len(packets_all))))
        packets_part = [packets_all[0]] + [p for p in decoded_all if p != packets_all[0]]
        return recall_from_packets_decoded(
            packets_part=packets_part,
            keywords=keywords,
            require_all=require_all,
            prefix_len=5,
            selected_packets=len(packets_part) - 1,
            selected_blocks=0,
            total_blocks=0,
        )

    kwi = _KwiLite(
        m_bits=footer.m_bits,
        k_hashes=footer.k_hashes,
        prefix_len=footer.prefix_len,
        packet_blooms=footer.packet_blooms,
    )

    # ✅ THIS IS THE KEY FIX:
    # reuse the real bloom query logic that matches how blooms were created
    pkt_ids = query_packets_for_keywords(
        kwi,
        packets_all,
        keywords,
        enable_stem=True,
        prefix_len=footer.prefix_len,
        require_all=require_all,
    )

    want = set(pkt_ids)
    want.add(0)  # dict always

    decoded = pf0_decode_packet_indices(pf0_blob, want)

    packets_part = [packets_all[0]] + [p for p in decoded if p != packets_all[0]]

    return recall_from_packets_decoded(
        packets_part=packets_part,
        keywords=keywords,
        require_all=require_all,
        prefix_len=footer.prefix_len,
        selected_packets=max(0, len(want) - 1),
        selected_blocks=0,
        total_blocks=footer.packet_count,
    )
