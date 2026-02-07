from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set

from usc.api.pf0_block_bloom_footer_v0 import read_pf0_block_bloom_footer
from usc.api.odc2_pf0_v0 import pf0_decode_packet_indices
from usc.mem.block_bloom_index_v0 import query_blocks_for_keywords
from usc.mem.usc_recall_v0 import recall_from_packets_decoded, RecallResult


@dataclass
class _BBILite:
    """
    Minimal BlockBloomIndex-compatible object for query_blocks_for_keywords().
    That function expects bbi.block_blooms to be a DICT and iterates .items().
    """
    m_bits: int
    k_hashes: int
    group_size: int
    block_count: int
    block_blooms: Dict[int, bytes]


def pf0_block_smart_recall_from_blob(
    pf0_blob: bytes,
    packets_all: List[bytes],
    keywords: Set[str],
    require_all: bool = True,
) -> RecallResult:
    footer = read_pf0_block_bloom_footer(pf0_blob)

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

    # ✅ IMPORTANT FIX:
    # footer.block_blooms is a LIST, but query_blocks_for_keywords expects a DICT
    block_blooms_dict = {i: bits for i, bits in enumerate(footer.block_blooms)}

    bbi = _BBILite(
        m_bits=footer.m_bits,
        k_hashes=footer.k_hashes,
        group_size=footer.group_size,
        block_count=footer.block_count,
        block_blooms=block_blooms_dict,
    )

    block_ids = query_blocks_for_keywords(
        bbi,
        keywords,
        enable_stem=True,
        prefix_len=footer.prefix_len,
        require_all=require_all,
    )

    # Always include dict block (packet 0 lives in block 0)
    block_ids.add(0)

    # Blocks -> packet indices
    want: Set[int] = set()
    for bid in sorted(block_ids):
        base = bid * footer.group_size
        for j in range(footer.group_size):
            pi = base + j
            if pi < len(packets_all):
                want.add(pi)

    decoded = pf0_decode_packet_indices(pf0_blob, want)

    packets_part = [packets_all[0]] + [p for p in decoded if p != packets_all[0]]

    return recall_from_packets_decoded(
        packets_part=packets_part,
        keywords=keywords,
        require_all=require_all,
        prefix_len=footer.prefix_len,
        selected_packets=max(0, len(want) - 1),
        selected_blocks=len(block_ids),
        total_blocks=footer.block_count,
    )
