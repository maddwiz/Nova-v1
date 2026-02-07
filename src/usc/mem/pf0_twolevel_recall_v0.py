from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set

from usc.api.pf0_twolevel_footer_v0 import PF0TwoLevelFooter, read_pf0_twolevel_footer
from usc.api.odc2_pf0_v0 import pf0_decode_packet_indices
from usc.mem.block_bloom_index_v0 import query_blocks_for_keywords
from usc.mem.sas_keyword_index_v0 import query_packets_for_keywords
from usc.mem.usc_recall_v0 import recall_from_packets_decoded, RecallResult


@dataclass
class _BBILite:
    m_bits: int
    k_hashes: int
    group_size: int
    block_count: int
    block_blooms: Dict[int, bytes]


@dataclass
class _KwiLite:
    m_bits: int
    k_hashes: int
    prefix_len: int
    packet_blooms: List[bytes]


@dataclass
class PF0TwoLevelCtx:
    """
    Cached structures for PF0 two-level recall.
    Build once, reuse many times.
    """
    footer: PF0TwoLevelFooter
    bbi: _BBILite
    kwi: _KwiLite


def build_pf0_twolevel_ctx_from_blob(pf0_blob: bytes) -> Optional[PF0TwoLevelCtx]:
    """
    Parse the two-level footer once and prepare bloom query structures.
    """
    footer = read_pf0_twolevel_footer(pf0_blob)
    if footer is None:
        return None

    block_blooms_dict = {i: bits for i, bits in enumerate(footer.block_blooms)}

    bbi = _BBILite(
        m_bits=footer.m_bits,
        k_hashes=footer.k_hashes,
        group_size=footer.group_size,
        block_count=footer.block_count,
        block_blooms=block_blooms_dict,
    )

    kwi = _KwiLite(
        m_bits=footer.m_bits,
        k_hashes=footer.k_hashes,
        prefix_len=footer.prefix_len,
        packet_blooms=footer.packet_blooms,
    )

    return PF0TwoLevelCtx(footer=footer, bbi=bbi, kwi=kwi)


def pf0_twolevel_smart_recall_cached(
    ctx: PF0TwoLevelCtx,
    pf0_blob: bytes,
    packets_all: List[bytes],
    keywords: Set[str],
    require_all: bool = True,
) -> RecallResult:
    """
    FAST PATH:
      1) Block bloom -> select candidate blocks
      2) Blocks -> candidate packet IDs
      3) Packet refinement ONLY inside candidate packets ✅ (big speed win)
      4) Decode only refined packet IDs
    """
    footer = ctx.footer

    # ---- Level 1: block selection
    block_ids = query_blocks_for_keywords(
        ctx.bbi,
        keywords,
        enable_stem=True,
        prefix_len=footer.prefix_len,
        require_all=require_all,
    )
    block_ids.add(0)  # always include dict block

    # ---- Candidate packet IDs from block IDs
    candidate_ids: List[int] = []
    seen = set()

    for bid in sorted(block_ids):
        base = bid * footer.group_size
        for j in range(footer.group_size):
            pi = base + j
            if pi < footer.packet_count and pi not in seen:
                candidate_ids.append(pi)
                seen.add(pi)

    # Ensure dict packet always included
    if 0 not in seen:
        candidate_ids.insert(0, 0)

    # ---- Level 2: packet refinement ONLY within candidates ✅✅✅
    cand_packets = [packets_all[i] for i in candidate_ids]
    cand_blooms = [ctx.kwi.packet_blooms[i] for i in candidate_ids]

    kwi_sub = _KwiLite(
        m_bits=ctx.kwi.m_bits,
        k_hashes=ctx.kwi.k_hashes,
        prefix_len=ctx.kwi.prefix_len,
        packet_blooms=cand_blooms,
    )

    sub_hits = query_packets_for_keywords(
        kwi_sub,
        cand_packets,          # ✅ IMPORTANT: refine only candidate packets
        keywords,
        enable_stem=True,
        prefix_len=footer.prefix_len,
        require_all=require_all,
    )

    refined: Set[int] = set()
    for sub_i in sub_hits:
        if 0 <= sub_i < len(candidate_ids):
            refined.add(candidate_ids[sub_i])

    refined.add(0)  # dict packet always

    decoded = pf0_decode_packet_indices(pf0_blob, refined)
    packets_part = [packets_all[0]] + [p for p in decoded if p != packets_all[0]]

    return recall_from_packets_decoded(
        packets_part=packets_part,
        keywords=keywords,
        require_all=require_all,
        prefix_len=footer.prefix_len,
        selected_packets=max(0, len(refined) - 1),
        selected_blocks=len(block_ids),
        total_blocks=footer.block_count,
    )


def pf0_twolevel_smart_recall_from_blob(
    pf0_blob: bytes,
    packets_all: List[bytes],
    keywords: Set[str],
    require_all: bool = True,
) -> RecallResult:
    """
    Uncached version: builds ctx each call (slower), but uses same candidate-only refinement.
    """
    ctx = build_pf0_twolevel_ctx_from_blob(pf0_blob)
    if ctx is None:
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

    return pf0_twolevel_smart_recall_cached(
        ctx=ctx,
        pf0_blob=pf0_blob,
        packets_all=packets_all,
        keywords=keywords,
        require_all=require_all,
    )
