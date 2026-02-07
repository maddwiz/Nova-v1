from __future__ import annotations

from typing import List, Optional, Set, Dict, Tuple
import hashlib

from usc.api.odc2s_bloom_footer_v0 import read_block_bloom_footer
from usc.mem.block_bloom_index_v0 import query_blocks_for_keywords
from usc.mem.sas_keyword_index_v0 import SASKeywordIndex, query_packets_for_keywords
from usc.api.odc2_sharded_v0 import odc2s_decode_selected_blocks
from usc.mem.usc_recall_v0 import recall_from_packets_decoded, RecallResult


def _packet_index_to_block(packet_index: int, group_size: int) -> int:
    return packet_index // group_size


def _h64(b: bytes) -> int:
    return int.from_bytes(hashlib.blake2b(b, digest_size=8).digest(), "little", signed=False)


def smart_recall_twolevel_from_blob(
    blob: bytes,
    packets_all: List[bytes],
    keywords: Set[str],
    kwi: Optional[SASKeywordIndex],
    require_all: bool = True,
    prefix_len: int = 5,
) -> RecallResult:
    """
    Two-level selective recall (CORRECT + FAST):
      1) Block bloom footer chooses candidate blocks
      2) Packet bloom chooses candidate packets (global)
      3) Keep only packet ids whose block is selected
      4) Decode selected blocks ONCE
      5) Hash-map decoded packets -> real packet indices
      6) Filter decoded packets by those indices
      7) Run recall on filtered packets
    """
    # Need packet blooms
    if kwi is None:
        from usc.mem.usc_recall_v0 import recall_from_odc2s
        return recall_from_odc2s(
            blob=blob,
            packets_all=packets_all,
            keywords=keywords,
            kwi=None,
            group_size=2,
            require_all=require_all,
            prefix_len=prefix_len,
        )

    # Need footer block blooms
    bbi = read_block_bloom_footer(blob)
    if bbi is None:
        from usc.mem.usc_recall_v0 import recall_from_odc2s
        return recall_from_odc2s(
            blob=blob,
            packets_all=packets_all,
            keywords=keywords,
            kwi=kwi,
            group_size=2,
            require_all=require_all,
            prefix_len=prefix_len,
        )

    group_size = bbi.group_size

    # Level 1: choose blocks
    block_ids = query_blocks_for_keywords(
        bbi,
        keywords,
        enable_stem=True,
        prefix_len=prefix_len,
        require_all=require_all,
    )

    # Always include dict block
    dict_block = _packet_index_to_block(0, group_size)
    block_ids.add(dict_block)

    block_ids_sorted = sorted(block_ids)

    # Level 2: choose packets globally
    pkt_ids = query_packets_for_keywords(
        kwi,
        packets_all,
        keywords,
        enable_stem=True,
        prefix_len=prefix_len,
        require_all=require_all,
    )

    # Keep only packets inside chosen blocks
    pkt_ids_in_blocks = {pi for pi in pkt_ids if _packet_index_to_block(pi, group_size) in block_ids}

    # Decode blocks ONCE
    packets_part, meta = odc2s_decode_selected_blocks(blob, block_ids=block_ids_sorted)

    # Build fast hash->index map from original packets
    # (decoded packets should match exact bytes)
    h2i: Dict[int, int] = {}
    for i, pkt in enumerate(packets_all):
        h2i[_h64(pkt)] = i

    # Build filtered packet list:
    # ALWAYS use the canonical dict packet from packets_all[0]
    filtered: List[bytes] = [packets_all[0]]

    # Add only decoded packets whose real index is in pkt_ids_in_blocks
    # Skip dict packet if decoder returned it (avoid duplication)
    for pkt in packets_part:
        hi = _h64(pkt)
        pi = h2i.get(hi, None)
        if pi is None:
            continue
        if pi == 0:
            continue
        if pi in pkt_ids_in_blocks:
            filtered.append(pkt)

    return recall_from_packets_decoded(
        packets_part=filtered,
        keywords=keywords,
        require_all=require_all,
        prefix_len=prefix_len,
        selected_packets=len(pkt_ids_in_blocks),
        selected_blocks=len(block_ids_sorted),
        total_blocks=meta.block_count,
    )
