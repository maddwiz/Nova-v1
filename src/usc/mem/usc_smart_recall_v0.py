from __future__ import annotations

from typing import List, Optional, Set

from usc.api.odc2s_bloom_footer_v0 import read_block_bloom_footer
from usc.mem.block_bloom_index_v0 import query_blocks_for_keywords
from usc.api.odc2_sharded_v0 import odc2s_decode_selected_blocks
from usc.mem.usc_recall_v0 import recall_from_odc2s, recall_from_packets_decoded, RecallResult
from usc.mem.sas_keyword_index_v0 import SASKeywordIndex


def smart_recall_from_blob(
    blob: bytes,
    packets_all: List[bytes],
    keywords: Set[str],
    kwi: Optional[SASKeywordIndex],
    require_all: bool = True,
    prefix_len: int = 5,
) -> RecallResult:
    """
    Smart recall path:
      - read block bloom footer (portable blob index)
      - prefilter blocks
      - decode selected blocks ONCE
      - filter hits directly from decoded packets (no double decode)
    """
    bbi = read_block_bloom_footer(blob)
    if bbi is None:
        return recall_from_odc2s(
            blob=blob,
            packets_all=packets_all,
            keywords=keywords,
            kwi=kwi,
            group_size=2,
            require_all=require_all,
            prefix_len=prefix_len,
        )

    block_ids = query_blocks_for_keywords(
        bbi,
        keywords,
        enable_stem=True,
        prefix_len=prefix_len,
        require_all=require_all,
    )

    packets_part, meta = odc2s_decode_selected_blocks(blob, block_ids=block_ids)

    # Ensure dict packet is present first
    if not packets_part:
        packets_part = [packets_all[0]]
    else:
        if not packets_part[0].startswith(b"SASD"):
            packets_part = [packets_all[0]] + packets_part

    # selected_packets is unknown here (we selected blocks directly), but we can estimate:
    selected_packets = max(0, len(packets_part) - 1)

    return recall_from_packets_decoded(
        packets_part=packets_part,
        keywords=keywords,
        require_all=require_all,
        prefix_len=prefix_len,
        selected_packets=selected_packets,
        selected_blocks=len(block_ids),
        total_blocks=meta.block_count,
    )
