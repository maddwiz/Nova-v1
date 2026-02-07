from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

import hashlib

from usc.api.odc2_sharded_v0 import packet_indices_to_block_ids
from usc.mem.sas_keyword_index_v0 import _variants_for_keyword, _stem_lite


def _hash64(s: str) -> int:
    h = hashlib.blake2b(s.encode("utf-8", errors="replace"), digest_size=8).digest()
    return int.from_bytes(h, "little", signed=False)


def _k_hashes(h64: int, k: int, m_bits: int) -> List[int]:
    out: List[int] = []
    x = h64 & ((1 << 64) - 1)
    for i in range(k):
        x = (x * 0x9E3779B97F4A7C15 + (i + 1) * 0xD1B54A32D192ED03) & ((1 << 64) - 1)
        out.append(int(x % m_bits))
    return out


def _get_bit(bits: bytes, pos: int) -> bool:
    return (bits[pos >> 3] >> (pos & 7)) & 1 == 1


def _or_into(dst: bytearray, src: bytes) -> None:
    # dst and src must have same length
    for i in range(len(dst)):
        dst[i] |= src[i]


@dataclass
class BlockBloomIndex:
    """
    Block-level Bloom filter index.
    Each block bloom = OR of all packet blooms inside that block.
    """
    m_bits: int
    k_hashes: int
    group_size: int
    block_blooms: Dict[int, bytes]


def build_block_bloom_index(
    packet_blooms: List[bytes],
    m_bits: int,
    k_hashes: int,
    group_size: int,
) -> BlockBloomIndex:
    """
    packet_blooms corresponds to packets[1:] blooms from SASKeywordIndex,
    i.e. packet id (pi) is (index+1).
    """
    bloom_bytes = m_bits // 8
    out: Dict[int, bytearray] = {}

    # packet_id starts at 1 (dict packet is packet 0 / not included here)
    for j, pb in enumerate(packet_blooms):
        packet_id = j + 1  # this matches kwi.packet_blooms indexing logic
        # Convert to actual packet index used by packet_indices_to_block_ids:
        # In your code, packet indices include dict packet at index 0, so data packets are 1..N
        actual_packet_index = packet_id  # already aligned to "packets_all" indexing (skip dict bloom)
        block_ids = packet_indices_to_block_ids({actual_packet_index + 0}, group_size)

        for bid in block_ids:
            if bid not in out:
                out[bid] = bytearray(bloom_bytes)
            _or_into(out[bid], pb)

    return BlockBloomIndex(
        m_bits=int(m_bits),
        k_hashes=int(k_hashes),
        group_size=int(group_size),
        block_blooms={k: bytes(v) for k, v in out.items()},
    )


def query_blocks_for_keywords(
    bbi: BlockBloomIndex,
    keywords: Set[str],
    enable_stem: bool = True,
    prefix_len: int = 5,
    require_all: bool = True,
) -> Set[int]:
    """
    Query block blooms directly for keywords.
    Same logic as packet bloom query, but over blocks.
    """
    kws = {k.strip().lower() for k in keywords if k.strip()}
    if not kws:
        return set()

    kw_groups: List[Set[str]] = []
    for k in kws:
        if any(k.startswith(pfx) for pfx in ("k:", "v:", "kv:", "n:", "tool:")):
            kw_groups.append({k})
        else:
            # plain keywords get stem/prefix variants
            kw_groups.append(_variants_for_keyword(k, enable_stem=enable_stem, prefix_len=prefix_len))

    # Precompute hash positions per variant
    variant_positions: Dict[str, List[int]] = {}
    for group in kw_groups:
        for v in group:
            if v not in variant_positions:
                variant_positions[v] = _k_hashes(_hash64(v), bbi.k_hashes, bbi.m_bits)

    matched: Set[int] = set()

    for bid, bits in bbi.block_blooms.items():
        if require_all:
            ok = True
            for group in kw_groups:
                g_ok = False
                for v in group:
                    pos_list = variant_positions[v]
                    if all(_get_bit(bits, pos) for pos in pos_list):
                        g_ok = True
                        break
                if not g_ok:
                    ok = False
                    break
            if ok:
                matched.add(bid)
        else:
            hit = False
            for group in kw_groups:
                for v in group:
                    pos_list = variant_positions[v]
                    if all(_get_bit(bits, pos) for pos in pos_list):
                        hit = True
                        break
                if hit:
                    break
            if hit:
                matched.add(bid)

    return matched
