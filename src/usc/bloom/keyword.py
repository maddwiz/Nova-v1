"""
Keyword bloom filter — extracted from existing PFQ1 code for reuse.
Uses FNV-1a hashing (same as tpl_pfq1_query_v1).
"""
from __future__ import annotations

from typing import List


def fnv1a_hash32(s: str, seed: int = 0) -> int:
    """FNV-1a 32-bit hash with seed mixing."""
    h = 2166136261 ^ seed
    for ch in s:
        h ^= ord(ch)
        h = (h * 16777619) & 0xFFFFFFFF
    return h


def bloom_make(n_bits: int) -> bytearray:
    """Allocate a bloom filter with n_bits bits."""
    return bytearray((n_bits + 7) // 8)


def bloom_add(bloom: bytearray, n_bits: int, n_hashes: int, token: str) -> None:
    """Add a token to the bloom filter."""
    t = token.lower()
    for i in range(n_hashes):
        h = fnv1a_hash32(t, 0x9E3779B9 + i * 0x85EBCA6B)
        pos = h % n_bits
        bloom[pos // 8] |= (1 << (pos % 8))


def bloom_check(bloom: bytes, n_bits: int, n_hashes: int, token: str) -> bool:
    """Check if a token might be in the bloom filter."""
    t = token.lower()
    for i in range(n_hashes):
        h = fnv1a_hash32(t, 0x9E3779B9 + i * 0x85EBCA6B)
        pos = h % n_bits
        if not (bloom[pos // 8] & (1 << (pos % 8))):
            return False
    return True


def bloom_check_all(bloom: bytes, n_bits: int, n_hashes: int, tokens: List[str]) -> bool:
    """Check if ALL tokens might be in the bloom filter."""
    return all(bloom_check(bloom, n_bits, n_hashes, t) for t in tokens)
