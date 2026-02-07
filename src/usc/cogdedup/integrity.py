"""Adversarial Robustness — integrity verification for cogdedup.

Upgrade #10: After delta decompression, verify the output matches
expected hash. Prevents bit-flip attacks and corrupted delta chains.
Also adds ref_count thresholds to limit similarity search targets
(prevents adversarial chunks from becoming universal delta bases).

Uses xxHash for fast verification (falls back to CRC32 if unavailable).

Usage:
    # Wrap codec with integrity verification
    verifier = IntegrityVerifier()

    # After delta decompress:
    data = apply_delta(src, delta_bytes)
    if not verifier.verify(data, expected_hash):
        raise CorruptionError("delta decompression produced corrupt output")

    # Configure store with ref_count limits:
    policy = SecurityPolicy(max_ref_count_for_similarity=100)
"""
from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass
from typing import Optional

try:
    import xxhash
    _HAS_XXHASH = True
except ImportError:
    _HAS_XXHASH = False


def fast_hash(data: bytes) -> int:
    """Fast 64-bit hash for integrity verification.

    Uses xxHash if available, otherwise CRC32 (less bits but still catches corruption).
    """
    if _HAS_XXHASH:
        return xxhash.xxh64(data).intdigest()
    # Fallback: CRC32 (32-bit, but still catches accidental corruption)
    return zlib.crc32(data) & 0xFFFFFFFF


def fast_hash_bytes(data: bytes) -> bytes:
    """Return hash as bytes for embedding in wire format."""
    h = fast_hash(data)
    if _HAS_XXHASH:
        return struct.pack("<Q", h)  # 8 bytes
    return struct.pack("<I", h)  # 4 bytes


def verify_hash(data: bytes, expected: bytes) -> bool:
    """Verify data matches expected hash bytes."""
    actual = fast_hash_bytes(data)
    return actual == expected


@dataclass
class SecurityPolicy:
    """Security policy for cogstore operations.

    Attributes:
        max_ref_count_for_similarity: Maximum ref_count for a chunk to be
            used as a delta base. Prevents a single adversarial chunk from
            becoming a universal delta target.
        verify_deltas: Whether to verify hash after delta decompression.
        max_delta_expansion: Maximum allowed expansion ratio for deltas.
            Prevents decompression bombs.
    """
    max_ref_count_for_similarity: int = 1000
    verify_deltas: bool = True
    max_delta_expansion: float = 100.0


class IntegrityVerifier:
    """Verify integrity of decompressed chunks.

    Tracks verification stats for monitoring.
    """

    def __init__(self, policy: Optional[SecurityPolicy] = None) -> None:
        self._policy = policy or SecurityPolicy()
        self._verified: int = 0
        self._failed: int = 0

    def compute_hash(self, data: bytes) -> bytes:
        """Compute hash bytes for embedding in wire format."""
        return fast_hash_bytes(data)

    def verify(self, data: bytes, expected_hash: bytes) -> bool:
        """Verify data integrity after decompression."""
        if verify_hash(data, expected_hash):
            self._verified += 1
            return True
        self._failed += 1
        return False

    def check_delta_expansion(self, src_len: int, result_len: int) -> bool:
        """Check if delta expansion is within safe bounds."""
        if src_len == 0:
            return result_len < 1024 * 1024  # 1MB absolute limit
        ratio = result_len / src_len
        return ratio <= self._policy.max_delta_expansion

    def check_ref_count(self, ref_count: int) -> bool:
        """Check if a chunk's ref_count is within safe bounds for similarity use."""
        return ref_count <= self._policy.max_ref_count_for_similarity

    @property
    def policy(self) -> SecurityPolicy:
        return self._policy

    def stats(self) -> dict:
        return {
            "verified": self._verified,
            "failed": self._failed,
            "failure_rate": (
                self._failed / max(1, self._verified + self._failed)
            ),
        }
