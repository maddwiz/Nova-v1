"""Locality-Sensitive Hashing index for fast SimHash similarity lookup.

Partitions 64-bit SimHash into N_BANDS bands of BAND_WIDTH bits.
Two hashes sharing ANY band value are candidate matches.
This reduces similarity search from O(n) to O(1) per band.

The band parameters control the trade-off:
  - More bands = higher recall (catches more similar pairs)
  - Wider bands = higher precision (fewer false positives per band)

With 8 bands of 8 bits:
  - Two hashes differing by <=8 bits have ~97% chance of sharing at least one band
  - This matches SIMILARITY_THRESHOLD = 8
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Set

from usc.cogdedup.hasher import hamming_distance, SIMILARITY_THRESHOLD

N_BANDS = 8
BAND_WIDTH = 8  # bits per band


def _extract_bands(simhash: int) -> List[int]:
    """Extract N_BANDS band values from a 64-bit SimHash."""
    bands = []
    for i in range(N_BANDS):
        band_val = (simhash >> (i * BAND_WIDTH)) & ((1 << BAND_WIDTH) - 1)
        bands.append(band_val)
    return bands


class LSHIndex:
    """In-memory LSH index for fast approximate similarity search.

    Supports O(1) candidate retrieval per band, then filters by
    exact hamming distance.
    """

    def __init__(self) -> None:
        # band_id -> band_value -> set of chunk_ids
        self._buckets: List[Dict[int, Set[int]]] = [
            defaultdict(set) for _ in range(N_BANDS)
        ]
        # chunk_id -> simhash (for hamming distance verification)
        self._simhashes: Dict[int, int] = {}

    def insert(self, chunk_id: int, simhash: int) -> None:
        """Add a chunk to the LSH index."""
        self._simhashes[chunk_id] = simhash
        bands = _extract_bands(simhash)
        for band_id, band_val in enumerate(bands):
            self._buckets[band_id][band_val].add(chunk_id)

    def remove(self, chunk_id: int) -> None:
        """Remove a chunk from the LSH index."""
        sh = self._simhashes.pop(chunk_id, None)
        if sh is None:
            return
        bands = _extract_bands(sh)
        for band_id, band_val in enumerate(bands):
            self._buckets[band_id][band_val].discard(chunk_id)

    def query_candidates(self, simhash: int) -> Set[int]:
        """Get candidate chunk IDs that share at least one band value.

        These are potential similar matches — verify with hamming_distance().
        """
        candidates: Set[int] = set()
        bands = _extract_bands(simhash)
        for band_id, band_val in enumerate(bands):
            candidates.update(self._buckets[band_id][band_val])
        return candidates

    def query_nearest(self, simhash: int, threshold: int = SIMILARITY_THRESHOLD) -> Optional[int]:
        """Find the nearest chunk_id within hamming distance threshold.

        Returns chunk_id or None.
        """
        candidates = self.query_candidates(simhash)
        best_dist = threshold + 1
        best_id: Optional[int] = None

        for cid in candidates:
            d = hamming_distance(simhash, self._simhashes[cid])
            if d < best_dist:
                best_dist = d
                best_id = cid

        return best_id

    @property
    def size(self) -> int:
        return len(self._simhashes)

    def rebuild(self, entries: List[tuple]) -> None:
        """Bulk rebuild from list of (chunk_id, simhash) tuples."""
        self._buckets = [defaultdict(set) for _ in range(N_BANDS)]
        self._simhashes.clear()
        for chunk_id, simhash in entries:
            self.insert(chunk_id, simhash)
