"""Predictive Pre-Compression — anticipate chunks before they arrive.

Uses co-occurrence data from the CogStore to predict which chunks are
likely to appear together. When chunk A is seen, speculatively loads
predicted chunks B, C as zstd dictionaries so delta compression
becomes nearly free.

This is what makes cogdedup truly "cognitive" — the system doesn't just
remember patterns, it anticipates them.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

try:
    import zstandard as zstd
except ImportError:
    zstd = None

from usc.cogdedup.store import CogStore, ChunkEntry


class PredictiveCompressor:
    """Maintains pre-built zstd dictionaries from predicted co-occurring chunks.

    Usage:
        predictor = PredictiveCompressor(store)

        # When encoding chunk A, check for a pre-built dictionary:
        result = predictor.get_dictionary_and_ids(chunk_a_id)
        if result:
            dict_data, dict_chunk_ids = result
            # Compress with dictionary — much smaller delta
            compressed = zstd.ZstdCompressor(dict_data=dict_data).compress(new_chunk)
            # Store dict_chunk_ids in blob for deterministic decode

        # After encoding a batch, update predictions:
        predictor.update_after_encode(chunk_ids_in_batch)
    """

    def __init__(self, store: CogStore, dict_cache_size: int = 256) -> None:
        self._store = store
        self._dict_cache_size = dict_cache_size
        # chunk_id -> (zstd dict, list of chunk IDs that built it)
        self._dict_cache: Dict[int, Tuple["zstd.ZstdCompressionDict", List[int]]] = {}
        # Track which chunk_ids we've seen recently for cache warming
        self._recent_ids: List[int] = []

    def get_dictionary_for(self, trigger_chunk_id: int) -> Optional["zstd.ZstdCompressionDict"]:
        """Get a pre-built zstd dictionary based on predicted co-occurring chunks.

        Returns None if no predictions are available or zstd is not installed.
        """
        result = self.get_dictionary_and_ids(trigger_chunk_id)
        return result[0] if result is not None else None

    def get_dictionary_and_ids(
        self, trigger_chunk_id: int
    ) -> Optional[Tuple["zstd.ZstdCompressionDict", List[int]]]:
        """Get dictionary AND the chunk IDs that built it.

        Returns (dict_data, chunk_ids) or None. The chunk_ids must be
        embedded in the PRED_DELTA token for deterministic decode.
        """
        if zstd is None:
            return None

        # Check cache first
        if trigger_chunk_id in self._dict_cache:
            return self._dict_cache[trigger_chunk_id]

        # Query co-occurrence predictions
        predicted = self._store.get_predicted_chunks(trigger_chunk_id, top_k=5)
        if not predicted:
            return None

        # Build dictionary from concatenated predicted chunk data
        dict_chunk_ids = [p.chunk_id for p in predicted if p.data]
        dict_content = b"".join(p.data for p in predicted if p.data)
        if len(dict_content) < 64:
            return None

        try:
            dict_data = zstd.ZstdCompressionDict(dict_content)
            # Cache it
            if len(self._dict_cache) >= self._dict_cache_size:
                # Evict oldest entry
                oldest = next(iter(self._dict_cache))
                del self._dict_cache[oldest]
            self._dict_cache[trigger_chunk_id] = (dict_data, dict_chunk_ids)
            return (dict_data, dict_chunk_ids)
        except Exception:
            return None

    def update_after_encode(self, chunk_ids: List[int]) -> None:
        """Update co-occurrence data and warm the prediction cache.

        Call this after encoding a batch of chunks.
        """
        if len(chunk_ids) < 2:
            return

        # Record co-occurrence in store
        self._store.record_cooccurrence(chunk_ids)

        # Pre-warm cache for recently seen chunks
        for cid in chunk_ids[-3:]:  # Warm cache for last 3 chunks
            if cid not in self._dict_cache:
                self.get_dictionary_for(cid)

    def invalidate(self, chunk_id: int) -> None:
        """Remove a chunk's dictionary from cache."""
        self._dict_cache.pop(chunk_id, None)

    @property
    def cache_size(self) -> int:
        return len(self._dict_cache)
