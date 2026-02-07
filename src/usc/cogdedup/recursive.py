"""Recursive Self-Compression — USC compressing C3's own state.

Upgrade #11: Use cogdedup to compress C3's internal data structures:
- Warm tier memories
- Cold tier archives
- Reasoning bank entries
- Audit log entries

This creates a virtuous cycle: as C3 accumulates more memories,
cogdedup gets better at compressing them (shared patterns), which
lets C3 store more memories in the same space, which gives cogdedup
more patterns, etc.

Usage:
    compressor = RecursiveCompressor(store)

    # Compress a batch of C3 memories
    result = compressor.compress_memories(memories)
    print(f"Saved {result.savings_pct}% on {result.items_count} items")

    # Decompress when needed
    memories = compressor.decompress_memories(result.blob)
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from usc.cogdedup.codec import cogdedup_encode, cogdedup_decode
from usc.cogdedup.store import CogStore


@dataclass
class CompressionResult:
    """Result of recursive compression."""
    blob: bytes
    original_size: int
    compressed_size: int
    ratio: float
    savings_pct: float
    items_count: int
    stats: dict


class RecursiveCompressor:
    """Compress C3's own state using cogdedup.

    Handles serialization of C3 data structures into bytes,
    then applies cogdedup encoding. On decompress, reverses the process.

    Args:
        store: CogStore (shared with the main cogdedup pipeline)
        data_id_prefix: Prefix for data_id tracking
    """

    def __init__(
        self,
        store: CogStore,
        data_id_prefix: str = "c3-internal",
    ) -> None:
        self._store = store
        self._prefix = data_id_prefix
        self._total_original: int = 0
        self._total_compressed: int = 0
        self._batches: int = 0

    def compress_memories(
        self,
        memories: List[Dict[str, Any]],
        batch_id: str = "",
    ) -> CompressionResult:
        """Compress a list of memory entries (dicts).

        Each memory is JSON-serialized, concatenated with newline delimiters,
        then encoded with cogdedup.
        """
        # Serialize: one JSON per line (JSONL format)
        lines = [json.dumps(m, separators=(",", ":"), sort_keys=True) for m in memories]
        raw = "\n".join(lines).encode("utf-8")

        data_id = f"{self._prefix}:{batch_id}" if batch_id else ""
        blob, stats = cogdedup_encode(raw, self._store, data_id=data_id)

        original = len(raw)
        compressed = len(blob)
        ratio = original / max(1, compressed)
        savings = ((original - compressed) / max(1, original)) * 100.0

        self._total_original += original
        self._total_compressed += compressed
        self._batches += 1

        return CompressionResult(
            blob=blob,
            original_size=original,
            compressed_size=compressed,
            ratio=round(ratio, 2),
            savings_pct=round(savings, 1),
            items_count=len(memories),
            stats=stats,
        )

    def decompress_memories(self, blob: bytes) -> List[Dict[str, Any]]:
        """Decompress a blob back to list of memory dicts."""
        raw = cogdedup_decode(blob, self._store)
        text = raw.decode("utf-8")
        lines = text.split("\n")
        memories = []
        for line in lines:
            line = line.strip()
            if line:
                memories.append(json.loads(line))
        return memories

    def compress_reasoning_bank(
        self,
        entries: List[str],
        batch_id: str = "",
    ) -> CompressionResult:
        """Compress reasoning bank entries (plain text)."""
        raw = "\n---\n".join(entries).encode("utf-8")

        data_id = f"{self._prefix}:reasoning:{batch_id}" if batch_id else ""
        blob, stats = cogdedup_encode(raw, self._store, data_id=data_id)

        original = len(raw)
        compressed = len(blob)

        self._total_original += original
        self._total_compressed += compressed
        self._batches += 1

        return CompressionResult(
            blob=blob,
            original_size=original,
            compressed_size=compressed,
            ratio=round(original / max(1, compressed), 2),
            savings_pct=round(((original - compressed) / max(1, original)) * 100.0, 1),
            items_count=len(entries),
            stats=stats,
        )

    def decompress_reasoning_bank(self, blob: bytes) -> List[str]:
        """Decompress reasoning bank entries."""
        raw = cogdedup_decode(blob, self._store)
        text = raw.decode("utf-8")
        return text.split("\n---\n")

    def compress_audit_log(
        self,
        events: List[Dict[str, Any]],
        batch_id: str = "",
    ) -> CompressionResult:
        """Compress audit log events."""
        return self.compress_memories(events, batch_id=f"audit:{batch_id}")

    def decompress_audit_log(self, blob: bytes) -> List[Dict[str, Any]]:
        """Decompress audit log events."""
        return self.decompress_memories(blob)

    def stats(self) -> dict:
        return {
            "batches_compressed": self._batches,
            "total_original_bytes": self._total_original,
            "total_compressed_bytes": self._total_compressed,
            "overall_ratio": round(
                self._total_original / max(1, self._total_compressed), 2
            ),
            "overall_savings_pct": round(
                ((self._total_original - self._total_compressed)
                 / max(1, self._total_original)) * 100.0, 1
            ),
        }
