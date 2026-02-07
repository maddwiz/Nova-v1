"""Abstract and in-memory cognitive store for chunk deduplication."""
from __future__ import annotations

import time
import zlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from usc.cogdedup.hasher import sha256_hash, simhash64, hamming_distance, SIMILARITY_THRESHOLD
from usc.cogdedup.lsh import LSHIndex


@dataclass
class ChunkEntry:
    """A stored chunk in the cognitive memory."""
    chunk_id: int
    sha256: str
    simhash: int
    data: bytes
    ref_count: int = 1
    last_access: float = 0.0  # time.time()


class CogStore(ABC):
    """Abstract interface for C3-compatible cognitive chunk store.

    Implementations can be:
    - MemoryCogStore: in-memory dict (for testing / standalone USC)
    - C3CogStore: backed by C3/Ae's Hot tier (SQLite + FAISS)
    """

    @abstractmethod
    def lookup_exact(self, sha256: str) -> Optional[ChunkEntry]:
        """Find a chunk with exact SHA-256 match."""
        ...

    @abstractmethod
    def lookup_similar(self, simhash: int) -> Optional[ChunkEntry]:
        """Find the most similar chunk (hamming distance < threshold)."""
        ...

    @abstractmethod
    def store(self, data: bytes) -> ChunkEntry:
        """Store a new chunk and return its entry."""
        ...

    @abstractmethod
    def get(self, chunk_id: int) -> Optional[ChunkEntry]:
        """Retrieve a chunk by ID."""
        ...

    def record_cooccurrence(self, chunk_ids: List[int]) -> None:
        """Record that these chunks appeared together (for predictive pre-compression)."""
        pass  # Optional — override in implementations that support it

    def get_predicted_chunks(self, chunk_id: int, top_k: int = 5) -> List[ChunkEntry]:
        """Get chunks that frequently co-occur with the given chunk."""
        return []  # Optional — override in implementations that support it

    def get_chunk_ids_for_data(self, data_id: str) -> Set[int]:
        """Get all chunk IDs associated with a data/memory ID."""
        return set()  # Optional — override in implementations that support it


class MemoryCogStore(CogStore):
    """In-memory cognitive store with LSH-accelerated similarity search.

    Features:
    - O(1) exact lookup via hash table
    - O(1) per-band similarity search via LSH index
    - Co-occurrence tracking for predictive pre-compression
    - Ref count tracking for tiered eviction
    - Automatic cold archival (zlib-compress infrequently-used chunks)
    """

    # Cold archival thresholds
    COLD_REF_COUNT = 1         # chunks with ref_count <= this are candidates
    COLD_AGE_SECONDS = 60      # chunks not accessed for this long
    ARCHIVE_TRIGGER = 500      # auto-archive when store exceeds this many chunks
    ARCHIVE_KEEP_WARM = 200    # keep this many most-referenced chunks warm after archive

    def __init__(self) -> None:
        self._by_id: Dict[int, ChunkEntry] = {}
        self._by_sha: Dict[str, ChunkEntry] = {}
        self._lsh = LSHIndex()
        self._next_id = 0

        # Co-occurrence tracking (upgrade #3)
        self._cooccurrence: Dict[int, Dict[int, int]] = {}  # chunk_a -> {chunk_b -> count}

        # Data-to-chunks mapping (upgrade #5)
        self._data_chunks: Dict[str, Set[int]] = {}  # data_id -> set of chunk_ids

        # Cold archive: compressed data for evicted chunks
        self._cold_archive: Dict[int, bytes] = {}  # chunk_id -> zlib-compressed data
        self._cold_meta: Dict[int, Tuple[str, int]] = {}  # chunk_id -> (sha256, simhash)

    @property
    def size(self) -> int:
        return len(self._by_id)

    def lookup_exact(self, sha256: str) -> Optional[ChunkEntry]:
        entry = self._by_sha.get(sha256)
        if entry is not None:
            entry.ref_count += 1
            entry.last_access = time.time()
        return entry

    def lookup_similar(self, simhash: int) -> Optional[ChunkEntry]:
        """LSH-accelerated similarity search — O(1) per band instead of O(n) scan."""
        best_id = self._lsh.query_nearest(simhash)
        if best_id is not None:
            entry = self._by_id[best_id]
            entry.last_access = time.time()
            return entry
        return None

    def store(self, data: bytes) -> ChunkEntry:
        sha = sha256_hash(data)

        # Check if already stored (warm tier)
        existing = self._by_sha.get(sha)
        if existing is not None:
            # Promote from cold if needed
            if existing.data is None and existing.chunk_id in self._cold_archive:
                existing.data = zlib.decompress(self._cold_archive.pop(existing.chunk_id))
            existing.ref_count += 1
            existing.last_access = time.time()
            return existing

        cid = self._next_id
        self._next_id += 1
        sh = simhash64(data)
        now = time.time()

        entry = ChunkEntry(chunk_id=cid, sha256=sha, simhash=sh, data=data,
                           ref_count=1, last_access=now)
        self._by_id[cid] = entry
        self._by_sha[sha] = entry
        self._lsh.insert(cid, sh)

        # Auto-archive if store is getting large
        if len(self._by_id) > self.ARCHIVE_TRIGGER and len(self._by_id) % 100 == 0:
            self._archive_cold()

        return entry

    def get(self, chunk_id: int) -> Optional[ChunkEntry]:
        entry = self._by_id.get(chunk_id)
        if entry is not None:
            # Decompress from cold archive on demand
            if entry.data is None and chunk_id in self._cold_archive:
                entry.data = zlib.decompress(self._cold_archive[chunk_id])
            return entry
        return None

    def _archive_cold(self) -> int:
        """Move infrequently-used chunks to cold storage (zlib-compressed).

        Returns number of chunks archived.
        """
        now = time.time()
        candidates = []

        for cid, entry in self._by_id.items():
            if entry.data is None:
                continue  # Already cold
            if entry.ref_count <= self.COLD_REF_COUNT and (now - entry.last_access) > self.COLD_AGE_SECONDS:
                candidates.append((entry.ref_count, entry.last_access, cid))

        if not candidates:
            return 0

        # Sort: least referenced, oldest first
        candidates.sort()

        # Keep at least ARCHIVE_KEEP_WARM chunks with data
        warm_count = sum(1 for e in self._by_id.values() if e.data is not None)
        max_to_archive = max(0, warm_count - self.ARCHIVE_KEEP_WARM)
        to_archive = candidates[:max_to_archive]

        archived = 0
        for _, _, cid in to_archive:
            entry = self._by_id[cid]
            if entry.data is not None:
                self._cold_archive[cid] = zlib.compress(entry.data, 6)
                self._cold_meta[cid] = (entry.sha256, entry.simhash)
                entry.data = None  # Free memory
                archived += 1

        return archived

    def record_cooccurrence(self, chunk_ids: List[int]) -> None:
        """Track which chunks appear together for predictive pre-compression."""
        for i, a in enumerate(chunk_ids):
            if a not in self._cooccurrence:
                self._cooccurrence[a] = {}
            for j, b in enumerate(chunk_ids):
                if i != j:
                    self._cooccurrence[a][b] = self._cooccurrence[a].get(b, 0) + 1

    def get_predicted_chunks(self, chunk_id: int, top_k: int = 5) -> List[ChunkEntry]:
        """Get chunks that frequently co-occur with the given chunk."""
        neighbors = self._cooccurrence.get(chunk_id, {})
        if not neighbors:
            return []
        sorted_ids = sorted(neighbors, key=neighbors.get, reverse=True)[:top_k]
        return [self._by_id[cid] for cid in sorted_ids if cid in self._by_id]

    def register_data_chunks(self, data_id: str, chunk_ids: Set[int]) -> None:
        """Register chunk IDs associated with a data/memory entry."""
        self._data_chunks[data_id] = chunk_ids

    def get_chunk_ids_for_data(self, data_id: str) -> Set[int]:
        return self._data_chunks.get(data_id, set())

    def stats(self) -> dict:
        """Get store statistics."""
        total_refs = sum(e.ref_count for e in self._by_id.values())
        warm_entries = [e for e in self._by_id.values() if e.data is not None]
        warm_bytes = sum(len(e.data) for e in warm_entries)
        cold_bytes_compressed = sum(len(v) for v in self._cold_archive.values())
        cold_count = len(self._cold_archive)
        return {
            "unique_chunks": len(self._by_id),
            "warm_chunks": len(warm_entries),
            "warm_bytes": warm_bytes,
            "cold_chunks": cold_count,
            "cold_bytes_compressed": cold_bytes_compressed,
            "total_memory": warm_bytes + cold_bytes_compressed,
            "total_references": total_refs,
            "dedup_ratio": round(total_refs / max(1, len(self._by_id)), 2),
            "lsh_index_size": self._lsh.size,
            "cooccurrence_pairs": sum(len(v) for v in self._cooccurrence.values()),
        }
