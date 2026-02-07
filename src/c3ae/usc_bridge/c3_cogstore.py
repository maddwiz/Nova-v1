"""C3-backed Cognitive Store with LSH index, tiered storage,
co-occurrence tracking, and compression-aware retrieval.

Uses C3's SQLite database to persist chunk hashes and data,
enabling cognitive deduplication across agent sessions.
The more sessions run, the better compression becomes —
common patterns (tool calls, error handling, search results)
approach zero encoding cost.

Upgrades over v1:
- LSH band index for O(1) similarity lookup (was O(n) scan)
- Tiered chunks: Hot (in-memory), Warm (LSH indexed), Cold (compressed archive)
- Co-occurrence tracking for predictive pre-compression
- Memory-to-chunks mapping for compression-aware retrieval
"""
from __future__ import annotations

import sqlite3
import sys
import time
import zlib
from pathlib import Path
from typing import Dict, List, Optional, Set

# USC cogdedup is a sibling package in the Nova-v1 monorepo

from usc.cogdedup.hasher import sha256_hash, simhash64, hamming_distance, SIMILARITY_THRESHOLD
from usc.cogdedup.lsh import LSHIndex, N_BANDS, _extract_bands
from usc.cogdedup.store import CogStore, ChunkEntry


def _to_signed64(n: int) -> int:
    """Convert unsigned 64-bit to signed for SQLite storage."""
    if n >= (1 << 63):
        return n - (1 << 64)
    return n


def _to_unsigned64(n: int) -> int:
    """Convert signed 64-bit back to unsigned."""
    if n < 0:
        return n + (1 << 64)
    return n


# --- Tier thresholds ---
HOT_MIN_REF_COUNT = 5       # Chunks referenced >= 5 times are hot
HOT_MAX_CHUNKS = 10000      # Cap hot tier to 10k chunks
COLD_AGE_DAYS = 30          # Chunks untouched for 30 days can be archived
COLD_MIN_AGE_SECONDS = COLD_AGE_DAYS * 86400


_C3_COGDEDUP_SCHEMA = """
CREATE TABLE IF NOT EXISTS cogdedup_chunks (
    chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
    sha256 TEXT NOT NULL UNIQUE,
    simhash INTEGER NOT NULL,
    data BLOB NOT NULL,
    size_bytes INTEGER NOT NULL,
    ref_count INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    last_access REAL NOT NULL DEFAULT 0,
    tier TEXT NOT NULL DEFAULT 'warm'
);
CREATE INDEX IF NOT EXISTS idx_cogdedup_sha ON cogdedup_chunks(sha256);
CREATE INDEX IF NOT EXISTS idx_cogdedup_simhash ON cogdedup_chunks(simhash);
CREATE INDEX IF NOT EXISTS idx_cogdedup_tier ON cogdedup_chunks(tier);

-- LSH band index: O(1) similarity lookup
CREATE TABLE IF NOT EXISTS cogdedup_lsh_bands (
    band_id INTEGER NOT NULL,
    band_value INTEGER NOT NULL,
    chunk_id INTEGER NOT NULL REFERENCES cogdedup_chunks(chunk_id),
    PRIMARY KEY (band_id, band_value, chunk_id)
);
CREATE INDEX IF NOT EXISTS idx_lsh_lookup ON cogdedup_lsh_bands(band_id, band_value);

-- Co-occurrence tracking for predictive pre-compression
CREATE TABLE IF NOT EXISTS cogdedup_cooccurrence (
    chunk_a INTEGER NOT NULL,
    chunk_b INTEGER NOT NULL,
    count INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY (chunk_a, chunk_b)
);
CREATE INDEX IF NOT EXISTS idx_cooccur_a ON cogdedup_cooccurrence(chunk_a);

-- Memory-to-chunks mapping for compression-aware retrieval
CREATE TABLE IF NOT EXISTS cogdedup_memory_chunks (
    memory_id TEXT NOT NULL,
    chunk_id INTEGER NOT NULL,
    PRIMARY KEY (memory_id, chunk_id)
);
CREATE INDEX IF NOT EXISTS idx_memchunk_mem ON cogdedup_memory_chunks(memory_id);
CREATE INDEX IF NOT EXISTS idx_memchunk_chunk ON cogdedup_memory_chunks(chunk_id);

-- Cold archive: compressed chunk data for rarely-accessed chunks
CREATE TABLE IF NOT EXISTS cogdedup_cold_archive (
    chunk_id INTEGER PRIMARY KEY REFERENCES cogdedup_chunks(chunk_id),
    compressed_data BLOB NOT NULL,
    original_size INTEGER NOT NULL
);
"""


class C3CogStore(CogStore):
    """CogStore backed by C3's SQLite database with LSH index and tiered storage.

    Tiers:
    - Hot: high ref_count chunks cached in-memory for zero-latency exact match
    - Warm: LSH-indexed chunks for fast similarity search
    - Cold: compressed archive for rarely-accessed chunks

    Persists chunk data across sessions so future encoding can
    reference previously seen chunks (REF) or use similar chunks
    as compression dictionaries (DELTA).
    """

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = str(db_path)
        self._conn = sqlite3.connect(self._db_path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_C3_COGDEDUP_SCHEMA)
        self._conn.commit()

        # Hot tier: in-memory cache for frequent chunks
        self._hot_exact: Dict[str, ChunkEntry] = {}  # sha256 -> entry
        self._hot_by_id: Dict[int, ChunkEntry] = {}   # chunk_id -> entry

        # In-memory LSH index (rebuilt from SQLite on startup)
        self._lsh = LSHIndex()
        self._rebuild_lsh_index()
        self._load_hot_tier()

    def _rebuild_lsh_index(self) -> None:
        """Rebuild in-memory LSH index from SQLite on startup."""
        rows = self._conn.execute(
            "SELECT chunk_id, simhash FROM cogdedup_chunks WHERE tier != 'cold'"
        ).fetchall()
        entries = [(r[0], _to_unsigned64(r[1])) for r in rows]
        self._lsh.rebuild(entries)

    def _load_hot_tier(self) -> None:
        """Load high-frequency chunks into memory."""
        rows = self._conn.execute(
            "SELECT chunk_id, sha256, simhash, data, ref_count, last_access "
            "FROM cogdedup_chunks WHERE ref_count >= ? "
            "ORDER BY ref_count DESC LIMIT ?",
            (HOT_MIN_REF_COUNT, HOT_MAX_CHUNKS),
        ).fetchall()
        for row in rows:
            entry = ChunkEntry(
                chunk_id=row[0], sha256=row[1],
                simhash=_to_unsigned64(row[2]), data=row[3],
                ref_count=row[4], last_access=row[5],
            )
            self._hot_exact[entry.sha256] = entry
            self._hot_by_id[entry.chunk_id] = entry
            # Update tier in DB
            self._conn.execute(
                "UPDATE cogdedup_chunks SET tier = 'hot' WHERE chunk_id = ?",
                (entry.chunk_id,),
            )
        self._conn.commit()

    def lookup_exact(self, sha256: str) -> Optional[ChunkEntry]:
        # Hot tier first (zero-latency)
        hot = self._hot_exact.get(sha256)
        if hot is not None:
            hot.ref_count += 1
            hot.last_access = time.time()
            return hot

        # Warm tier (SQLite indexed lookup)
        row = self._conn.execute(
            "SELECT chunk_id, sha256, simhash, data, ref_count FROM cogdedup_chunks WHERE sha256 = ?",
            (sha256,),
        ).fetchone()
        if row is None:
            return None
        entry = ChunkEntry(chunk_id=row[0], sha256=row[1],
                           simhash=_to_unsigned64(row[2]), data=row[3],
                           ref_count=row[4])

        # Check if data is in cold archive (compressed)
        if entry.data is None or len(entry.data) == 0:
            entry = self._decompress_cold(entry)

        # Promote to hot if ref_count is high enough
        self._maybe_promote_hot(entry)
        return entry

    def lookup_similar(self, simhash: int) -> Optional[ChunkEntry]:
        """LSH-accelerated similarity search.

        Uses in-memory LSH index for O(1) per-band candidate retrieval,
        then verifies candidates with exact hamming distance.
        """
        best_id = self._lsh.query_nearest(simhash)
        if best_id is not None:
            return self.get(best_id)

        # Fallback: also check SQLite LSH bands table for any chunks
        # not in memory (e.g., recently added by another process)
        bands = _extract_bands(simhash)
        candidate_ids: set = set()
        for band_id, band_val in enumerate(bands):
            rows = self._conn.execute(
                "SELECT chunk_id FROM cogdedup_lsh_bands WHERE band_id = ? AND band_value = ?",
                (band_id, band_val),
            ).fetchall()
            candidate_ids.update(r[0] for r in rows)

        best_dist = SIMILARITY_THRESHOLD + 1
        best_entry: Optional[ChunkEntry] = None
        for cid in candidate_ids:
            entry = self.get(cid)
            if entry is None:
                continue
            d = hamming_distance(simhash, entry.simhash)
            if d < best_dist:
                best_dist = d
                best_entry = entry

        return best_entry

    def store(self, data: bytes) -> ChunkEntry:
        sha = sha256_hash(data)

        # Check hot cache first
        existing = self._hot_exact.get(sha)
        if existing is not None:
            existing.ref_count += 1
            existing.last_access = time.time()
            self._conn.execute(
                "UPDATE cogdedup_chunks SET ref_count = ref_count + 1, last_access = ? WHERE sha256 = ?",
                (time.time(), sha),
            )
            self._conn.commit()
            return existing

        # Check SQLite
        row = self._conn.execute(
            "SELECT chunk_id, sha256, simhash, data, ref_count FROM cogdedup_chunks WHERE sha256 = ?",
            (sha,),
        ).fetchone()
        if row is not None:
            self._conn.execute(
                "UPDATE cogdedup_chunks SET ref_count = ref_count + 1, last_access = ? WHERE sha256 = ?",
                (time.time(), sha),
            )
            self._conn.commit()
            entry = ChunkEntry(chunk_id=row[0], sha256=row[1],
                               simhash=_to_unsigned64(row[2]), data=row[3],
                               ref_count=row[4] + 1)
            self._maybe_promote_hot(entry)
            return entry

        # New chunk — insert
        sh = simhash64(data)
        now = time.time()
        cursor = self._conn.execute(
            "INSERT INTO cogdedup_chunks (sha256, simhash, data, size_bytes, last_access, tier) "
            "VALUES (?, ?, ?, ?, ?, 'warm')",
            (sha, _to_signed64(sh), data, len(data), now),
        )
        cid = cursor.lastrowid

        # Insert LSH band entries
        bands = _extract_bands(sh)
        for band_id, band_val in enumerate(bands):
            self._conn.execute(
                "INSERT OR IGNORE INTO cogdedup_lsh_bands (band_id, band_value, chunk_id) VALUES (?, ?, ?)",
                (band_id, band_val, cid),
            )
        self._conn.commit()

        # Update in-memory LSH index
        self._lsh.insert(cid, sh)

        return ChunkEntry(chunk_id=cid, sha256=sha, simhash=sh, data=data,
                          ref_count=1, last_access=now)

    def get(self, chunk_id: int) -> Optional[ChunkEntry]:
        # Hot cache first
        hot = self._hot_by_id.get(chunk_id)
        if hot is not None:
            return hot

        row = self._conn.execute(
            "SELECT chunk_id, sha256, simhash, data, ref_count, last_access "
            "FROM cogdedup_chunks WHERE chunk_id = ?",
            (chunk_id,),
        ).fetchone()
        if row is None:
            return None
        entry = ChunkEntry(chunk_id=row[0], sha256=row[1],
                           simhash=_to_unsigned64(row[2]), data=row[3],
                           ref_count=row[4], last_access=row[5])

        # If cold, decompress
        if entry.data is None or len(entry.data) == 0:
            entry = self._decompress_cold(entry)

        return entry

    # --- Tiered storage ---

    def _maybe_promote_hot(self, entry: ChunkEntry) -> None:
        """Promote a warm chunk to hot tier if it's frequently accessed."""
        if entry.ref_count >= HOT_MIN_REF_COUNT and entry.chunk_id not in self._hot_by_id:
            if len(self._hot_by_id) < HOT_MAX_CHUNKS:
                self._hot_exact[entry.sha256] = entry
                self._hot_by_id[entry.chunk_id] = entry
                self._conn.execute(
                    "UPDATE cogdedup_chunks SET tier = 'hot' WHERE chunk_id = ?",
                    (entry.chunk_id,),
                )

    def archive_cold_chunks(self) -> int:
        """Move old, rarely-accessed chunks to cold storage (compressed).

        Returns number of chunks archived.
        """
        cutoff = time.time() - COLD_MIN_AGE_SECONDS
        rows = self._conn.execute(
            "SELECT chunk_id, data, size_bytes FROM cogdedup_chunks "
            "WHERE tier = 'warm' AND ref_count < ? AND last_access < ? AND last_access > 0",
            (HOT_MIN_REF_COUNT, cutoff),
        ).fetchall()

        archived = 0
        for row in rows:
            cid, data, orig_size = row
            if data is None:
                continue
            compressed = zlib.compress(data, 9)
            self._conn.execute(
                "INSERT OR REPLACE INTO cogdedup_cold_archive (chunk_id, compressed_data, original_size) "
                "VALUES (?, ?, ?)",
                (cid, compressed, orig_size),
            )
            # Clear data blob from main table to save space, keep metadata
            self._conn.execute(
                "UPDATE cogdedup_chunks SET data = X'', tier = 'cold' WHERE chunk_id = ?",
                (cid,),
            )
            # Remove from LSH index (cold chunks aren't similarity-searched)
            self._lsh.remove(cid)
            archived += 1

        if archived:
            self._conn.commit()
        return archived

    def _decompress_cold(self, entry: ChunkEntry) -> ChunkEntry:
        """Decompress a cold chunk's data from archive."""
        row = self._conn.execute(
            "SELECT compressed_data, original_size FROM cogdedup_cold_archive WHERE chunk_id = ?",
            (entry.chunk_id,),
        ).fetchone()
        if row is not None:
            entry.data = zlib.decompress(row[0])
        return entry

    # --- Co-occurrence tracking (Upgrade #3: Predictive Pre-Compression) ---

    def record_cooccurrence(self, chunk_ids: List[int]) -> None:
        """Record which chunks appeared together in an encode operation."""
        for i, a in enumerate(chunk_ids):
            for j, b in enumerate(chunk_ids):
                if i != j:
                    self._conn.execute(
                        "INSERT INTO cogdedup_cooccurrence (chunk_a, chunk_b, count) "
                        "VALUES (?, ?, 1) "
                        "ON CONFLICT(chunk_a, chunk_b) DO UPDATE SET count = count + 1",
                        (a, b),
                    )
        self._conn.commit()

    def get_predicted_chunks(self, chunk_id: int, top_k: int = 5) -> List[ChunkEntry]:
        """Get chunks that frequently co-occur with the given chunk.

        Used for predictive pre-compression: when we see chunk A,
        we pre-load predicted chunks B, C as zstd dictionaries.
        """
        rows = self._conn.execute(
            "SELECT chunk_b, count FROM cogdedup_cooccurrence "
            "WHERE chunk_a = ? ORDER BY count DESC LIMIT ?",
            (chunk_id, top_k),
        ).fetchall()
        results = []
        for r in rows:
            entry = self.get(r[0])
            if entry is not None:
                results.append(entry)
        return results

    # --- Memory-to-chunks mapping (Upgrade #5: Compression-Aware Retrieval) ---

    def register_data_chunks(self, data_id: str, chunk_ids: Set[int]) -> None:
        """Register chunk IDs associated with a data/memory entry."""
        for cid in chunk_ids:
            self._conn.execute(
                "INSERT OR IGNORE INTO cogdedup_memory_chunks (memory_id, chunk_id) VALUES (?, ?)",
                (data_id, cid),
            )
        self._conn.commit()

    def get_chunk_ids_for_data(self, data_id: str) -> Set[int]:
        """Get all chunk IDs associated with a data/memory ID."""
        rows = self._conn.execute(
            "SELECT chunk_id FROM cogdedup_memory_chunks WHERE memory_id = ?",
            (data_id,),
        ).fetchall()
        return {r[0] for r in rows}

    def structural_similarity(self, data_id_a: str, data_id_b: str) -> float:
        """Compute Jaccard similarity over shared chunk IDs.

        Returns a value between 0.0 (no overlap) and 1.0 (identical chunks).
        This is a free structural similarity signal from the compression layer.
        """
        chunks_a = self.get_chunk_ids_for_data(data_id_a)
        chunks_b = self.get_chunk_ids_for_data(data_id_b)
        if not chunks_a and not chunks_b:
            return 0.0
        intersection = chunks_a & chunks_b
        union = chunks_a | chunks_b
        return len(intersection) / len(union) if union else 0.0

    def find_structurally_similar(self, data_id: str, threshold: float = 0.3) -> List[tuple]:
        """Find memories structurally similar to the given one.

        Returns list of (memory_id, jaccard_score) sorted by similarity.
        """
        target_chunks = self.get_chunk_ids_for_data(data_id)
        if not target_chunks:
            return []

        # Find all memories that share at least one chunk
        placeholders = ",".join("?" * len(target_chunks))
        rows = self._conn.execute(
            f"SELECT DISTINCT memory_id FROM cogdedup_memory_chunks "
            f"WHERE chunk_id IN ({placeholders}) AND memory_id != ?",
            list(target_chunks) + [data_id],
        ).fetchall()

        results = []
        for (mem_id,) in rows:
            score = self.structural_similarity(data_id, mem_id)
            if score >= threshold:
                results.append((mem_id, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results

    # --- Stats ---

    @property
    def size(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM cogdedup_chunks").fetchone()
        return row[0] if row else 0

    @property
    def total_bytes_stored(self) -> int:
        row = self._conn.execute("SELECT COALESCE(SUM(size_bytes), 0) FROM cogdedup_chunks").fetchone()
        return row[0] if row else 0

    def stats(self) -> dict:
        """Get dedup store statistics with tier breakdown."""
        row = self._conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(size_bytes), 0), "
            "COALESCE(SUM(ref_count), 0) FROM cogdedup_chunks"
        ).fetchone()

        tier_counts = {}
        for (tier, cnt) in self._conn.execute(
            "SELECT tier, COUNT(*) FROM cogdedup_chunks GROUP BY tier"
        ).fetchall():
            tier_counts[tier] = cnt

        cooccur_row = self._conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(count), 0) FROM cogdedup_cooccurrence"
        ).fetchone()

        return {
            "unique_chunks": row[0],
            "total_bytes": row[1],
            "total_references": row[2],
            "dedup_ratio": round(row[2] / max(1, row[0]), 2),
            "tiers": {
                "hot": tier_counts.get("hot", 0),
                "warm": tier_counts.get("warm", 0),
                "cold": tier_counts.get("cold", 0),
            },
            "hot_cache_size": len(self._hot_by_id),
            "lsh_index_size": self._lsh.size,
            "cooccurrence_pairs": cooccur_row[0] if cooccur_row else 0,
            "cooccurrence_total": cooccur_row[1] if cooccur_row else 0,
        }

    def close(self) -> None:
        self._conn.close()
