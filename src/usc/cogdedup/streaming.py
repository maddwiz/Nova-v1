"""Streaming Cognitive Deduplication — compress in real-time as data arrives.

For live agent sessions, compresses as the session runs rather than
after it ends. The architecture:

1. Rolling Rabin hash detects chunk boundaries on the fly
2. On each boundary, immediately do cogstore lookup
3. Emit REF/DELTA/FULL tokens to a write-ahead log
4. At session end, the log IS the compressed session — no second pass

This means memory cost is amortized across the session rather than
spiking at the end. For long-running agents, this is the difference
between "works in a demo" and "works in production."
"""
from __future__ import annotations

from typing import List, Optional, Tuple

try:
    import zstandard as zstd
except ImportError:
    zstd = None

from usc.cogdedup.hasher import (
    sha256_hash,
    simhash64,
    _RABIN_POLY,
    _MIN_CHUNK,
    _AVG_CHUNK,
    _MAX_CHUNK,
    _MASK,
)
from usc.cogdedup.store import CogStore, ChunkEntry
from usc.cogdedup.predictor import PredictiveCompressor
from usc.cogdedup.codec import (
    MAGIC, VERSION, REF, DELTA, FULL, PRED_DELTA,
    _zstd_compress, _compute_delta, _compress_with_dict,
)
from usc.mem.varint import encode_uvarint


class CogdedupStream:
    """Streaming cognitive dedup encoder.

    Usage:
        stream = CogdedupStream(store)

        # Feed data as it arrives (from agent session)
        stream.feed(b"[TOOL_CALL] web_search query='python'\\n")
        stream.feed(b"[TOOL_RESULT] Found 5 results\\n")
        stream.feed(b"[THINKING] Analyzing...\\n")
        # ... more data over time ...

        # When session ends, finalize
        blob, stats = stream.finish()
        # blob is a valid UCOG blob
    """

    def __init__(
        self,
        store: CogStore,
        *,
        zstd_level: int = 10,
        data_id: str = "",
        predictor: Optional[PredictiveCompressor] = None,
    ) -> None:
        self._store = store
        self._zstd_level = zstd_level
        self._data_id = data_id
        self._predictor = predictor

        # Rolling Rabin hash state
        self._fp: int = 0
        self._buf = bytearray()
        self._chunk_start: int = 0

        # Write-ahead log: list of (chunk_type, encoded_bytes) tokens
        self._tokens: List[bytes] = []
        self._n_chunks: int = 0
        self._chunk_ids: List[int] = []

        # Stats
        self._stats = {"ref": 0, "delta": 0, "full": 0, "pred_delta": 0, "chunks": 0}
        self._total_fed: int = 0
        self._finished: bool = False

    def feed(self, data: bytes) -> int:
        """Feed data into the stream. Returns number of chunks emitted so far.

        Chunk boundaries are detected using a rolling Rabin fingerprint.
        On each boundary, the chunk is immediately encoded against the store.
        """
        if self._finished:
            raise RuntimeError("stream already finished")

        for byte in data:
            self._buf.append(byte)
            self._fp = ((self._fp << 1) ^ byte) & 0xFFFFFFFFFFFFFFFF
            self._total_fed += 1

            chunk_len = len(self._buf) - self._chunk_start
            if chunk_len < _MIN_CHUNK:
                continue

            if chunk_len >= _MAX_CHUNK or (self._fp & _MASK) == 0:
                # Chunk boundary detected — encode immediately
                chunk_data = bytes(self._buf[self._chunk_start:])
                self._emit_chunk(chunk_data)
                self._buf = bytearray()
                self._chunk_start = 0
                self._fp = 0

        return self._n_chunks

    def feed_line(self, line: str) -> int:
        """Convenience: feed a text line (with newline appended)."""
        return self.feed((line + "\n").encode("utf-8"))

    def _emit_chunk(self, chunk: bytes) -> None:
        """Encode a single chunk and append to the write-ahead log."""
        sha = sha256_hash(chunk)
        sh = simhash64(chunk)

        # 1. Exact match (zero cost)
        exact = self._store.lookup_exact(sha)
        if exact is not None:
            token = bytearray([REF])
            token += encode_uvarint(exact.chunk_id)
            self._stats["ref"] += 1
            self._chunk_ids.append(exact.chunk_id)
            self._tokens.append(bytes(token))
            self._n_chunks += 1
            return

        # 2. Try all candidate encodings, pick smallest
        full_bytes = _zstd_compress(chunk, level=self._zstd_level)
        full_token = bytearray([FULL])
        full_token += encode_uvarint(len(full_bytes))
        full_token += full_bytes
        best_token = bytes(full_token)
        best_type = "full"

        # 2a. Similarity delta
        similar = self._store.lookup_similar(sh)
        if similar is not None:
            delta_bytes = _compute_delta(similar.data, chunk)
            delta_token = bytearray([DELTA])
            delta_token += encode_uvarint(similar.chunk_id)
            delta_token += encode_uvarint(len(delta_bytes))
            delta_token += delta_bytes
            if len(delta_token) < len(best_token):
                best_token = bytes(delta_token)
                best_type = "delta"

        # 2b. Predictive pre-compression
        if self._predictor is not None and self._chunk_ids:
            last_id = self._chunk_ids[-1]
            pred_result = self._predictor.get_dictionary_and_ids(last_id)
            if pred_result is not None:
                pred_dict, dict_chunk_ids = pred_result
                try:
                    pred_bytes = _compress_with_dict(chunk, pred_dict, level=self._zstd_level)
                    pred_token = bytearray([PRED_DELTA])
                    pred_token += encode_uvarint(len(dict_chunk_ids))
                    for did in dict_chunk_ids:
                        pred_token += encode_uvarint(did)
                    pred_token += encode_uvarint(len(pred_bytes))
                    pred_token += pred_bytes
                    if len(pred_token) < len(best_token):
                        best_token = bytes(pred_token)
                        best_type = "pred_delta"
                except Exception:
                    pass

        # Emit the winning token
        self._stats[best_type] += 1
        entry = self._store.store(chunk)
        self._chunk_ids.append(entry.chunk_id)
        self._tokens.append(best_token)
        self._n_chunks += 1

    def finish(self) -> Tuple[bytes, dict]:
        """Finalize the stream and return a UCOG blob.

        Flushes any remaining buffered data as a final chunk,
        then assembles the write-ahead log into a valid UCOG blob.
        """
        if self._finished:
            raise RuntimeError("stream already finished")
        self._finished = True

        # Flush remaining buffer
        remaining = bytes(self._buf[self._chunk_start:])
        if remaining:
            self._emit_chunk(remaining)

        # Assemble UCOG blob from write-ahead log
        out = bytearray()
        out += MAGIC
        out.append(VERSION)
        out += encode_uvarint(self._n_chunks)
        for token in self._tokens:
            out += token

        self._stats["chunks"] = self._n_chunks

        # Post-encode: update co-occurrence
        if self._predictor is not None and len(self._chunk_ids) >= 2:
            self._predictor.update_after_encode(self._chunk_ids)

        if self._data_id and hasattr(self._store, 'register_data_chunks'):
            self._store.register_data_chunks(self._data_id, set(self._chunk_ids))

        return bytes(out), self._stats

    @property
    def chunks_emitted(self) -> int:
        return self._n_chunks

    @property
    def bytes_fed(self) -> int:
        return self._total_fed

    @property
    def current_ratio(self) -> float:
        """Current compression ratio (may change as more data arrives)."""
        if self._total_fed == 0:
            return 1.0
        compressed_size = sum(len(t) for t in self._tokens) + 6  # header overhead
        return self._total_fed / max(1, compressed_size)
