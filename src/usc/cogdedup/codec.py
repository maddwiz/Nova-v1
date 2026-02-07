"""Cognitive Deduplication codec — encode/decode with C3 memory backing.

Wire format: UCOG (4 bytes)
  version (u8) = 2
  n_chunks (uvarint)
  for each chunk:
    chunk_type (u8):
      0x00 = REF   → chunk_id (uvarint)
      0x01 = DELTA → ref_chunk_id (uvarint) + delta_len (uvarint) + delta_bytes
      0x02 = FULL  → data_len (uvarint) + zstd(data)
      0x03 = PRED_DELTA → n_dict_ids (uvarint) + dict_id_1..N (uvarint each)
                          + delta_len (uvarint) + delta_bytes
                          Dictionary is rebuilt from exact chunk IDs at decode time.

v2 additions:
- Predictive pre-compression (PRED_DELTA)
- Co-occurrence tracking after encode
- Data-to-chunks mapping for retrieval integration
"""
from __future__ import annotations

import struct
from typing import Dict, List, Optional, Set, Tuple

try:
    import zstandard as zstd
except ImportError:
    zstd = None

from usc.cogdedup.hasher import (
    content_defined_chunks,
    sha256_hash,
    simhash64,
    hamming_distance,
    SIMILARITY_THRESHOLD,
)
from usc.cogdedup.store import CogStore, ChunkEntry
from usc.cogdedup.predictor import PredictiveCompressor
from usc.mem.varint import encode_uvarint, decode_uvarint


MAGIC = b"UCOG"
VERSION = 2  # v2: predictive pre-compression

# Chunk types
REF = 0x00
DELTA = 0x01
FULL = 0x02
PRED_DELTA = 0x03


def _zstd_compress(data: bytes, level: int = 10) -> bytes:
    if zstd is None:
        raise RuntimeError("zstandard required")
    return zstd.ZstdCompressor(level=level).compress(data)


def _zstd_decompress(data: bytes) -> bytes:
    if zstd is None:
        raise RuntimeError("zstandard required")
    return zstd.ZstdDecompressor().decompress(data)


def _compute_delta(src: bytes, dst: bytes) -> bytes:
    """Simple delta: zstd compress dst using src as dictionary."""
    if zstd is None:
        raise RuntimeError("zstandard required")
    dict_data = zstd.ZstdCompressionDict(src)
    cctx = zstd.ZstdCompressor(level=10, dict_data=dict_data)
    return cctx.compress(dst)


def _apply_delta(src: bytes, delta: bytes) -> bytes:
    """Apply delta: decompress with src as dictionary."""
    if zstd is None:
        raise RuntimeError("zstandard required")
    dict_data = zstd.ZstdCompressionDict(src)
    dctx = zstd.ZstdDecompressor(dict_data=dict_data)
    return dctx.decompress(delta)


def _compress_with_dict(data: bytes, dict_data: "zstd.ZstdCompressionDict", level: int = 10) -> bytes:
    """Compress using a pre-built predictive dictionary."""
    if zstd is None:
        raise RuntimeError("zstandard required")
    cctx = zstd.ZstdCompressor(level=level, dict_data=dict_data)
    return cctx.compress(data)


def _decompress_with_dict(data: bytes, dict_data: "zstd.ZstdCompressionDict") -> bytes:
    """Decompress using a pre-built predictive dictionary."""
    if zstd is None:
        raise RuntimeError("zstandard required")
    dctx = zstd.ZstdDecompressor(dict_data=dict_data)
    return dctx.decompress(data)


def cogdedup_encode(
    data: bytes,
    store: CogStore,
    *,
    zstd_level: int = 10,
    data_id: str = "",
    predictor: Optional[PredictiveCompressor] = None,
) -> Tuple[bytes, dict]:
    """Encode data using cognitive deduplication.

    Returns (blob, stats) where stats contains counts of REF/DELTA/FULL/PRED_DELTA chunks.
    Side effects:
    - Stores new chunks in the CogStore
    - Updates co-occurrence data if predictor is provided
    - Registers data-to-chunks mapping if data_id is provided

    Args:
        data: Raw bytes to encode
        store: CogStore backend
        zstd_level: Compression level for FULL chunks
        data_id: Optional ID for compression-aware retrieval mapping
        predictor: Optional PredictiveCompressor for anticipatory compression
    """
    chunks = content_defined_chunks(data)
    if not chunks:
        chunks = [data] if data else [b""]

    out = bytearray()
    out += MAGIC
    out.append(VERSION)
    out += encode_uvarint(len(chunks))

    stats = {"ref": 0, "delta": 0, "full": 0, "pred_delta": 0, "chunks": len(chunks)}
    chunk_ids_in_batch: List[int] = []

    for chunk in chunks:
        sha = sha256_hash(chunk)
        sh = simhash64(chunk)

        # 1. Try exact match first (zero cost — just a reference)
        exact = store.lookup_exact(sha)
        if exact is not None:
            out.append(REF)
            out += encode_uvarint(exact.chunk_id)
            stats["ref"] += 1
            chunk_ids_in_batch.append(exact.chunk_id)
            continue

        # 2. Compute all candidate encodings, pick the smallest
        full_bytes = _zstd_compress(chunk, level=zstd_level)
        full_token = bytearray([FULL])
        full_token += encode_uvarint(len(full_bytes))
        full_token += full_bytes
        best_token = bytes(full_token)
        best_type = "full"

        # 2a. Similarity delta
        similar = store.lookup_similar(sh)
        if similar is not None:
            delta_bytes = _compute_delta(similar.data, chunk)
            delta_token = bytearray([DELTA])
            delta_token += encode_uvarint(similar.chunk_id)
            delta_token += encode_uvarint(len(delta_bytes))
            delta_token += delta_bytes
            if len(delta_token) < len(best_token):
                best_token = bytes(delta_token)
                best_type = "delta"

        # 2b. Predictive pre-compression (only if no exact/similarity delta)
        if predictor is not None and chunk_ids_in_batch:
            last_id = chunk_ids_in_batch[-1]
            pred_result = predictor.get_dictionary_and_ids(last_id)
            if pred_result is not None:
                pred_dict, dict_chunk_ids = pred_result
                try:
                    pred_bytes = _compress_with_dict(chunk, pred_dict, level=zstd_level)
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
        out += best_token
        stats[best_type] += 1
        entry = store.store(chunk)
        chunk_ids_in_batch.append(entry.chunk_id)

    # Post-encode: update co-occurrence and data mapping
    if predictor is not None and len(chunk_ids_in_batch) >= 2:
        predictor.update_after_encode(chunk_ids_in_batch)

    if data_id and hasattr(store, 'register_data_chunks'):
        store.register_data_chunks(data_id, set(chunk_ids_in_batch))

    return bytes(out), stats


def cogdedup_decode(blob: bytes, store: CogStore,
                    predictor: Optional[PredictiveCompressor] = None) -> bytes:
    """Decode a UCOG blob back to original data.

    Requires the same CogStore that was used during encoding.
    Predictor is needed for PRED_DELTA chunks.
    """
    if blob[:4] != MAGIC:
        raise ValueError(f"not a UCOG blob (got {blob[:4]!r})")

    off = 4
    ver = blob[off]
    off += 1
    if ver not in (1, 2):
        raise ValueError(f"unsupported UCOG version {ver}")

    n_chunks, off = decode_uvarint(blob, off)
    parts: List[bytes] = []

    for _ in range(n_chunks):
        chunk_type = blob[off]
        off += 1

        if chunk_type == REF:
            chunk_id, off = decode_uvarint(blob, off)
            entry = store.get(chunk_id)
            if entry is None:
                raise ValueError(f"REF to unknown chunk_id={chunk_id}")
            parts.append(entry.data)

        elif chunk_type == DELTA:
            ref_id, off = decode_uvarint(blob, off)
            delta_len, off = decode_uvarint(blob, off)
            delta_bytes = blob[off:off + delta_len]
            off += delta_len

            ref_entry = store.get(ref_id)
            if ref_entry is None:
                raise ValueError(f"DELTA ref to unknown chunk_id={ref_id}")
            reconstructed = _apply_delta(ref_entry.data, delta_bytes)
            parts.append(reconstructed)

        elif chunk_type == FULL:
            data_len, off = decode_uvarint(blob, off)
            compressed = blob[off:off + data_len]
            off += data_len
            parts.append(_zstd_decompress(compressed))

        elif chunk_type == PRED_DELTA:
            # Read the exact chunk IDs that formed the dictionary
            n_dict_ids, off = decode_uvarint(blob, off)
            dict_chunk_ids = []
            for _ in range(n_dict_ids):
                did, off = decode_uvarint(blob, off)
                dict_chunk_ids.append(did)
            delta_len, off = decode_uvarint(blob, off)
            delta_bytes = blob[off:off + delta_len]
            off += delta_len

            # Rebuild dictionary from the exact chunk IDs stored in the blob
            dict_content = b""
            for did in dict_chunk_ids:
                entry = store.get(did)
                if entry is not None and entry.data:
                    dict_content += entry.data

            if not dict_content:
                raise ValueError(
                    f"PRED_DELTA: cannot rebuild dictionary from chunk_ids={dict_chunk_ids}"
                )

            dict_data = zstd.ZstdCompressionDict(dict_content)
            reconstructed = _decompress_with_dict(delta_bytes, dict_data)
            parts.append(reconstructed)

        else:
            raise ValueError(f"unknown chunk type 0x{chunk_type:02x}")

    return b"".join(parts)
