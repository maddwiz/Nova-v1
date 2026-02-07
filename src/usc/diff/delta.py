"""
N6: Diff/Delta Storage — binary delta encoding using zstd dictionary context.
"""
from __future__ import annotations

from typing import Tuple

import zstandard as zstd

from usc.mem.varint import encode_uvarint, decode_uvarint

MAGIC = b"UDIF"  # 4 bytes


def delta_encode(old_blob: bytes, new_blob: bytes, level: int = 10) -> bytes:
    """
    Encode new_blob as a delta against old_blob.
    Uses old_blob as a zstd dictionary for context-aware compression.

    Wire format:
        UDIF (4B) + old_hash_prefix (8B) + old_len (uvarint)
        + compressed_len (uvarint) + zstd_compressed_new

    The old_hash_prefix lets the decoder verify it has the right base.
    """
    import hashlib
    old_hash = hashlib.sha256(old_blob).digest()[:8]

    # Use old_blob as dictionary for compressing new_blob
    if len(old_blob) > 0:
        zdict = zstd.ZstdCompressionDict(old_blob)
        cctx = zstd.ZstdCompressor(level=level, dict_data=zdict)
    else:
        cctx = zstd.ZstdCompressor(level=level)

    compressed = cctx.compress(new_blob)

    out = bytearray(MAGIC)
    out += old_hash
    out += encode_uvarint(len(old_blob))
    out += encode_uvarint(len(compressed))
    out += compressed
    return bytes(out)


def delta_decode(old_blob: bytes, delta: bytes) -> bytes:
    """Reconstruct new_blob from old_blob + delta."""
    if len(delta) < 4:
        raise ValueError("delta: blob too small")
    if delta[:4] != MAGIC:
        raise ValueError("delta: bad magic")

    import hashlib
    old_hash = hashlib.sha256(old_blob).digest()[:8]

    off = 4
    stored_hash = delta[off:off + 8]
    off += 8
    if stored_hash != old_hash:
        raise ValueError("delta: old_blob hash mismatch — wrong base")

    old_len, off = decode_uvarint(delta, off)
    if old_len != len(old_blob):
        raise ValueError("delta: old_blob length mismatch")

    comp_len, off = decode_uvarint(delta, off)
    compressed = delta[off:off + comp_len]

    if len(old_blob) > 0:
        zdict = zstd.ZstdCompressionDict(old_blob)
        dctx = zstd.ZstdDecompressor(dict_data=zdict)
    else:
        dctx = zstd.ZstdDecompressor()

    return dctx.decompress(compressed)
