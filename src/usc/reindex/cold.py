"""
N10: Cold Re-indexing — add bloom index to already-encoded cold blobs.

Non-destructive: original blob preserved, index appended as wrapper.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Set

from usc.bloom.keyword import bloom_make, bloom_add, bloom_check
from usc.mem.varint import encode_uvarint, decode_uvarint

MAGIC = b"UIDX"  # 4 bytes — indexed wrapper

_WORD_RE = re.compile(r"[A-Za-z0-9_./:-]{2,}")


@dataclass
class ColdIndex:
    """Bloom index over a cold blob's content."""
    n_bits: int
    n_hashes: int
    bloom: bytes
    keywords: int  # number of keywords indexed


def reindex_cold_blob(
    decoded_texts: List[str],
    n_bits: int = 8192,
    n_hashes: int = 4,
) -> ColdIndex:
    """
    Build a bloom index from decoded text content.
    This can be attached to an already-encoded cold blob.
    """
    bits = bloom_make(n_bits)
    keyword_count = 0

    for text in decoded_texts:
        for word in _WORD_RE.findall(text.lower()):
            bloom_add(bits, n_bits, n_hashes, word)
            keyword_count += 1

    return ColdIndex(
        n_bits=n_bits,
        n_hashes=n_hashes,
        bloom=bytes(bits),
        keywords=keyword_count,
    )


def wrap_with_index(original_blob: bytes, index: ColdIndex) -> bytes:
    """
    Wrap an original blob with a prepended index.

    Wire format:
        UIDX (4B) + n_bits (u32) + n_hashes (u16) + bloom_len (uvarint) + bloom
        + original_len (uvarint) + original_blob
    """
    out = bytearray(MAGIC)
    out += index.n_bits.to_bytes(4, "little")
    out += index.n_hashes.to_bytes(2, "little")
    out += encode_uvarint(len(index.bloom))
    out += index.bloom
    out += encode_uvarint(len(original_blob))
    out += original_blob
    return bytes(out)


def unwrap_index(blob: bytes):
    """Unwrap an indexed blob into (ColdIndex, original_blob)."""
    if len(blob) < 4 or blob[:4] != MAGIC:
        raise ValueError("reindex: bad magic")

    off = 4
    n_bits = int.from_bytes(blob[off:off + 4], "little")
    off += 4
    n_hashes = int.from_bytes(blob[off:off + 2], "little")
    off += 2
    bloom_len, off = decode_uvarint(blob, off)
    bloom = blob[off:off + bloom_len]
    off += bloom_len
    orig_len, off = decode_uvarint(blob, off)
    original = blob[off:off + orig_len]

    index = ColdIndex(n_bits=n_bits, n_hashes=n_hashes, bloom=bloom, keywords=0)
    return index, bytes(original)


def query_cold_index(index: ColdIndex, keywords: Set[str]) -> bool:
    """Query the cold index to check if keywords might be present."""
    for kw in keywords:
        if not bloom_check(index.bloom, index.n_bits, index.n_hashes, kw.lower()):
            return False
    return True
