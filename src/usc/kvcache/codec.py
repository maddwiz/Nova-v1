"""
N12: KV-Cache Codec — specialized compression for key-value cache state.
"""
from __future__ import annotations

import struct
from typing import Dict, List, Tuple

import zstandard as zstd

from usc.mem.varint import encode_uvarint, decode_uvarint

MAGIC = b"UKVC"  # 4 bytes


def kvcache_encode(kv_pairs: Dict[str, bytes], level: int = 10) -> bytes:
    """
    Encode a dictionary of key-value pairs.

    Wire format:
        UKVC (4B) + n_pairs (uvarint)
        + For each pair: key_len (uvarint) + key + val_len (uvarint) + val
        All compressed with zstd.
    """
    inner = bytearray()
    inner += encode_uvarint(len(kv_pairs))
    for key, val in sorted(kv_pairs.items()):
        key_bytes = key.encode("utf-8")
        inner += encode_uvarint(len(key_bytes))
        inner += key_bytes
        inner += encode_uvarint(len(val))
        inner += val

    cctx = zstd.ZstdCompressor(level=level)
    compressed = cctx.compress(bytes(inner))

    out = bytearray(MAGIC)
    out += encode_uvarint(len(compressed))
    out += compressed
    return bytes(out)


def kvcache_decode(blob: bytes) -> Dict[str, bytes]:
    """Decode a UKVC blob back to key-value pairs."""
    if len(blob) < 4:
        raise ValueError("kvcache: blob too small")
    if blob[:4] != MAGIC:
        raise ValueError("kvcache: bad magic")

    off = 4
    comp_len, off = decode_uvarint(blob, off)
    compressed = blob[off:off + comp_len]

    dctx = zstd.ZstdDecompressor()
    inner = dctx.decompress(compressed)

    pos = 0
    n_pairs, pos = decode_uvarint(inner, pos)
    result: Dict[str, bytes] = {}
    for _ in range(n_pairs):
        key_len, pos = decode_uvarint(inner, pos)
        key = inner[pos:pos + key_len].decode("utf-8")
        pos += key_len
        val_len, pos = decode_uvarint(inner, pos)
        val = inner[pos:pos + val_len]
        pos += val_len
        result[key] = bytes(val)

    return result


def kvcache_delta_encode(
    old_kv: Dict[str, bytes],
    new_kv: Dict[str, bytes],
    level: int = 10,
) -> bytes:
    """
    Delta-encode a KV cache update: only stores changed/added keys.

    Wire format:
        UKVC (4B) + 0x01 (delta flag) + n_changes (uvarint)
        + For each change: op (1B) + key_len + key [+ val_len + val]
        op: 0x01 = set, 0x02 = delete
    """
    changes = bytearray()
    n_changes = 0

    # Added or changed keys
    for key in sorted(new_kv.keys()):
        if key not in old_kv or old_kv[key] != new_kv[key]:
            changes += b"\x01"  # set
            key_bytes = key.encode("utf-8")
            changes += encode_uvarint(len(key_bytes))
            changes += key_bytes
            changes += encode_uvarint(len(new_kv[key]))
            changes += new_kv[key]
            n_changes += 1

    # Deleted keys
    for key in sorted(old_kv.keys()):
        if key not in new_kv:
            changes += b"\x02"  # delete
            key_bytes = key.encode("utf-8")
            changes += encode_uvarint(len(key_bytes))
            changes += key_bytes
            n_changes += 1

    inner = bytearray()
    inner += encode_uvarint(n_changes)
    inner += changes

    cctx = zstd.ZstdCompressor(level=level)
    compressed = cctx.compress(bytes(inner))

    out = bytearray(MAGIC)
    out += b"\x01"  # delta flag
    out += encode_uvarint(len(compressed))
    out += compressed
    return bytes(out)


def kvcache_delta_decode(
    old_kv: Dict[str, bytes],
    delta: bytes,
) -> Dict[str, bytes]:
    """Apply a delta to reconstruct the new KV cache state."""
    if len(delta) < 5 or delta[:4] != MAGIC:
        raise ValueError("kvcache: bad delta magic")
    if delta[4] != 0x01:
        raise ValueError("kvcache: not a delta blob")

    off = 5
    comp_len, off = decode_uvarint(delta, off)
    compressed = delta[off:off + comp_len]

    dctx = zstd.ZstdDecompressor()
    inner = dctx.decompress(compressed)

    pos = 0
    n_changes, pos = decode_uvarint(inner, pos)

    result = dict(old_kv)
    for _ in range(n_changes):
        op = inner[pos]
        pos += 1
        key_len, pos = decode_uvarint(inner, pos)
        key = inner[pos:pos + key_len].decode("utf-8")
        pos += key_len

        if op == 0x01:  # set
            val_len, pos = decode_uvarint(inner, pos)
            val = inner[pos:pos + val_len]
            pos += val_len
            result[key] = bytes(val)
        elif op == 0x02:  # delete
            result.pop(key, None)

    return result
