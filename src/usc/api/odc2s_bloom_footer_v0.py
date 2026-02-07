from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import struct

from usc.mem.block_bloom_index_v0 import BlockBloomIndex

try:
    import zstandard as zstd
except Exception:
    zstd = None


MAGIC_RAW = b"BBI0"   # raw payload
MAGIC_ZSTD = b"BBI1"  # zstd-compressed payload


def _u16(x: int) -> bytes:
    return struct.pack("<H", x)


def _u32(x: int) -> bytes:
    return struct.pack("<I", x)


def _p_u16(b: bytes, off: int) -> tuple[int, int]:
    return struct.unpack_from("<H", b, off)[0], off + 2


def _p_u32(b: bytes, off: int) -> tuple[int, int]:
    return struct.unpack_from("<I", b, off)[0], off + 4


def _build_payload(bbi: BlockBloomIndex, block_count: int) -> bytes:
    bloom_bytes = bbi.m_bits // 8

    payload = bytearray()
    payload += _u16(int(bbi.m_bits))
    payload += _u16(int(bloom_bytes))
    payload += _u16(int(bbi.k_hashes))
    payload += _u16(int(bbi.group_size))
    payload += _u32(int(block_count))

    # blooms in order: 0..block_count-1
    for bid in range(block_count):
        bb = bbi.block_blooms.get(bid)
        if bb is None:
            payload += bytes(bloom_bytes)
        else:
            if len(bb) != bloom_bytes:
                raise ValueError("Bloom length mismatch")
            payload += bb

    return bytes(payload)


def append_block_bloom_footer(blob: bytes, bbi: BlockBloomIndex, block_count: int, compress: bool = True) -> bytes:
    """
    Appends a portable block-bloom footer.

    Format:
      [blob_data][payload][MAGIC][u32 payload_len]

    payload is either:
      - RAW (MAGIC_RAW)  : uncompressed
      - ZSTD (MAGIC_ZSTD): zstd compressed, if available
    """
    payload_raw = _build_payload(bbi, block_count)

    if compress and zstd is not None:
        cctx = zstd.ZstdCompressor(level=7)
        payload = cctx.compress(payload_raw)
        magic = MAGIC_ZSTD
    else:
        payload = payload_raw
        magic = MAGIC_RAW

    out = blob + payload + magic + _u32(len(payload))
    return out


def _parse_payload_to_bbi(payload: bytes) -> Optional[BlockBloomIndex]:
    off = 0
    if len(payload) < 12:
        return None

    m_bits, off = _p_u16(payload, off)
    bloom_bytes, off = _p_u16(payload, off)
    k_hashes, off = _p_u16(payload, off)
    group_size, off = _p_u16(payload, off)
    block_count, off = _p_u32(payload, off)

    expected = 2 + 2 + 2 + 2 + 4 + (block_count * bloom_bytes)
    if len(payload) != expected:
        return None

    block_blooms: Dict[int, bytes] = {}
    for bid in range(block_count):
        bb = payload[off : off + bloom_bytes]
        off += bloom_bytes
        block_blooms[bid] = bytes(bb)

    return BlockBloomIndex(
        m_bits=int(m_bits),
        k_hashes=int(k_hashes),
        group_size=int(group_size),
        block_blooms=block_blooms,
    )


def read_block_bloom_footer(blob: bytes) -> Optional[BlockBloomIndex]:
    """
    Reads raw or zstd-compressed block bloom footer if present.
    """
    if len(blob) < 8:
        return None

    magic = blob[-8:-4]
    payload_len = struct.unpack_from("<I", blob, len(blob) - 4)[0]
    start = len(blob) - 8 - payload_len
    if start < 0:
        return None

    payload = blob[start : start + payload_len]

    if magic == MAGIC_RAW:
        return _parse_payload_to_bbi(payload)

    if magic == MAGIC_ZSTD:
        if zstd is None:
            return None
        try:
            dctx = zstd.ZstdDecompressor()
            raw = dctx.decompress(payload)
        except Exception:
            return None
        return _parse_payload_to_bbi(raw)

    return None
