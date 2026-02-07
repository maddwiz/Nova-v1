from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import struct

try:
    import zstandard as zstd
except Exception:
    zstd = None


MAGIC_RAW = b"PBB0"   # PF0 Block Bloom footer v0 (raw)
MAGIC_ZSTD = b"PBB1"  # PF0 Block Bloom footer v0 (zstd)


def _u16(x: int) -> bytes:
    return struct.pack("<H", x)


def _u32(x: int) -> bytes:
    return struct.pack("<I", x)


def _p_u16(b: bytes, off: int) -> tuple[int, int]:
    return struct.unpack_from("<H", b, off)[0], off + 2


def _p_u32(b: bytes, off: int) -> tuple[int, int]:
    return struct.unpack_from("<I", b, off)[0], off + 4


@dataclass
class PF0BlockBloomFooter:
    """
    Stores BLOCK bloom filters inside PF0 blob.
    Much smaller than packet-level blooms.

    We do NOT implement query hashing here.
    We reuse existing query_blocks_for_keywords() logic.
    """
    m_bits: int
    k_hashes: int
    prefix_len: int
    group_size: int
    block_count: int
    block_blooms: List[bytes]


def _build_payload(footer: PF0BlockBloomFooter) -> bytes:
    bloom_bytes = footer.m_bits // 8
    payload = bytearray()

    payload += _u16(int(footer.m_bits))
    payload += _u16(int(bloom_bytes))
    payload += _u16(int(footer.k_hashes))
    payload += _u16(int(footer.prefix_len))
    payload += _u16(int(footer.group_size))
    payload += _u32(int(footer.block_count))

    for i in range(footer.block_count):
        b = footer.block_blooms[i] if i < len(footer.block_blooms) else b""
        if len(b) != bloom_bytes:
            b = b"\x00" * bloom_bytes
        payload += b

    return bytes(payload)


def append_pf0_block_bloom_footer(blob: bytes, footer: PF0BlockBloomFooter, compress: bool = True) -> bytes:
    """
    Appends BLOCK bloom footer to the end of blob.

    Layout:
      payload_bytes
      MAGIC (4)
      u32 payload_len
    """
    payload = _build_payload(footer)

    if compress:
        if zstd is None:
            raise RuntimeError("zstandard is required for compressed PF0 block bloom footer")
        cctx = zstd.ZstdCompressor(level=10)
        payload_final = cctx.compress(payload)
        magic = MAGIC_ZSTD
    else:
        payload_final = payload
        magic = MAGIC_RAW

    out = bytearray(blob)
    out += payload_final
    out += magic
    out += _u32(len(payload_final))
    return bytes(out)


def read_pf0_block_bloom_footer(blob: bytes) -> Optional[PF0BlockBloomFooter]:
    if len(blob) < 8:
        return None

    magic = blob[-8:-4]
    payload_len = struct.unpack_from("<I", blob, len(blob) - 4)[0]

    if magic not in (MAGIC_RAW, MAGIC_ZSTD):
        return None

    payload_start = len(blob) - 8 - payload_len
    if payload_start < 0:
        return None

    payload = blob[payload_start:payload_start + payload_len]

    if magic == MAGIC_ZSTD:
        if zstd is None:
            raise RuntimeError("zstandard is required to read compressed PF0 block bloom footer")
        dctx = zstd.ZstdDecompressor()
        payload = dctx.decompress(payload)

    off = 0
    m_bits, off = _p_u16(payload, off)
    bloom_bytes, off = _p_u16(payload, off)
    k_hashes, off = _p_u16(payload, off)
    prefix_len, off = _p_u16(payload, off)
    group_size, off = _p_u16(payload, off)
    block_count, off = _p_u32(payload, off)

    if bloom_bytes * 8 != m_bits:
        return None

    need = block_count * bloom_bytes
    if off + need > len(payload):
        return None

    block_blooms: List[bytes] = []
    for _ in range(block_count):
        block_blooms.append(payload[off:off + bloom_bytes])
        off += bloom_bytes

    return PF0BlockBloomFooter(
        m_bits=int(m_bits),
        k_hashes=int(k_hashes),
        prefix_len=int(prefix_len),
        group_size=int(group_size),
        block_count=int(block_count),
        block_blooms=block_blooms,
    )
