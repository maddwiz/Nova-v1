from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import struct

try:
    import zstandard as zstd
except Exception:
    zstd = None


MAGIC_RAW = b"PBI0"   # PF0 Bloom Index v0 (raw)
MAGIC_ZSTD = b"PBI1"  # PF0 Bloom Index v0 (zstd-compressed)


def _u16(x: int) -> bytes:
    return struct.pack("<H", x)


def _u32(x: int) -> bytes:
    return struct.pack("<I", x)


def _p_u16(b: bytes, off: int) -> tuple[int, int]:
    return struct.unpack_from("<H", b, off)[0], off + 2


def _p_u32(b: bytes, off: int) -> tuple[int, int]:
    return struct.unpack_from("<I", b, off)[0], off + 4


@dataclass
class PF0BloomFooter:
    """
    Stores packet bloom filters inside PF0 blob.
    IMPORTANT: we do NOT implement query hashing here.
    We only serialize/deserialize blooms, and we reuse the existing query code.
    """
    m_bits: int
    k_hashes: int
    prefix_len: int
    packet_count: int
    packet_blooms: List[bytes]


def _build_payload(footer: PF0BloomFooter) -> bytes:
    bloom_bytes = footer.m_bits // 8
    payload = bytearray()

    payload += _u16(int(footer.m_bits))
    payload += _u16(int(bloom_bytes))
    payload += _u16(int(footer.k_hashes))
    payload += _u16(int(footer.prefix_len))
    payload += _u32(int(footer.packet_count))

    for i in range(footer.packet_count):
        b = footer.packet_blooms[i] if i < len(footer.packet_blooms) else b""
        if len(b) != bloom_bytes:
            b = b"\x00" * bloom_bytes
        payload += b

    return bytes(payload)


def append_pf0_bloom_footer(blob: bytes, footer: PF0BloomFooter, compress: bool = True) -> bytes:
    """
    Layout appended to end:
      payload_bytes
      MAGIC (4)
      u32 payload_len
    """
    payload = _build_payload(footer)

    if compress:
        if zstd is None:
            raise RuntimeError("zstandard is required for compressed PF0 bloom footer")
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


def read_pf0_bloom_footer(blob: bytes) -> Optional[PF0BloomFooter]:
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
            raise RuntimeError("zstandard is required to read compressed PF0 bloom footer")
        dctx = zstd.ZstdDecompressor()
        payload = dctx.decompress(payload)

    off = 0
    m_bits, off = _p_u16(payload, off)
    bloom_bytes, off = _p_u16(payload, off)
    k_hashes, off = _p_u16(payload, off)
    prefix_len, off = _p_u16(payload, off)
    packet_count, off = _p_u32(payload, off)

    if bloom_bytes * 8 != m_bits:
        return None

    need = packet_count * bloom_bytes
    if off + need > len(payload):
        return None

    packet_blooms: List[bytes] = []
    for _ in range(packet_count):
        packet_blooms.append(payload[off:off + bloom_bytes])
        off += bloom_bytes

    return PF0BloomFooter(
        m_bits=int(m_bits),
        k_hashes=int(k_hashes),
        prefix_len=int(prefix_len),
        packet_count=int(packet_count),
        packet_blooms=packet_blooms,
    )
