from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import struct

try:
    import zstandard as zstd
except Exception:
    zstd = None


MAGIC_RAW = b"PTF0"   # PF0 Two-Level Footer v0 (raw)
MAGIC_ZSTD = b"PTF1"  # PF0 Two-Level Footer v0 (zstd)


def _u16(x: int) -> bytes:
    return struct.pack("<H", x)


def _u32(x: int) -> bytes:
    return struct.pack("<I", x)


def _p_u16(b: bytes, off: int) -> tuple[int, int]:
    return struct.unpack_from("<H", b, off)[0], off + 2


def _p_u32(b: bytes, off: int) -> tuple[int, int]:
    return struct.unpack_from("<I", b, off)[0], off + 4


@dataclass
class PF0TwoLevelFooter:
    """
    Two-level index:
      - block blooms (small)
      - packet blooms (full)
    """
    m_bits: int
    k_hashes: int
    prefix_len: int
    group_size: int

    packet_count: int
    block_count: int

    block_blooms: List[bytes]      # length block_count
    packet_blooms: List[bytes]     # length packet_count


def _build_payload(footer: PF0TwoLevelFooter) -> bytes:
    bloom_bytes = footer.m_bits // 8

    payload = bytearray()
    payload += _u16(int(footer.m_bits))
    payload += _u16(int(bloom_bytes))
    payload += _u16(int(footer.k_hashes))
    payload += _u16(int(footer.prefix_len))
    payload += _u16(int(footer.group_size))

    payload += _u32(int(footer.packet_count))
    payload += _u32(int(footer.block_count))

    # block blooms
    for i in range(footer.block_count):
        b = footer.block_blooms[i] if i < len(footer.block_blooms) else b""
        if len(b) != bloom_bytes:
            b = b"\x00" * bloom_bytes
        payload += b

    # packet blooms
    for i in range(footer.packet_count):
        b = footer.packet_blooms[i] if i < len(footer.packet_blooms) else b""
        if len(b) != bloom_bytes:
            b = b"\x00" * bloom_bytes
        payload += b

    return bytes(payload)


def append_pf0_twolevel_footer(blob: bytes, footer: PF0TwoLevelFooter, compress: bool = True) -> bytes:
    payload = _build_payload(footer)

    if compress:
        if zstd is None:
            raise RuntimeError("zstandard is required for compressed PF0 twolevel footer")
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


def read_pf0_twolevel_footer(blob: bytes) -> Optional[PF0TwoLevelFooter]:
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
            raise RuntimeError("zstandard is required to read compressed PF0 twolevel footer")
        dctx = zstd.ZstdDecompressor()
        payload = dctx.decompress(payload)

    off = 0
    m_bits, off = _p_u16(payload, off)
    bloom_bytes, off = _p_u16(payload, off)
    k_hashes, off = _p_u16(payload, off)
    prefix_len, off = _p_u16(payload, off)
    group_size, off = _p_u16(payload, off)

    packet_count, off = _p_u32(payload, off)
    block_count, off = _p_u32(payload, off)

    if bloom_bytes * 8 != m_bits:
        return None

    need_blocks = block_count * bloom_bytes
    need_packets = packet_count * bloom_bytes
    if off + need_blocks + need_packets > len(payload):
        return None

    block_blooms: List[bytes] = []
    for _ in range(block_count):
        block_blooms.append(payload[off:off + bloom_bytes])
        off += bloom_bytes

    packet_blooms: List[bytes] = []
    for _ in range(packet_count):
        packet_blooms.append(payload[off:off + bloom_bytes])
        off += bloom_bytes

    return PF0TwoLevelFooter(
        m_bits=int(m_bits),
        k_hashes=int(k_hashes),
        prefix_len=int(prefix_len),
        group_size=int(group_size),
        packet_count=int(packet_count),
        block_count=int(block_count),
        block_blooms=block_blooms,
        packet_blooms=packet_blooms,
    )
