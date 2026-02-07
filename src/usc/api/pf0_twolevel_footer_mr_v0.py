from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import struct

try:
    import zstandard as zstd
except Exception:
    zstd = None


MAGIC_RAW = b"PMR0"   # PF0 Multi-Res Two-Level Footer v0 (raw)
MAGIC_ZSTD = b"PMR1"  # PF0 Multi-Res Two-Level Footer v0 (zstd)


def _u16(x: int) -> bytes:
    return struct.pack("<H", x)


def _u32(x: int) -> bytes:
    return struct.pack("<I", x)


def _p_u16(b: bytes, off: int) -> tuple[int, int]:
    return struct.unpack_from("<H", b, off)[0], off + 2


def _p_u32(b: bytes, off: int) -> tuple[int, int]:
    return struct.unpack_from("<I", b, off)[0], off + 4


@dataclass
class PF0TwoLevelFooterMR:
    """
    Multi-Resolution Two-Level Index:

      - Block blooms: high resolution (block_m_bits)
      - Packet blooms: low resolution  (packet_m_bits) to cut footer size

    We keep prefix_len + group_size shared.
    """
    prefix_len: int
    group_size: int

    block_m_bits: int
    block_k_hashes: int

    packet_m_bits: int
    packet_k_hashes: int

    packet_count: int
    block_count: int

    block_blooms: List[bytes]      # len = block_count, each = block_m_bits/8
    packet_blooms: List[bytes]     # len = packet_count, each = packet_m_bits/8


def _build_payload(f: PF0TwoLevelFooterMR) -> bytes:
    block_bytes = f.block_m_bits // 8
    packet_bytes = f.packet_m_bits // 8

    payload = bytearray()

    payload += _u16(int(f.prefix_len))
    payload += _u16(int(f.group_size))

    payload += _u16(int(f.block_m_bits))
    payload += _u16(int(block_bytes))
    payload += _u16(int(f.block_k_hashes))

    payload += _u16(int(f.packet_m_bits))
    payload += _u16(int(packet_bytes))
    payload += _u16(int(f.packet_k_hashes))

    payload += _u32(int(f.packet_count))
    payload += _u32(int(f.block_count))

    # block blooms
    for i in range(f.block_count):
        b = f.block_blooms[i] if i < len(f.block_blooms) else b""
        if len(b) != block_bytes:
            b = b"\x00" * block_bytes
        payload += b

    # packet blooms
    for i in range(f.packet_count):
        b = f.packet_blooms[i] if i < len(f.packet_blooms) else b""
        if len(b) != packet_bytes:
            b = b"\x00" * packet_bytes
        payload += b

    return bytes(payload)


def append_pf0_twolevel_footer_mr(blob: bytes, footer: PF0TwoLevelFooterMR, compress: bool = True) -> bytes:
    payload = _build_payload(footer)

    if compress:
        if zstd is None:
            raise RuntimeError("zstandard is required for compressed PF0 multi-res footer")
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


def read_pf0_twolevel_footer_mr(blob: bytes) -> Optional[PF0TwoLevelFooterMR]:
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
            raise RuntimeError("zstandard is required to read compressed PF0 multi-res footer")
        dctx = zstd.ZstdDecompressor()
        payload = dctx.decompress(payload)

    off = 0
    prefix_len, off = _p_u16(payload, off)
    group_size, off = _p_u16(payload, off)

    block_m_bits, off = _p_u16(payload, off)
    block_bytes, off = _p_u16(payload, off)
    block_k_hashes, off = _p_u16(payload, off)

    packet_m_bits, off = _p_u16(payload, off)
    packet_bytes, off = _p_u16(payload, off)
    packet_k_hashes, off = _p_u16(payload, off)

    packet_count, off = _p_u32(payload, off)
    block_count, off = _p_u32(payload, off)

    if block_bytes * 8 != block_m_bits:
        return None
    if packet_bytes * 8 != packet_m_bits:
        return None

    need = block_count * block_bytes + packet_count * packet_bytes
    if off + need > len(payload):
        return None

    block_blooms: List[bytes] = []
    for _ in range(block_count):
        block_blooms.append(payload[off:off + block_bytes])
        off += block_bytes

    packet_blooms: List[bytes] = []
    for _ in range(packet_count):
        packet_blooms.append(payload[off:off + packet_bytes])
        off += packet_bytes

    return PF0TwoLevelFooterMR(
        prefix_len=int(prefix_len),
        group_size=int(group_size),
        block_m_bits=int(block_m_bits),
        block_k_hashes=int(block_k_hashes),
        packet_m_bits=int(packet_m_bits),
        packet_k_hashes=int(packet_k_hashes),
        packet_count=int(packet_count),
        block_count=int(block_count),
        block_blooms=block_blooms,
        packet_blooms=packet_blooms,
    )
