from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import zstandard as zstd


MAGIC = b"USC_OUT1"  # exactly 8 bytes


@dataclass
class OuterStreamMeta:
    level: int
    packet_count: int
    raw_stream_bytes: int
    comp_stream_bytes: int


def _u32(x: int) -> bytes:
    return int(x).to_bytes(4, "little", signed=False)


def _read_u32(buf: bytes, off: int) -> Tuple[int, int]:
    return int.from_bytes(buf[off:off + 4], "little", signed=False), off + 4


def pack_packets(packets: List[bytes]) -> bytes:
    """
    Frame format:
      [count u32]
      repeated: [len u32][packet bytes]
    """
    out = bytearray()
    out += _u32(len(packets))
    for p in packets:
        out += _u32(len(p))
        out += p
    return bytes(out)


def unpack_packets(blob: bytes) -> List[bytes]:
    off = 0
    n, off = _read_u32(blob, off)
    out: List[bytes] = []
    for _ in range(n):
        ln, off = _read_u32(blob, off)
        out.append(blob[off:off + ln])
        off += ln
    return out


def compress_outerstream(packets: List[bytes], level: int = 10) -> Tuple[bytes, OuterStreamMeta]:
    """
    Bytes:
      [MAGIC 8B][level u32][raw_len u32][comp_len u32][zstd_bytes...]
    """
    raw = pack_packets(packets)

    cctx = zstd.ZstdCompressor(level=level)
    comp = cctx.compress(raw)

    hdr = bytearray()
    hdr += MAGIC
    hdr += _u32(level)
    hdr += _u32(len(raw))
    hdr += _u32(len(comp))
    hdr += comp

    meta = OuterStreamMeta(
        level=level,
        packet_count=len(packets),
        raw_stream_bytes=len(raw),
        comp_stream_bytes=len(comp),
    )
    return bytes(hdr), meta


def decompress_outerstream(blob: bytes) -> List[bytes]:
    if len(blob) < 8 + 4 + 4 + 4:
        raise ValueError("outerstream: blob too small")
    if blob[:8] != MAGIC:
        raise ValueError("outerstream: bad magic")

    off = 8
    _level, off = _read_u32(blob, off)
    raw_len, off = _read_u32(blob, off)
    comp_len, off = _read_u32(blob, off)

    comp = blob[off:off + comp_len]

    dctx = zstd.ZstdDecompressor()
    raw = dctx.decompress(comp)

    if len(raw) != raw_len:
        raise ValueError("outerstream: raw_len mismatch")

    return unpack_packets(raw)
