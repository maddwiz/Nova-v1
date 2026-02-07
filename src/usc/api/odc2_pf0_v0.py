from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Set

import struct

try:
    import zstandard as zstd
except Exception:
    zstd = None


MAGIC = b"PF0\0"  # 4 bytes


@dataclass
class PF0Meta:
    group_size: int
    packet_count: int
    block_count: int


def _u16(x: int) -> bytes:
    return struct.pack("<H", x)


def _u32(x: int) -> bytes:
    return struct.pack("<I", x)


def pf0_encode_packets(
    packets: List[bytes],
    group_size: int = 2,
    zstd_level: int = 10,
) -> Tuple[bytes, PF0Meta]:
    """
    PF0 format (packet-framed blocks):

    Header:
      MAGIC (4)
      u16 version (=0)
      u16 group_size
      u32 packet_count
      u32 block_count

    For each block:
      u16 n_in_block
      u32 block_bytes
      offsets table: (n_in_block + 1) u32 offsets (relative to payload start)
      payload: concatenated zstd frames (one frame per packet)

    This enables random access decode for specific packets.
    """
    if zstd is None:
        raise RuntimeError("zstandard is required for PF0 codec")

    packet_count = len(packets)
    block_count = (packet_count + group_size - 1) // group_size if packet_count else 0

    cctx = zstd.ZstdCompressor(level=int(zstd_level))

    out = bytearray()
    out += MAGIC
    out += _u16(0)  # version
    out += _u16(int(group_size))
    out += _u32(int(packet_count))
    out += _u32(int(block_count))

    p = 0
    for _ in range(block_count):
        start = p
        end = min(packet_count, start + group_size)
        block_packets = packets[start:end]
        n_in_block = len(block_packets)

        frames: List[bytes] = []
        offsets: List[int] = [0]
        cur = 0

        for pkt in block_packets:
            fr = cctx.compress(pkt)
            frames.append(fr)
            cur += len(fr)
            offsets.append(cur)

        payload = b"".join(frames)
        offsets_table = b"".join(_u32(x) for x in offsets)

        block_bytes = len(offsets_table) + len(payload)

        out += _u16(int(n_in_block))
        out += _u32(int(block_bytes))
        out += offsets_table
        out += payload

        p = end

    return bytes(out), PF0Meta(group_size=group_size, packet_count=packet_count, block_count=block_count)


def _read_header(blob: bytes) -> Tuple[PF0Meta, int]:
    if len(blob) < 4 + 2 + 2 + 4 + 4:
        raise ValueError("PF0 blob too small")
    if blob[:4] != MAGIC:
        raise ValueError("Not PF0 blob")

    off = 4
    ver = struct.unpack_from("<H", blob, off)[0]
    off += 2
    if ver != 0:
        raise ValueError("Unsupported PF0 version")

    group_size = struct.unpack_from("<H", blob, off)[0]
    off += 2
    packet_count = struct.unpack_from("<I", blob, off)[0]
    off += 4
    block_count = struct.unpack_from("<I", blob, off)[0]
    off += 4

    return PF0Meta(group_size=group_size, packet_count=packet_count, block_count=block_count), off


def pf0_decode_packet_indices(blob: bytes, packet_indices: Set[int]) -> List[bytes]:
    """
    Random access decode: returns decoded packets for the requested indices.
    Output order is not guaranteed.
    """
    if zstd is None:
        raise RuntimeError("zstandard is required for PF0 codec")

    meta, off = _read_header(blob)

    want = {i for i in packet_indices if 0 <= i < meta.packet_count}
    if not want:
        return []

    dctx = zstd.ZstdDecompressor()
    out_packets: List[bytes] = []

    base_pi = 0
    for _ in range(meta.block_count):
        n_in_block = struct.unpack_from("<H", blob, off)[0]
        off += 2

        block_bytes = struct.unpack_from("<I", blob, off)[0]
        off += 4

        offsets_count = n_in_block + 1
        offsets_table_len = offsets_count * 4

        offsets = list(struct.unpack_from("<" + "I" * offsets_count, blob, off))
        off += offsets_table_len

        payload_start = off
        payload_end = off + (block_bytes - offsets_table_len)
        payload = memoryview(blob)[payload_start:payload_end]
        off = payload_end

        for j in range(n_in_block):
            pi = base_pi + j
            if pi not in want:
                continue
            a = offsets[j]
            z = offsets[j + 1]
            frame = payload[a:z].tobytes()
            out_packets.append(dctx.decompress(frame))

        base_pi += n_in_block

    return out_packets
