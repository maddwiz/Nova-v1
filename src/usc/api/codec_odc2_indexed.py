from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import zstandard as zstd

from usc.mem.outerstream_zstd import pack_packets, unpack_packets
from usc.mem.zstd_trained_dict import train_dict


MAGIC = b"USC_ODC2"  # 8 bytes


@dataclass
class ODC2BlockIndex:
    packet_start: int
    packet_count: int
    raw_len: int
    comp_len: int


@dataclass
class ODC2Meta:
    level: int
    dict_bytes: int
    packet_count: int
    group_size: int
    block_count: int
    total_comp_bytes: int
    used_mode: str  # "dict" or "plain"


def _u32(x: int) -> bytes:
    return int(x).to_bytes(4, "little", signed=False)


def _read_u32(buf: bytes, off: int) -> Tuple[int, int]:
    return int.from_bytes(buf[off:off + 4], "little", signed=False), off + 4


def _group_packets(packets: List[bytes], group_size: int) -> List[List[bytes]]:
    if group_size < 1:
        group_size = 1
    groups: List[List[bytes]] = []
    for i in range(0, len(packets), group_size):
        groups.append(packets[i:i + group_size])
    return groups


def odc2_encode_packets(
    packets: List[bytes],
    level: int = 10,
    dict_target_size: int = 8192,
    sample_chunk_size: int = 1024,
    group_size: int = 4,
) -> Tuple[bytes, ODC2Meta]:
    """
    ODC2 format: indexed block compression for selective replay.

    Header:
      [MAGIC 8B]
      [level u32]
      [dict_len u32]
      [dict_bytes...  (dict_len may be 0 => plain mode)]
      [packet_count u32]
      [group_size u32]
      [block_count u32]
      index table (block_count entries):
         [packet_start u32][packet_count u32][raw_len u32][comp_len u32]
      then concatenated compressed blocks

    Each block is: pack_packets(block_packets) then zstd compress (dict or plain).
    """
    packet_count = len(packets)
    groups = _group_packets(packets, group_size)
    block_count = len(groups)

    # Attempt dictionary training on full framed stream.
    # If it fails (small inputs), fall back to plain zstd.
    full_framed = pack_packets(packets)
    samples = [full_framed[i:i + sample_chunk_size] for i in range(0, len(full_framed), sample_chunk_size)]

    dict_bytes = b""
    used_mode = "plain"

    try:
        bundle = train_dict(samples, dict_size=dict_target_size)
        dict_bytes = bundle.dict_bytes
        cctx = zstd.ZstdCompressor(level=level, dict_data=bundle.cdict)
        used_mode = "dict"
    except Exception:
        cctx = zstd.ZstdCompressor(level=level)

    index: List[ODC2BlockIndex] = []
    comp_blocks: List[bytes] = []

    pkt_cursor = 0
    total_comp = 0

    for g in groups:
        raw_block = pack_packets(g)
        comp_block = cctx.compress(raw_block)

        index.append(
            ODC2BlockIndex(
                packet_start=pkt_cursor,
                packet_count=len(g),
                raw_len=len(raw_block),
                comp_len=len(comp_block),
            )
        )
        comp_blocks.append(comp_block)

        pkt_cursor += len(g)
        total_comp += len(comp_block)

    out = bytearray()
    out += MAGIC
    out += _u32(level)
    out += _u32(len(dict_bytes))
    out += dict_bytes
    out += _u32(packet_count)
    out += _u32(group_size)
    out += _u32(block_count)

    # index table
    for b in index:
        out += _u32(b.packet_start)
        out += _u32(b.packet_count)
        out += _u32(b.raw_len)
        out += _u32(b.comp_len)

    # blocks
    for cb in comp_blocks:
        out += cb

    meta = ODC2Meta(
        level=level,
        dict_bytes=len(dict_bytes),
        packet_count=packet_count,
        group_size=group_size,
        block_count=block_count,
        total_comp_bytes=total_comp,
        used_mode=used_mode,
    )
    return bytes(out), meta


def _parse_odc2(blob: bytes) -> Tuple[int, bytes, int, int, int, List[ODC2BlockIndex], int]:
    """
    Returns:
      level, dict_bytes, packet_count, group_size, block_count, index_list, data_offset
    """
    if len(blob) < 8 + 4 + 4 + 4 + 4 + 4:
        raise ValueError("odc2: blob too small")
    if blob[:8] != MAGIC:
        raise ValueError("odc2: bad magic")

    off = 8
    level, off = _read_u32(blob, off)
    dict_len, off = _read_u32(blob, off)
    dict_bytes = blob[off:off + dict_len]
    off += dict_len

    packet_count, off = _read_u32(blob, off)
    group_size, off = _read_u32(blob, off)
    block_count, off = _read_u32(blob, off)

    index: List[ODC2BlockIndex] = []
    for _ in range(block_count):
        ps, off = _read_u32(blob, off)
        pc, off = _read_u32(blob, off)
        rl, off = _read_u32(blob, off)
        cl, off = _read_u32(blob, off)
        index.append(ODC2BlockIndex(ps, pc, rl, cl))

    data_offset = off
    return level, dict_bytes, packet_count, group_size, block_count, index, data_offset


def odc2_decode_all_packets(blob: bytes) -> List[bytes]:
    _level, dict_bytes, packet_count, _group_size, _block_count, index, data_off = _parse_odc2(blob)

    # dict_len==0 => plain mode
    if len(dict_bytes) > 0:
        ddict = zstd.ZstdCompressionDict(dict_bytes)
        dctx = zstd.ZstdDecompressor(dict_data=ddict)
    else:
        dctx = zstd.ZstdDecompressor()

    out_packets: List[bytes] = []
    off = data_off

    for b in index:
        comp = blob[off:off + b.comp_len]
        off += b.comp_len
        raw = dctx.decompress(comp)
        pkts = unpack_packets(raw)
        out_packets.extend(pkts)

    if len(out_packets) != packet_count:
        raise ValueError(f"odc2: packet_count mismatch (expected {packet_count}, got {len(out_packets)})")

    return out_packets


def odc2_decode_packet_range(blob: bytes, start: int, end: int) -> List[bytes]:
    """
    Selective decode: returns packets [start:end].
    Only decompress blocks that overlap this range.
    """
    if start < 0:
        start = 0
    if end < start:
        end = start

    _level, dict_bytes, packet_count, _group_size, _block_count, index, data_off = _parse_odc2(blob)

    if start >= packet_count:
        return []
    if end > packet_count:
        end = packet_count

    # dict_len==0 => plain mode
    if len(dict_bytes) > 0:
        ddict = zstd.ZstdCompressionDict(dict_bytes)
        dctx = zstd.ZstdDecompressor(dict_data=ddict)
    else:
        dctx = zstd.ZstdDecompressor()

    # compute block byte offsets
    block_offsets: List[int] = []
    o = data_off
    for b in index:
        block_offsets.append(o)
        o += b.comp_len

    out: List[bytes] = []

    for bi, b in enumerate(index):
        b_start = b.packet_start
        b_end = b.packet_start + b.packet_count

        if b_end <= start or b_start >= end:
            continue

        comp = blob[block_offsets[bi]:block_offsets[bi] + b.comp_len]
        raw = dctx.decompress(comp)
        pkts = unpack_packets(raw)

        for j, p in enumerate(pkts):
            pkt_i = b_start + j
            if start <= pkt_i < end:
                out.append(p)

    return out
