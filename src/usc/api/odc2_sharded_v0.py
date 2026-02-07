from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Set, Optional

import zstandard as zstd


MAGIC = b"USC_ODC2S0"  # Sharded ODC2-like container v0


# -----------------------------
# Varint helpers
# -----------------------------
def uvarint_encode(x: int) -> bytes:
    x = int(x)
    if x < 0:
        raise ValueError("uvarint cannot encode negative")
    out = bytearray()
    while True:
        b = x & 0x7F
        x >>= 7
        if x:
            out.append(b | 0x80)
        else:
            out.append(b)
            break
    return bytes(out)


def uvarint_decode(buf: bytes, off: int) -> Tuple[int, int]:
    x = 0
    shift = 0
    while True:
        if off >= len(buf):
            raise ValueError("uvarint decode overflow")
        b = buf[off]
        off += 1
        x |= (b & 0x7F) << shift
        if (b & 0x80) == 0:
            return x, off
        shift += 7
        if shift > 63:
            raise ValueError("uvarint too large")


# -----------------------------
# Container structs
# -----------------------------
@dataclass
class ODC2SMeta:
    dict_bytes: int
    group_size: int
    block_count: int
    total_packets: int


def _pack_block(packets: List[bytes]) -> bytes:
    """
    Block plaintext format:
      uvarint(num_packets)
      repeat:
        uvarint(len) + bytes
    """
    out = bytearray()
    out += uvarint_encode(len(packets))
    for p in packets:
        out += uvarint_encode(len(p))
        out += p
    return bytes(out)


def _unpack_block(block_plain: bytes) -> List[bytes]:
    off = 0
    n, off = uvarint_decode(block_plain, off)
    out: List[bytes] = []
    for _ in range(n):
        ln, off = uvarint_decode(block_plain, off)
        out.append(block_plain[off:off + ln])
        off += ln
    return out


def odc2s_encode_packets(
    packets: List[bytes],
    group_size: int = 8,
    dict_target_size: int = 8192,
    zstd_level: int = 10,
    sample_blocks: int = 64,
) -> Tuple[bytes, ODC2SMeta]:
    """
    Encodes packets into a sharded container with:
      - global zstd dictionary trained on first N block plaintexts
      - each block compressed independently

    Container format:
      MAGIC
      uvarint(group_size)
      uvarint(total_packets)
      uvarint(dict_len) + dict_bytes
      uvarint(block_count)
      repeat blocks:
        uvarint(comp_len) + comp_bytes
    """
    if group_size < 1:
        raise ValueError("group_size must be >= 1")

    total_packets = len(packets)

    # Build packet blocks
    blocks: List[List[bytes]] = []
    for i in range(0, total_packets, group_size):
        blocks.append(packets[i:i + group_size])

    # Prepare training samples from block plaintext
    samples: List[bytes] = []
    for b in blocks[:max(1, min(sample_blocks, len(blocks)))]:
        samples.append(_pack_block(b))

    # Train dictionary (or empty if tiny)
    if samples:
        try:
            dict_obj = zstd.train_dictionary(dict_target_size, samples)
            dict_bytes = dict_obj.as_bytes()
        except Exception:
            dict_obj = None
            dict_bytes = b""
    else:
        dict_obj = None
        dict_bytes = b""

    # Compress blocks
    if dict_obj is not None and len(dict_bytes) > 0:
        cctx = zstd.ZstdCompressor(level=zstd_level, dict_data=dict_obj)
    else:
        cctx = zstd.ZstdCompressor(level=zstd_level)

    comp_blocks: List[bytes] = []
    for b in blocks:
        plain = _pack_block(b)
        comp = cctx.compress(plain)
        comp_blocks.append(comp)

    # Pack container
    out = bytearray()
    out += MAGIC
    out += uvarint_encode(group_size)
    out += uvarint_encode(total_packets)

    out += uvarint_encode(len(dict_bytes))
    out += dict_bytes

    out += uvarint_encode(len(comp_blocks))
    for cb in comp_blocks:
        out += uvarint_encode(len(cb))
        out += cb

    meta = ODC2SMeta(
        dict_bytes=len(dict_bytes),
        group_size=group_size,
        block_count=len(comp_blocks),
        total_packets=total_packets,
    )
    return bytes(out), meta


def odc2s_decode_all(blob: bytes) -> List[bytes]:
    """
    Decode all packets from container.
    """
    packets, _meta = odc2s_decode_selected_blocks(blob, block_ids=None)
    return packets


def odc2s_decode_selected_blocks(
    blob: bytes,
    block_ids: Optional[Set[int]] = None,
) -> Tuple[List[bytes], ODC2SMeta]:
    """
    Decode only selected blocks (0-indexed).
    If block_ids is None => decode all blocks.
    """
    if not blob.startswith(MAGIC):
        raise ValueError("bad ODC2S magic")

    off = len(MAGIC)

    group_size, off = uvarint_decode(blob, off)
    total_packets, off = uvarint_decode(blob, off)

    dict_len, off = uvarint_decode(blob, off)
    dict_bytes = blob[off:off + dict_len]
    off += dict_len

    block_count, off = uvarint_decode(blob, off)

    if dict_len > 0:
        dctx = zstd.ZstdDecompressor(dict_data=zstd.ZstdCompressionDict(dict_bytes))
    else:
        dctx = zstd.ZstdDecompressor()

    out_packets: List[bytes] = []

    # Iterate blocks
    for bi in range(block_count):
        clen, off = uvarint_decode(blob, off)
        cb = blob[off:off + clen]
        off += clen

        if block_ids is not None and bi not in block_ids:
            continue

        plain = dctx.decompress(cb)
        out_packets.extend(_unpack_block(plain))

    meta = ODC2SMeta(
        dict_bytes=int(dict_len),
        group_size=int(group_size),
        block_count=int(block_count),
        total_packets=int(total_packets),
    )
    return out_packets, meta


def packet_indices_to_block_ids(packet_indices: Set[int], group_size: int) -> Set[int]:
    """
    Map packet indices -> block ids.
    packet_indices are 0-indexed relative to the encoded packets list.
    """
    out: Set[int] = set()
    for pi in packet_indices:
        out.add(int(pi // group_size))
    return out
