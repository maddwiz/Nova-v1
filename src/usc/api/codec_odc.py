from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import zstandard as zstd

from usc.mem.chunking import chunk_by_lines
from usc.mem.outerstream_zstd import pack_packets, unpack_packets

from usc.mem.stream_proto_canz_v3b import (
    StreamStateV3B,
    build_dict_state_from_chunks as build_v3b,
    encode_dict_packet as dict_v3b,
    apply_dict_packet as apply_v3b,
    encode_data_packet as data_v3b,
)

from usc.mem.zstd_trained_dict import train_dict


MAGIC = b"USC_ODC1"  # 8 bytes, ODC = Outer Dictionary Codec v1


@dataclass
class ODCMeta:
    level: int
    dict_bytes: int
    framed_bytes: int
    compressed_bytes: int
    packets: int


def _u32(x: int) -> bytes:
    return int(x).to_bytes(4, "little", signed=False)


def _read_u32(buf: bytes, off: int) -> Tuple[int, int]:
    return int.from_bytes(buf[off:off + 4], "little", signed=False), off + 4


def _windows(items: List[str], win: int):
    for i in range(0, len(items), win):
        yield items[i:i + win]


def build_v3b_packets_from_text(
    text: str,
    max_lines_per_chunk: int = 60,
    window_chunks: int = 1,
    level: int = 10,
) -> List[bytes]:
    """
    Builds USC v3b packet list: [DICT_PACKET] + [DATA_PACKET, DATA_PACKET, ...]
    """
    chunks = [c.text for c in chunk_by_lines(text, max_lines=max_lines_per_chunk)]

    st_build = StreamStateV3B()
    build_v3b(chunks, state=st_build)
    pkt_dict = dict_v3b(st_build, level=level)

    st_send = StreamStateV3B()
    apply_v3b(pkt_dict, state=st_send)

    packets: List[bytes] = [pkt_dict]
    for w in _windows(chunks, window_chunks):
        packets.append(data_v3b(w, st_send, level=level))

    return packets


def odc_encode_packets(
    packets: List[bytes],
    level: int = 10,
    dict_target_size: int = 8192,
    sample_chunk_size: int = 1024,
) -> Tuple[bytes, ODCMeta]:
    """
    ODC format:
      [MAGIC 8B]
      [level u32]
      [dict_len u32]
      [dict_bytes...]
      [framed_len u32]
      [comp_len u32]
      [zstd(comp_with_dict(framed))...]

    Dictionary is embedded so decode is always possible from blob alone.
    Falls back to plain zstd (dict_len=0) if dict training fails.
    """
    framed = pack_packets(packets)

    # train dict from framed stream; fall back to plain zstd on failure
    samples = [framed[i:i + sample_chunk_size] for i in range(0, len(framed), sample_chunk_size)]
    dict_bytes = b""
    try:
        bundle = train_dict(samples, dict_size=dict_target_size)
        dict_bytes = bundle.dict_bytes
        cctx = zstd.ZstdCompressor(level=level, dict_data=bundle.cdict)
    except Exception:
        cctx = zstd.ZstdCompressor(level=level)

    comp = cctx.compress(framed)

    out = bytearray()
    out += MAGIC
    out += _u32(level)
    out += _u32(len(dict_bytes))
    out += dict_bytes
    out += _u32(len(framed))
    out += _u32(len(comp))
    out += comp

    meta = ODCMeta(
        level=level,
        dict_bytes=len(dict_bytes),
        framed_bytes=len(framed),
        compressed_bytes=len(comp),
        packets=len(packets),
    )
    return bytes(out), meta


def odc_decode_to_packets(blob: bytes) -> List[bytes]:
    """
    Decodes ODC blob back to original packet list.
    """
    if len(blob) < 8 + 4 + 4 + 4 + 4:
        raise ValueError("odc: blob too small")
    if blob[:8] != MAGIC:
        raise ValueError("odc: bad magic")

    off = 8
    _level, off = _read_u32(blob, off)
    dict_len, off = _read_u32(blob, off)
    dict_bytes = blob[off:off + dict_len]
    off += dict_len

    framed_len, off = _read_u32(blob, off)
    comp_len, off = _read_u32(blob, off)
    comp = blob[off:off + comp_len]

    if len(dict_bytes) > 0:
        ddict = zstd.ZstdCompressionDict(dict_bytes)
        dctx = zstd.ZstdDecompressor(dict_data=ddict)
    else:
        dctx = zstd.ZstdDecompressor()
    framed = dctx.decompress(comp)

    if len(framed) != framed_len:
        raise ValueError("odc: framed_len mismatch")

    return unpack_packets(framed)


def odc_encode_text(
    text: str,
    max_lines_per_chunk: int = 60,
    window_chunks: int = 1,
    level: int = 10,
) -> Tuple[bytes, ODCMeta]:
    """
    Convenience: text -> USC packets -> ODC blob
    """
    packets = build_v3b_packets_from_text(
        text,
        max_lines_per_chunk=max_lines_per_chunk,
        window_chunks=window_chunks,
        level=level,
    )
    return odc_encode_packets(packets, level=level)
