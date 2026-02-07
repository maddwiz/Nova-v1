from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
import struct

try:
    import zstandard as zstd
except Exception:
    zstd = None

from usc.api.hdfs_template_codec_h1m2_rowmask import encode_h1m2_rowmask_blob

MAGIC = b"TPF3"
VERSION = 1


def _uvarint_encode(x: int) -> bytes:
    out = bytearray()
    n = int(x)
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            break
    return bytes(out)


def _bytes_encode(b: bytes) -> bytes:
    return _uvarint_encode(len(b)) + b


def _zstd_compress(buf: bytes, level: int = 10) -> bytes:
    if zstd is None:
        raise RuntimeError("zstandard missing (pip install zstandard)")
    return zstd.ZstdCompressor(level=level).compress(buf)


@dataclass
class PF3Meta:
    rows: int
    chunks: int
    unknown_rows: int
    bytes_raw_chunks: int
    bytes_zstd_chunks: int


def build_tpl_pf3_blob_h1m2(
    rows: List[Optional[Tuple[int, List[str]]]],
    unknown_lines: List[str],
    tpl_text: str,
    packet_events: int = 25,
    zstd_level: int = 10,
) -> tuple[bytes, PF3Meta]:
    """
    PF3(H1M2):
    - preserves original row order using rowmask
    - supports unknown rows inline (lossless)
    - chunked for streaming / query future expansion

    Output format:
      MAGIC(4) + VERSION(uvarint)
      tpl_text_len(uvarint) + tpl_text_bytes
      packet_events(uvarint)
      total_rows(uvarint)
      chunk_count(uvarint)
      for each chunk:
         flags(u8)  (bit0=zstd)
         raw_len(uvarint)
         payload_len(uvarint)
         payload_bytes
    """
    if packet_events <= 0:
        packet_events = 25

    tpl_bytes = (tpl_text or "").encode("utf-8", errors="ignore")

    out = bytearray()
    out += MAGIC
    out += _uvarint_encode(VERSION)
    out += _bytes_encode(tpl_bytes)
    out += _uvarint_encode(packet_events)
    out += _uvarint_encode(len(rows))

    # chunking
    chunks: List[bytes] = []
    uidx = 0
    unknown_rows_total = 0

    for i in range(0, len(rows), packet_events):
        chunk_rows = rows[i : i + packet_events]

        # slice the correct unknown_lines for this chunk
        need = sum(1 for r in chunk_rows if r is None)
        unknown_rows_total += need
        chunk_unknown = unknown_lines[uidx : uidx + need]
        uidx += need

        blob = encode_h1m2_rowmask_blob(chunk_rows, chunk_unknown)
        chunks.append(blob)

    out += _uvarint_encode(len(chunks))

    bytes_raw = 0
    bytes_z = 0

    for blob in chunks:
        bytes_raw += len(blob)

        # always zstd-compress chunk (fast + strong)
        zpayload = _zstd_compress(blob, level=zstd_level)
        bytes_z += len(zpayload)

        flags = 1  # bit0 = zstd
        out += struct.pack("<B", flags)
        out += _uvarint_encode(len(blob))
        out += _uvarint_encode(len(zpayload))
        out += zpayload

    meta = PF3Meta(
        rows=len(rows),
        chunks=len(chunks),
        unknown_rows=unknown_rows_total,
        bytes_raw_chunks=bytes_raw,
        bytes_zstd_chunks=bytes_z,
    )
    return bytes(out), meta
