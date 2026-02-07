import struct
from dataclasses import dataclass
from typing import List, Tuple

try:
    import zstandard as zstd
except Exception:
    zstd = None


@dataclass
class EncodedTemplateStream:
    raw_structured_bytes: int
    compressed_bytes: int
    event_count: int
    unknown_count: int


def _uvarint_encode(x: int) -> bytes:
    # unsigned varint
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


def _encode_str(s: str) -> bytes:
    b = s.encode("utf-8", errors="replace")
    return _uvarint_encode(len(b)) + b


def encode_template_stream(events: List[Tuple[int, List[str]]], unknown_lines: List[str]) -> bytes:
    """
    Very simple lossless structured format:

    Header:
      magic 'HDFT' (4 bytes)
      version (u32)
      event_count (uvarint)
      unknown_count (uvarint)

    Body:
      For each event:
        event_id (uvarint)
        nparams (uvarint)
        param_0 ... param_{n-1} as (len + utf8 bytes)

      Unknown lines:
        each unknown as (len + utf8 bytes)

    This is NOT the final best format — it’s the correct foundation.
    """
    out = bytearray()
    out += b"HDFT"
    out += struct.pack("<I", 0)  # version 0
    out += _uvarint_encode(len(events))
    out += _uvarint_encode(len(unknown_lines))

    for (eid, params) in events:
        out += _uvarint_encode(int(eid))
        out += _uvarint_encode(len(params))
        for p in params:
            out += _encode_str(p)

    for ln in unknown_lines:
        out += _encode_str(ln)

    return bytes(out)


def compress_bytes(b: bytes, level: int = 10) -> bytes:
    if zstd is None:
        raise RuntimeError("zstandard not installed. Install with: pip install zstandard")
    return zstd.ZstdCompressor(level=level).compress(b)


def encode_and_compress(events: List[Tuple[int, List[str]]], unknown_lines: List[str], zstd_level: int = 10) -> Tuple[bytes, EncodedTemplateStream]:
    raw_struct = encode_template_stream(events, unknown_lines)
    comp = compress_bytes(raw_struct, level=zstd_level)
    meta = EncodedTemplateStream(
        raw_structured_bytes=len(raw_struct),
        compressed_bytes=len(comp),
        event_count=len(events),
        unknown_count=len(unknown_lines),
    )
    return comp, meta
