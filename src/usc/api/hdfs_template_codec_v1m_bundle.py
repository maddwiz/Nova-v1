import struct
from dataclasses import dataclass
from typing import List, Tuple

try:
    import zstandard as zstd
except Exception:
    zstd = None

from usc.api.hdfs_template_codec_v1_channels_mask import encode_and_compress_v1m


@dataclass
class EncodedTemplateBundleV1M:
    raw_structured_bytes: int
    compressed_bytes: int
    event_count: int
    unknown_count: int
    channel_count: int
    bundle_bytes: int


MAGIC = b"USC1M"
MAGIC_V2 = b"US1M2"  # V2: zstd-compressed template CSV


def _u32(x: int) -> bytes:
    return struct.pack("<I", int(x))


def bundle_encode_and_compress_v1m(
    events: List[Tuple[int, List[str]]],
    unknown_lines: List[str],
    template_csv_text: str,
    zstd_level: int = 10,
) -> Tuple[bytes, EncodedTemplateBundleV1M]:
    """
    Bundles template bank + V1M payload into a single self-contained blob.

    Layout:
      MAGIC (4 bytes) = b"USC1M"
      zstd_level (u32)
      template_csv_len (u32)
      template_csv_bytes (N)
      payload_len (u32)
      payload_bytes (M)

    payload_bytes is the output of encode_and_compress_v1m().
    """
    payload, meta = encode_and_compress_v1m(
        events=events,
        unknown_lines=unknown_lines,
        zstd_level=zstd_level,
    )

    tpl_raw = template_csv_text.encode("utf-8", errors="replace")
    tpl_bytes = zstd.ZstdCompressor(level=19).compress(tpl_raw) if zstd else tpl_raw

    out = bytearray()
    out += MAGIC_V2
    out += _u32(zstd_level)
    out += _u32(len(tpl_bytes))
    out += tpl_bytes
    out += _u32(len(payload))
    out += payload

    bundle_meta = EncodedTemplateBundleV1M(
        raw_structured_bytes=meta.raw_structured_bytes,
        compressed_bytes=meta.compressed_bytes,
        event_count=meta.event_count,
        unknown_count=meta.unknown_count,
        channel_count=meta.channel_count,
        bundle_bytes=len(out),
    )
    return bytes(out), bundle_meta


def bundle_decode_header(blob: bytes) -> Tuple[int, str, bytes]:
    """
    Returns:
      (zstd_level, template_csv_text, payload_bytes)
    Handles both V1 (USC1M, uncompressed CSV) and V2 (US1M2, zstd-compressed CSV).
    """
    if len(blob) < 16:
        raise ValueError("blob too small")

    magic5 = blob[:5]
    if magic5 == MAGIC_V2:
        is_v2 = True
        off = 5
    elif magic5 == MAGIC:
        is_v2 = False
        off = 5
    else:
        raise ValueError(f"bad bundle magic: {blob[:5]!r}")

    zstd_level = struct.unpack("<I", blob[off:off+4])[0]
    off += 4

    tpl_len = struct.unpack("<I", blob[off:off+4])[0]
    off += 4
    tpl_bytes = blob[off:off+tpl_len]
    off += tpl_len

    payload_len = struct.unpack("<I", blob[off:off+4])[0]
    off += 4
    payload = blob[off:off+payload_len]

    if is_v2 and zstd:
        tpl_text = zstd.ZstdDecompressor().decompress(tpl_bytes).decode("utf-8", errors="replace")
    else:
        tpl_text = tpl_bytes.decode("utf-8", errors="replace")

    return int(zstd_level), tpl_text, payload
