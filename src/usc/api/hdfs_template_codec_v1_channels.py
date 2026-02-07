import struct
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

try:
    import zstandard as zstd
except Exception:
    zstd = None


@dataclass
class EncodedTemplateChannelsV1:
    raw_structured_bytes: int
    compressed_bytes: int
    event_count: int
    unknown_count: int
    channel_count: int


# -------------------------
# basic encoders
# -------------------------

def _uvarint_encode(x: int) -> bytes:
    out = bytearray()
    x = int(x)
    while True:
        b = x & 0x7F
        x >>= 7
        if x:
            out.append(b | 0x80)
        else:
            out.append(b)
            break
    return bytes(out)


def _zigzag_encode(n: int) -> int:
    # signed -> unsigned
    return (n << 1) ^ (n >> 63)


def _svarint_encode(n: int) -> bytes:
    return _uvarint_encode(_zigzag_encode(int(n)))


def _encode_bytes(b: bytes) -> bytes:
    return _uvarint_encode(len(b)) + b


def _encode_str(s: str) -> bytes:
    b = s.encode("utf-8", errors="replace")
    return _encode_bytes(b)


# -------------------------
# type detection
# -------------------------

_INT_RE = re.compile(r"^[+-]?\d+$")
_HEX_RE = re.compile(r"^(0x)?[0-9a-fA-F]{8,}$")
_IP_RE  = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")


def _is_int(s: str) -> bool:
    return bool(_INT_RE.match(s.strip()))


def _is_hex(s: str) -> bool:
    s = s.strip()
    return bool(_HEX_RE.match(s))


def _is_ip(s: str) -> bool:
    s = s.strip()
    if not _IP_RE.match(s):
        return False
    parts = s.split(".")
    try:
        return all(0 <= int(p) <= 255 for p in parts)
    except Exception:
        return False


def _hex_to_bytes(s: str) -> bytes:
    s = s.strip()
    if s.startswith("0x") or s.startswith("0X"):
        s = s[2:]
    if len(s) % 2 == 1:
        s = "0" + s
    return bytes.fromhex(s)


# -------------------------
# channel encoders
# -------------------------

def _encode_channel_int(values: List[str]) -> bytes:
    """
    Encode ints as delta stream:
      base = first
      then deltas as svarint
    """
    ints = [int(v.strip()) for v in values]
    out = bytearray()
    out += _uvarint_encode(len(ints))
    if not ints:
        return bytes(out)
    out += _svarint_encode(ints[0])
    prev = ints[0]
    for x in ints[1:]:
        out += _svarint_encode(x - prev)
        prev = x
    return bytes(out)


def _encode_channel_ip(values: List[str]) -> bytes:
    """
    Encode IPs as 4 raw bytes each.
    """
    out = bytearray()
    out += _uvarint_encode(len(values))
    for v in values:
        parts = [int(p) for p in v.strip().split(".")]
        out += bytes(parts)
    return bytes(out)


def _encode_channel_hex(values: List[str]) -> bytes:
    """
    Encode hex-ish tokens as length + bytes (compact).
    """
    out = bytearray()
    out += _uvarint_encode(len(values))
    for v in values:
        b = _hex_to_bytes(v)
        out += _encode_bytes(b)
    return bytes(out)


def _encode_channel_dict(values: List[str], max_dict: int = 4096) -> bytes:
    """
    Dictionary encode repeated strings.
      - build vocab of most frequent strings (up to max_dict)
      - encode ids for each value (uvarint)
    """
    from collections import Counter

    counts = Counter(values)
    # keep most common first
    vocab = [s for (s, _) in counts.most_common(max_dict)]
    idx = {s: i for i, s in enumerate(vocab)}

    out = bytearray()
    out += _uvarint_encode(len(values))

    # store dict
    out += _uvarint_encode(len(vocab))
    for s in vocab:
        out += _encode_str(s)

    # store id stream
    for v in values:
        out += _uvarint_encode(idx.get(v, 0))  # fallback to most common bucket
    return bytes(out)


def _encode_channel_raw(values: List[str]) -> bytes:
    """
    Raw strings: len + bytes
    """
    out = bytearray()
    out += _uvarint_encode(len(values))
    for v in values:
        out += _encode_str(v)
    return bytes(out)


# -------------------------
# V1 encoding
# -------------------------

def encode_template_channels_v1(
    events: List[Tuple[int, List[str]]],
    unknown_lines: List[str],
    dict_threshold: int = 12,
    max_dict: int = 4096,
) -> bytes:
    """
    V1 structured format:

    Header:
      magic 'HDFC' (4 bytes)
      version u32 (1)
      event_count uvarint
      unknown_count uvarint
      max_params uvarint

    Body:
      EventID stream:
        event_count uvarint
        first eid uvarint
        deltas uvarint (signed via svarint)

      Param channels:
        for each slot i in [0..max_params-1]:
          chan_type uvarint (0=raw,1=int,2=hex,3=ip,4=dict)
          chan_payload bytes (length + content)

      Unknown lines:
        each unknown as (len+utf8)
    """
    out = bytearray()

    out += b"HDFC"
    out += struct.pack("<I", 1)

    out += _uvarint_encode(len(events))
    out += _uvarint_encode(len(unknown_lines))

    max_params = 0
    for _, params in events:
        if len(params) > max_params:
            max_params = len(params)
    out += _uvarint_encode(max_params)

    # ---- event id stream (delta)
    eids = [int(eid) for (eid, _) in events]
    out += _uvarint_encode(len(eids))
    if eids:
        out += _uvarint_encode(eids[0])
        prev = eids[0]
        for x in eids[1:]:
            out += _svarint_encode(x - prev)
            prev = x

    # ---- build param channels
    channels: List[List[str]] = [[] for _ in range(max_params)]
    for _, params in events:
        for i in range(max_params):
            channels[i].append(params[i] if i < len(params) else "")

    # ---- encode each channel with best-fit type
    # type tags:
    # 0 raw, 1 int, 2 hex, 3 ip, 4 dict
    for i, vals in enumerate(channels):
        sample = vals[:256]

        is_all_int = all((_is_int(v) or v == "") for v in sample)
        is_all_ip  = all((_is_ip(v) or v == "") for v in sample)
        is_all_hex = all((_is_hex(v) or v == "") for v in sample)

        # dict mode if many repeats
        uniq = len(set(sample))
        repeats_heavy = (len(sample) - uniq) >= dict_threshold

        if is_all_int:
            chan_type = 1
            payload = _encode_channel_int([v if v != "" else "0" for v in vals])
        elif is_all_ip:
            chan_type = 3
            payload = _encode_channel_ip([v if v != "" else "0.0.0.0" for v in vals])
        elif is_all_hex:
            chan_type = 2
            payload = _encode_channel_hex([v if v != "" else "00" for v in vals])
        elif repeats_heavy:
            chan_type = 4
            payload = _encode_channel_dict(vals, max_dict=max_dict)
        else:
            chan_type = 0
            payload = _encode_channel_raw(vals)

        out += _uvarint_encode(chan_type)
        out += _encode_bytes(payload)

    # ---- unknown lines
    for ln in unknown_lines:
        out += _encode_str(ln)

    return bytes(out)


def compress_bytes(b: bytes, level: int = 10) -> bytes:
    if zstd is None:
        raise RuntimeError("zstandard not installed. Install with: pip install zstandard")
    return zstd.ZstdCompressor(level=level).compress(b)


def encode_and_compress_v1(
    events: List[Tuple[int, List[str]]],
    unknown_lines: List[str],
    zstd_level: int = 10,
) -> Tuple[bytes, EncodedTemplateChannelsV1]:
    raw_struct = encode_template_channels_v1(events, unknown_lines)
    comp = compress_bytes(raw_struct, level=zstd_level)
    meta = EncodedTemplateChannelsV1(
        raw_structured_bytes=len(raw_struct),
        compressed_bytes=len(comp),
        event_count=len(events),
        unknown_count=len(unknown_lines),
        channel_count=(max((len(p) for _, p in events), default=0)),
    )
    return comp, meta
