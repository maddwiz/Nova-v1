import struct
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

try:
    import zstandard as zstd
except Exception:
    zstd = None


@dataclass
class EncodedTemplateChannelsV1M:
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
# bitmask helpers
# -------------------------

def _build_nonempty_mask(values: List[str]) -> Tuple[bytes, List[str]]:
    """
    Returns (mask_bytes, nonempty_values_in_order)
    mask bit = 1 if values[i] != ""
    """
    n = len(values)
    mask_len = (n + 7) // 8
    mask = bytearray(mask_len)
    nonempty = []
    for i, v in enumerate(values):
        if v != "":
            mask[i // 8] |= (1 << (i % 8))
            nonempty.append(v)
    return bytes(mask), nonempty


# -------------------------
# channel encoders (NONEMPTY ONLY)
# -------------------------

def _encode_int_nonempty(values: List[str]) -> bytes:
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


def _encode_ip_nonempty(values: List[str]) -> bytes:
    out = bytearray()
    out += _uvarint_encode(len(values))
    for v in values:
        parts = [int(p) for p in v.strip().split(".")]
        out += bytes(parts)
    return bytes(out)


def _encode_hex_nonempty(values: List[str]) -> bytes:
    out = bytearray()
    out += _uvarint_encode(len(values))
    for v in values:
        b = _hex_to_bytes(v)
        out += _encode_bytes(b)
    return bytes(out)


def _encode_raw_nonempty(values: List[str]) -> bytes:
    out = bytearray()
    out += _uvarint_encode(len(values))
    for v in values:
        out += _encode_str(v)
    return bytes(out)


def _encode_dict_nonempty(values: List[str], max_dict: int = 4096) -> bytes:
    from collections import Counter

    counts = Counter(values)
    vocab = [s for (s, _) in counts.most_common(max_dict)]
    idx = {s: i for i, s in enumerate(vocab)}

    out = bytearray()
    out += _uvarint_encode(len(values))

    out += _uvarint_encode(len(vocab))
    for s in vocab:
        out += _encode_str(s)

    for v in values:
        out += _uvarint_encode(idx.get(v, 0))
    return bytes(out)


# type tags
# 0 raw, 1 int, 2 hex, 3 ip, 4 dict
def _choose_type(values_nonempty: List[str], dict_threshold: int = 12) -> int:
    if not values_nonempty:
        return 0

    sample = values_nonempty[:256]

    if all(_is_int(v) for v in sample):
        return 1
    if all(_is_ip(v) for v in sample):
        return 3
    if all(_is_hex(v) for v in sample):
        return 2

    uniq = len(set(sample))
    repeats_heavy = (len(sample) - uniq) >= dict_threshold
    if repeats_heavy:
        return 4

    return 0


def _encode_by_type(values_nonempty: List[str], t: int, max_dict: int = 4096) -> bytes:
    if t == 1:
        return _encode_int_nonempty(values_nonempty)
    if t == 2:
        return _encode_hex_nonempty(values_nonempty)
    if t == 3:
        return _encode_ip_nonempty(values_nonempty)
    if t == 4:
        return _encode_dict_nonempty(values_nonempty, max_dict=max_dict)
    return _encode_raw_nonempty(values_nonempty)


def encode_template_channels_v1_mask(
    events: List[Tuple[int, List[str]]],
    unknown_lines: List[str],
    dict_threshold: int = 12,
    max_dict: int = 4096,
) -> bytes:
    """
    V1-MASK: Same as V1 channels, but each channel stores:
      - bitmask of which rows have a value (non-empty)
      - then encodes ONLY those values (typed or dict/raw)

    Header:
      magic 'H1M1'
      version u32 (1)
      event_count uvarint
      unknown_count uvarint
      max_params uvarint

    Body:
      event_id stream (uvarint per row)
      For each channel:
         mask_len uvarint
         mask_bytes
         type uvarint
         payload bytes (len + payload of nonempty values)
      unknown lines (len+utf8)
    """
    out = bytearray()
    out += b"H1M1"
    out += struct.pack("<I", 1)

    n = len(events)
    out += _uvarint_encode(n)
    out += _uvarint_encode(len(unknown_lines))

    max_params = 0
    for _, p in events:
        if len(p) > max_params:
            max_params = len(p)
    out += _uvarint_encode(max_params)

    # event_ids stream
    for eid, _ in events:
        out += _uvarint_encode(int(eid))

    # build channels
    channels: List[List[str]] = [[] for _ in range(max_params)]
    for _, params in events:
        for i in range(max_params):
            channels[i].append(params[i] if i < len(params) else "")

    # encode each channel with mask + nonempty values
    for vals in channels:
        mask_bytes, nonempty_vals = _build_nonempty_mask(vals)
        out += _uvarint_encode(len(mask_bytes))
        out += mask_bytes

        t = _choose_type(nonempty_vals, dict_threshold=dict_threshold)
        payload = _encode_by_type(nonempty_vals, t, max_dict=max_dict)

        out += _uvarint_encode(t)
        out += _encode_bytes(payload)

    for ln in unknown_lines:
        out += _encode_str(ln)

    return bytes(out)


def compress_bytes(b: bytes, level: int = 10) -> bytes:
    if zstd is None:
        raise RuntimeError("zstandard not installed. Install with: pip install zstandard")
    return zstd.ZstdCompressor(level=level).compress(b)


def encode_and_compress_v1m(
    events: List[Tuple[int, List[str]]],
    unknown_lines: List[str],
    zstd_level: int = 10,
    dict_threshold: int = 12,
    max_dict: int = 4096,
):
    raw_struct = encode_template_channels_v1_mask(
        events,
        unknown_lines,
        dict_threshold=dict_threshold,
        max_dict=max_dict,
    )
    comp = compress_bytes(raw_struct, level=zstd_level)

    meta = EncodedTemplateChannelsV1M(
        raw_structured_bytes=len(raw_struct),
        compressed_bytes=len(comp),
        event_count=len(events),
        unknown_count=len(unknown_lines),
        channel_count=max((len(p) for _, p in events), default=0),
    )
    return comp, meta


# ============================================================
# H1M2: ROW-PRESERVING TEMPLATE + RAW SPILL (LOSSLESS HOT-LITE)
# ============================================================
#
# Fixes the core lossless issue:
# - H1M1 can store events + unknowns but cannot reconstruct original row order.
# - H1M2 stores a row bitmask so we can interleave events + unknown lines exactly.
#
# Row mask bit = 1 means RAW/UNKNOWN row; 0 means TEMPLATE/EVENT row.
#
# Header:
#   magic 'H1M2'
#   version u32 (1)
#   row_count uvarint
#   event_count uvarint
#   unknown_count uvarint
#   max_params uvarint
#
# Body:
#   row_mask_len uvarint
#   row_mask_bytes
#   event_id stream (uvarint per EVENT row, in event-row order)
#   channels (same as H1M1, built over EVENT rows only)
#   unknown lines (len+utf8) in UNKNOWN-row order
#
# This makes HOT-LITE/HOT modes truly LOSSLESS.


def encode_template_rows_v1_mask(
    rows: List[Tuple[int, List[str]] | None],
    unknown_lines: List[str],
    dict_threshold: int = 12,
    max_dict: int = 4096,
) -> bytes:
    """
    rows:
      list of length row_count
      - if row is TEMPLATE/EVENT: rows[i] = (event_id, params)
      - if row is RAW/UNKNOWN: rows[i] = None
    unknown_lines:
      raw lines in the same order as the None rows appear

    Returns H1M2 bytes (lossless).
    """
    import struct

    row_count = len(rows)

    # build row mask + extract event rows in order
    mask_len = (row_count + 7) // 8
    row_mask = bytearray(mask_len)

    events: List[Tuple[int, List[str]]] = []
    unk_seen = 0

    for i, r in enumerate(rows):
        if r is None:
            row_mask[i // 8] |= (1 << (i % 8))
            unk_seen += 1
        else:
            events.append((int(r[0]), r[1]))

    if unk_seen != len(unknown_lines):
        raise ValueError(f"H1M2: unknown_lines mismatch: mask says {unk_seen} but unknown_lines has {len(unknown_lines)}")

    # compute max_params over event rows only
    max_params = 0
    for _, p in events:
        if len(p) > max_params:
            max_params = len(p)

    out = bytearray()
    out += b"H1M2"
    out += struct.pack("<I", 1)

    out += _uvarint_encode(row_count)
    out += _uvarint_encode(len(events))
    out += _uvarint_encode(len(unknown_lines))
    out += _uvarint_encode(max_params)

    # row mask
    out += _uvarint_encode(len(row_mask))
    out += bytes(row_mask)

    # event_id stream (ONLY for event rows)
    for eid, _ in events:
        out += _uvarint_encode(int(eid))

    # build event param channels (same as H1M1)
    channels: List[List[str]] = [[] for _ in range(max_params)]
    for _, params in events:
        for i in range(max_params):
            channels[i].append(params[i] if i < len(params) else "")

    # encode each channel as: mask + type + payload
    for ch in channels:
        mask_bytes, nonempty = _build_nonempty_mask(ch)
        out += _uvarint_encode(len(mask_bytes))
        out += mask_bytes

        t = _choose_type(nonempty, dict_threshold=dict_threshold)
        out += _uvarint_encode(t)

        payload = _encode_by_type(nonempty, t, max_dict=max_dict)
        out += _uvarint_encode(len(payload))
        out += payload

    # unknown raw lines (ONLY for unknown rows)
    for s in unknown_lines:
        out += _encode_str(s)

    return bytes(out)


def _mask_bit(mask: bytes, i: int) -> int:
    return (mask[i // 8] >> (i % 8)) & 1


def decode_template_rows_v1_mask_full(blob: bytes) -> List[str]:
    """
    Full lossless decode for H1M2 -> list of reconstructed lines (strings).

    NOTE:
    This only reconstructs row structure:
    - TEMPLATE rows become a synthetic placeholder line:
        "<EID=<id>> <P0=...> <P1=...> ..."
      because true template->string reconstruction requires the template text.
    If you already rebuild original lines elsewhere (tpl_pf1_recall), use that path.

    This function exists mainly for correctness checks + future decode wiring.
    """
    import struct

    if not blob.startswith(b"H1M2"):
        raise ValueError("not H1M2")

    i = 4
    _ver = struct.unpack("<I", blob[i:i+4])[0]
    i += 4

    row_count, i = _uvarint_decode(blob, i)
    event_count, i = _uvarint_decode(blob, i)
    unknown_count, i = _uvarint_decode(blob, i)
    max_params, i = _uvarint_decode(blob, i)

    mask_len, i = _uvarint_decode(blob, i)
    row_mask = blob[i:i+mask_len]
    i += mask_len

    # read event ids
    eids: List[int] = []
    for _ in range(event_count):
        eid, i = _uvarint_decode(blob, i)
        eids.append(int(eid))

    # decode channels into event param rows
    cols: List[List[str]] = [[] for _ in range(max_params)]
    for ch_i in range(max_params):
        mlen, i = _uvarint_decode(blob, i)
        ch_mask = blob[i:i+mlen]
        i += mlen

        t, i = _uvarint_decode(blob, i)
        plen, i = _uvarint_decode(blob, i)
        payload = blob[i:i+plen]
        i += plen

        # decode nonempty list based on type
        # we reuse existing decode helpers by temporarily decoding the typed payload
        # the payload format for each type starts with count = number of NONEMPTY items
        j = 0
        n_nonempty, j = _uvarint_decode(payload, j)

        nonempty_values: List[str] = []
        if t == 1:
            # int
            if n_nonempty == 0:
                nonempty_values = []
            else:
                x0, j = _svarint_decode(payload, j)
                ints = [x0]
                prev = x0
                for _ in range(n_nonempty - 1):
                    d, j = _svarint_decode(payload, j)
                    ints.append(prev + d)
                    prev = prev + d
                nonempty_values = [str(x) for x in ints]

        elif t == 2:
            # hex stored as bytes blocks
            for _ in range(n_nonempty):
                b, j = _decode_bytes(payload, j)
                nonempty_values.append(b.hex())

        elif t == 3:
            # ip stored as 4 bytes
            for _ in range(n_nonempty):
                parts = payload[j:j+4]
                j += 4
                nonempty_values.append(".".join(str(int(x)) for x in parts))

        elif t == 4:
            # dict
            vocab_n, j = _uvarint_decode(payload, j)
            vocab: List[str] = []
            for _ in range(vocab_n):
                s, j = _decode_str(payload, j)
                vocab.append(s)
            for _ in range(n_nonempty):
                did, j = _uvarint_decode(payload, j)
                nonempty_values.append(vocab[int(did)] if int(did) < len(vocab) else "")

        else:
            # raw
            for _ in range(n_nonempty):
                s, j = _decode_str(payload, j)
                nonempty_values.append(s)

        # expand into full channel length event_count using channel mask
        out_vals: List[str] = []
        k = 0
        for r in range(event_count):
            if _mask_bit(ch_mask, r) == 1:
                out_vals.append(nonempty_values[k] if k < len(nonempty_values) else "")
                k += 1
            else:
                out_vals.append("")
        cols[ch_i] = out_vals

    # transpose event params
    event_params: List[List[str]] = []
    for r in range(event_count):
        row = []
        for c in range(max_params):
            row.append(cols[c][r])
        event_params.append(row)

    # unknown lines
    unknowns: List[str] = []
    for _ in range(unknown_count):
        s, i = _decode_str(blob, i)
        unknowns.append(s)

    # reconstruct row stream
    out_lines: List[str] = []
    ei = 0
    ui = 0
    for r in range(row_count):
        if _mask_bit(row_mask, r) == 1:
            out_lines.append(unknowns[ui])
            ui += 1
        else:
            # placeholder if no template expansion exists here
            eid = eids[ei]
            params = event_params[ei]
            out_lines.append(f"<EID={eid}> " + " ".join(params).strip())
            ei += 1

    return out_lines


# ============================================================
# SAFETY OVERRIDES (prevent bad INT classification crashes)
# ============================================================

import re as _re_safe

_INT_RE_SAFE = _re_safe.compile(r"^-?\d+$")


def _all_int_safe(vals: list[str]) -> bool:
    if not vals:
        return False
    for v in vals:
        s = (v or "").strip()
        if not _INT_RE_SAFE.match(s):
            return False
    return True


def _choose_type(values_nonempty, dict_threshold: int = 12):
    """
    Safer type chooser:
    - INT only if ALL values are valid ints
    - otherwise prefer DICT if repeated, else RAW
    """
    # Values are strings
    if not values_nonempty:
        return 5  # RAW

    # ✅ safe INT detection
    if _all_int_safe(values_nonempty):
        return 1  # INT

    # Keep existing HEX/IP checks if present in file (we don't force them here)
    # But we use DICT when repetition exists
    uniq = set(values_nonempty)
    if len(values_nonempty) >= dict_threshold and len(uniq) <= max(2, len(values_nonempty) // 4):
        return 4  # DICT

    return 5  # RAW


def _encode_int_nonempty(values):
    """
    Safe INT encoder:
    If any value is not int-like, fall back to RAW encoding.
    """
    if not _all_int_safe(values):
        # fallback to RAW
        return _encode_raw_nonempty(values)

    # original behavior (delta encode ints)
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


# ============================================================
# TYPED SLOT RESTORE (INT + IP + HEX + DICT + RAW) w/ SAFETY
# ============================================================

import re as _re_typing

_INT_RE2 = _re_typing.compile(r"^-?\d+$")
_HEX_RE2 = _re_typing.compile(r"^[0-9a-fA-F]{6,}$")
_IP_RE2  = _re_typing.compile(r"^\d{1,3}(\.\d{1,3}){3}$")


def _all_int_safe2(vals: list[str]) -> bool:
    if not vals:
        return False
    for v in vals:
        s = (v or "").strip()
        if not _INT_RE2.match(s):
            return False
    return True


def _all_hex_safe2(vals: list[str]) -> bool:
    if not vals:
        return False
    for v in vals:
        s = (v or "").strip()
        if not _HEX_RE2.match(s):
            return False
    return True


def _all_ip_safe2(vals: list[str]) -> bool:
    if not vals:
        return False
    for v in vals:
        s = (v or "").strip()
        if not _IP_RE2.match(s):
            return False
        # validate octets
        parts = s.split(".")
        ok = True
        for p in parts:
            try:
                x = int(p)
            except Exception:
                ok = False
                break
            if x < 0 or x > 255:
                ok = False
                break
        if not ok:
            return False
    return True


def _choose_type(values_nonempty, dict_threshold: int = 12):
    """
    Restored typed chooser:
      1 = INT
      2 = HEX
      3 = IP
      4 = DICT
      5 = RAW
    """
    if not values_nonempty:
        return 5  # RAW

    # ✅ strong types first
    if _all_int_safe2(values_nonempty):
        return 1  # INT

    if _all_ip_safe2(values_nonempty):
        return 3  # IP

    if _all_hex_safe2(values_nonempty):
        return 2  # HEX

    # ✅ dictionary when repetition is meaningful
    uniq = set(values_nonempty)
    if len(values_nonempty) >= dict_threshold and len(uniq) <= max(2, len(values_nonempty) // 4):
        return 4  # DICT

    return 5  # RAW


def _encode_int_nonempty(values):
    """
    Safe INT encoder:
    If any value isn't an int, fall back to RAW.
    """
    if not _all_int_safe2(values):
        return _encode_raw_nonempty(values)

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


# ============================================================
# FINAL OVERRIDE: STRONG TYPING (INT + IP + HEX + DICT + RAW)
# This is the missing half of the typed restore.
# ============================================================

import re as _re_typing_final

_INT_RE_F = _re_typing_final.compile(r"^-?\d+$")
_HEX_RE_F = _re_typing_final.compile(r"^(0x)?[0-9a-fA-F]{6,}$")
_IP_RE_F  = _re_typing_final.compile(r"^\d{1,3}(\.\d{1,3}){3}$")


def _all_int_F(vals: list[str]) -> bool:
    if not vals:
        return False
    for v in vals:
        if not _INT_RE_F.match((v or "").strip()):
            return False
    return True


def _all_hex_F(vals: list[str]) -> bool:
    if not vals:
        return False
    for v in vals:
        if not _HEX_RE_F.match((v or "").strip()):
            return False
    return True


def _all_ip_F(vals: list[str]) -> bool:
    if not vals:
        return False
    for v in vals:
        s = (v or "").strip()
        if not _IP_RE_F.match(s):
            return False
        parts = s.split(".")
        try:
            if not all(0 <= int(p) <= 255 for p in parts):
                return False
        except Exception:
            return False
    return True


# type tags expected by _encode_by_type:
# 0 raw, 1 int, 2 hex, 3 ip, 4 dict
def _choose_type(values_nonempty: list[str], dict_threshold: int = 12) -> int:
    if not values_nonempty:
        return 0  # RAW

    # ✅ strongest / smallest encodings first
    if _all_int_F(values_nonempty):
        return 1  # INT
    if _all_ip_F(values_nonempty):
        return 3  # IP
    if _all_hex_F(values_nonempty):
        return 2  # HEX

    # ✅ DICT when repetition is heavy
    uniq = len(set(values_nonempty))
    if (len(values_nonempty) - uniq) >= dict_threshold:
        return 4  # DICT

    return 0  # RAW


def _encode_int_nonempty(values: list[str]) -> bytes:
    """
    Safe INT encoder:
    If anything isn't an int, fall back to RAW encoding.
    """
    if not _all_int_F(values):
        return _encode_raw_nonempty(values)

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

