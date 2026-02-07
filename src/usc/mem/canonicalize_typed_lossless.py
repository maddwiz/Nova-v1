import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Tuple

from usc.mem.varint import encode_uvarint, decode_uvarint


# Same patterns as before
RE_ISO_TS = re.compile(
    r"\b(\d{4})-(\d{2})-(\d{2})([T ])(\d{2}):(\d{2}):(\d{2})(?:\.(\d+))?(Z)?\b"
)

RE_UUID = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)

RE_LONG_HEX = re.compile(r"\b[0-9a-fA-F]{16,}\b")
RE_LONG_INT = re.compile(r"\b\d{7,}\b")

PLACEHOLDER = "<@>"


# Token types
T_TS = 1
T_UUID = 2
T_HEX = 3
T_INT = 4
T_RAW = 255


@dataclass
class TypedToken:
    ttype: int
    payload: bytes


def _hex_case_bitmap(s: str) -> bytes:
    """
    1 bit per hex char (1=uppercase, 0=lowercase or digit).
    Packed into bytes (little-endian bit order within each byte).
    """
    bits = 0
    out = bytearray()
    bitpos = 0
    for ch in s:
        is_upper = 1 if ("A" <= ch <= "F") else 0
        bits |= (is_upper << bitpos)
        bitpos += 1
        if bitpos == 8:
            out.append(bits & 0xFF)
            bits = 0
            bitpos = 0
    if bitpos != 0:
        out.append(bits & 0xFF)
    return bytes(out)


def _apply_hex_case_bitmap(lower_hex: str, bitmap: bytes) -> str:
    """
    Reapply uppercase bits to a lowercase hex string.
    """
    chars = list(lower_hex)
    bi = 0
    bitpos = 0
    cur = bitmap[0] if bitmap else 0

    for i, ch in enumerate(chars):
        if bitpos == 8:
            bi += 1
            bitpos = 0
            cur = bitmap[bi] if bi < len(bitmap) else 0
        is_upper = (cur >> bitpos) & 1
        bitpos += 1
        if is_upper and ("a" <= ch <= "f"):
            chars[i] = ch.upper()
    return "".join(chars)


def _pack_uuid(u: str) -> bytes:
    """
    UUID string -> 16 bytes raw + 4 bytes case bitmap for 32 hex chars.
    """
    hexchars = u.replace("-", "")
    lower = hexchars.lower()
    raw = bytes.fromhex(lower)
    bitmap = _hex_case_bitmap(hexchars)
    # store bitmap length + bitmap so it's robust
    return raw + encode_uvarint(len(bitmap)) + bitmap


def _unpack_uuid(payload: bytes) -> str:
    """
    16 bytes + (uvarint bitmap_len) + bitmap -> uuid string with original case.
    """
    raw16 = payload[:16]
    off = 16
    blen, off = decode_uvarint(payload, off)
    bitmap = payload[off : off + blen]
    off += blen

    lower = raw16.hex()
    hex_with_case = _apply_hex_case_bitmap(lower, bitmap)

    return (
        hex_with_case[0:8]
        + "-"
        + hex_with_case[8:12]
        + "-"
        + hex_with_case[12:16]
        + "-"
        + hex_with_case[16:20]
        + "-"
        + hex_with_case[20:32]
    )


def _pack_int(s: str) -> bytes:
    """
    Store int value + digit length (lossless even if leading zeros).
    """
    digit_len = len(s)
    val = int(s)
    return encode_uvarint(digit_len) + encode_uvarint(val)


def _unpack_int(payload: bytes) -> str:
    off = 0
    digit_len, off = decode_uvarint(payload, off)
    val, off = decode_uvarint(payload, off)
    s = str(val)
    if len(s) < digit_len:
        s = ("0" * (digit_len - len(s))) + s
    return s


def _pack_hex(s: str) -> bytes:
    """
    Long hex -> bytes + case bitmap
    """
    lower = s.lower()
    raw = bytes.fromhex(lower if len(lower) % 2 == 0 else ("0" + lower))
    bitmap = _hex_case_bitmap(s)
    return encode_uvarint(len(s)) + encode_uvarint(len(raw)) + raw + encode_uvarint(len(bitmap)) + bitmap


def _unpack_hex(payload: bytes) -> str:
    off = 0
    orig_char_len, off = decode_uvarint(payload, off)
    raw_len, off = decode_uvarint(payload, off)
    raw = payload[off : off + raw_len]
    off += raw_len
    blen, off = decode_uvarint(payload, off)
    bitmap = payload[off : off + blen]
    off += blen

    lower = raw.hex()
    # If original was odd-length, drop the first nibble
    if orig_char_len % 2 == 1:
        lower = lower[1:]
    # Apply case
    return _apply_hex_case_bitmap(lower, bitmap)


def _pack_ts(m: re.Match) -> bytes:
    """
    ISO timestamp -> seconds + (frac_digits + frac_value) + style flags
    Preserves:
      - 'T' vs ' '
      - presence of 'Z'
      - fractional digits count + exact fractional value
    """
    year = int(m.group(1))
    mon = int(m.group(2))
    day = int(m.group(3))
    sep = m.group(4)  # 'T' or ' '
    hh = int(m.group(5))
    mm = int(m.group(6))
    ss = int(m.group(7))
    frac = m.group(8) or ""
    zflag = 1 if m.group(9) else 0

    # treat as UTC if Z present, else naive -> assume UTC for packing
    dt = datetime(year, mon, day, hh, mm, ss, tzinfo=timezone.utc)
    epoch = int(dt.timestamp())

    sep_flag = 1 if sep == "T" else 0
    frac_digits = len(frac)
    frac_val = int(frac) if frac_digits > 0 else 0

    # payload: epoch + sep_flag + zflag + frac_digits + frac_val
    return (
        encode_uvarint(epoch)
        + encode_uvarint(sep_flag)
        + encode_uvarint(zflag)
        + encode_uvarint(frac_digits)
        + encode_uvarint(frac_val)
    )


def _unpack_ts(payload: bytes) -> str:
    off = 0
    epoch, off = decode_uvarint(payload, off)
    sep_flag, off = decode_uvarint(payload, off)
    zflag, off = decode_uvarint(payload, off)
    frac_digits, off = decode_uvarint(payload, off)
    frac_val, off = decode_uvarint(payload, off)

    dt = datetime.fromtimestamp(epoch, tz=timezone.utc)
    sep = "T" if sep_flag == 1 else " "
    base = dt.strftime(f"%Y-%m-%d{sep}%H:%M:%S")

    if frac_digits > 0:
        frac_str = str(frac_val).rjust(frac_digits, "0")
        base += "." + frac_str

    if zflag == 1:
        base += "Z"

    return base


def canonicalize_typed_lossless(line: str) -> Tuple[str, List[TypedToken]]:
    """
    Replaces TS/UUID/HEX/INT with PLACEHOLDER while storing typed tokens.
    Fully reversible by reinflate_typed().
    """
    tokens: List[TypedToken] = []
    s = line

    # timestamps first
    def _ts_repl(m: re.Match) -> str:
        tokens.append(TypedToken(T_TS, _pack_ts(m)))
        return PLACEHOLDER

    s = RE_ISO_TS.sub(_ts_repl, s)

    # UUID
    def _uuid_repl(m: re.Match) -> str:
        u = m.group(0)
        tokens.append(TypedToken(T_UUID, _pack_uuid(u)))
        return PLACEHOLDER

    s = RE_UUID.sub(_uuid_repl, s)

    # long hex
    def _hex_repl(m: re.Match) -> str:
        hx = m.group(0)
        tokens.append(TypedToken(T_HEX, _pack_hex(hx)))
        return PLACEHOLDER

    s = RE_LONG_HEX.sub(_hex_repl, s)

    # long int
    def _int_repl(m: re.Match) -> str:
        iv = m.group(0)
        tokens.append(TypedToken(T_INT, _pack_int(iv)))
        return PLACEHOLDER

    s = RE_LONG_INT.sub(_int_repl, s)

    return s, tokens


def reinflate_typed(canon_text: str, tokens: List[TypedToken]) -> str:
    out = canon_text
    for tok in tokens:
        if tok.ttype == T_TS:
            val = _unpack_ts(tok.payload)
        elif tok.ttype == T_UUID:
            val = _unpack_uuid(tok.payload)
        elif tok.ttype == T_HEX:
            val = _unpack_hex(tok.payload)
        elif tok.ttype == T_INT:
            val = _unpack_int(tok.payload)
        else:
            val = tok.payload.decode("utf-8", errors="replace")
        out = out.replace(PLACEHOLDER, val, 1)
    return out
