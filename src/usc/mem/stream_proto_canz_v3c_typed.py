from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any
import re
import struct
import string
from datetime import datetime, timezone

from usc.mem.varint import encode_uvarint, decode_uvarint
from usc.mem.zstd_codec import zstd_compress, zstd_decompress


# v3c = typed values in DATA, tiny DICT
MAGIC_DICT = b"USDICT3C"   # templates only, in order
MAGIC_DATA = b"USDAT3CT"   # typed data stream


# -------------------------
# Patterns (order matters)
# -------------------------
UUID_RE = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)
ISO_TS_RE = re.compile(
    r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?\b"
)
HEX_RE = re.compile(r"\b0x[0-9a-fA-F]+\b")
FLOAT_RE = re.compile(r"(?<![A-Za-z0-9_])-?\d+\.\d+(?:[eE][+-]?\d+)?")
INT_RE = re.compile(r"(?<![A-Za-z0-9_])-?\d+(?!\.\d)")

# key=value small-string slot
# We want to capture prefix AND the short value.
KV_SMALLSTR_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_\-]*=)([A-Za-z]{1,3})(?=[\s\)\]\}\.,;:]|$)")


# -------------------------
# Value opcodes
# -------------------------
# 0  INT_DELTA (zigzag varint)
# 1  INT_FULL  (zigzag varint)  <-- FIXED: always zigzag!
# 2  FLOAT64   (8 bytes)
# 3  UUID16    (16 bytes)
# 4  BYTES     (uvarint len + bytes)
# 5  TIME_MS   (uvarint epoch ms)
# 6  STR_REF   (uvarint id)
# 7  STR_NEW   (string bytes)

OP_INT_DELTA = 0
OP_INT_FULL  = 1
OP_FLOAT64   = 2
OP_UUID16    = 3
OP_BYTES     = 4
OP_TIME_MS   = 5
OP_STR_REF   = 6
OP_STR_NEW   = 7


def _zigzag_encode(n: int) -> int:
    return (n * 2) if n >= 0 else (-n * 2 - 1)


def _zigzag_decode(z: int) -> int:
    return (z // 2) if (z % 2 == 0) else -(z // 2) - 1


def _pack_string(s: str) -> bytes:
    b = s.encode("utf-8")
    return encode_uvarint(len(b)) + b


def _unpack_string(data: bytes, offset: int) -> Tuple[str, int]:
    n, off = decode_uvarint(data, offset)
    b = data[off : off + n]
    off += n
    return b.decode("utf-8"), off


def _count_format_fields(fmt: str) -> int:
    n = 0
    for _lit, field, _spec, _conv in string.Formatter().parse(fmt):
        if field is not None:
            n += 1
    return n


def _bitpack(values: List[int], bits: int) -> bytes:
    out = bytearray()
    acc = 0
    acc_bits = 0
    mask = (1 << bits) - 1

    for v in values:
        v &= mask
        acc = (acc << bits) | v
        acc_bits += bits

        while acc_bits >= 8:
            shift = acc_bits - 8
            out.append((acc >> shift) & 0xFF)
            acc_bits -= 8
            acc &= (1 << acc_bits) - 1 if acc_bits > 0 else 0

    if acc_bits > 0:
        out.append((acc << (8 - acc_bits)) & 0xFF)

    return bytes(out)


def _bitunpack(data: bytes, n: int, bits: int) -> List[int]:
    out: List[int] = []
    acc = 0
    acc_bits = 0
    idx = 0
    mask = (1 << bits) - 1

    for _ in range(n):
        while acc_bits < bits:
            acc = (acc << 8) | data[idx]
            idx += 1
            acc_bits += 8

        shift = acc_bits - bits
        v = (acc >> shift) & mask
        out.append(v)

        acc_bits -= bits
        acc &= (1 << acc_bits) - 1 if acc_bits > 0 else 0

    return out


def _uuid_to_bytes(u: str) -> bytes:
    h = u.replace("-", "")
    return bytes.fromhex(h)


def _parse_iso_to_epoch_ms(s: str) -> int:
    ss = s.replace(" ", "T")
    if ss.endswith("Z"):
        ss = ss[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(ss)
    except ValueError:
        base = re.sub(r"(Z|[+-]\d{2}:\d{2})$", "", ss)
        dt = datetime.fromisoformat(base).replace(tzinfo=timezone.utc)

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


# -------------------------------------------------
# Typed template extraction (lossless)
# -------------------------------------------------
def extract_template_typed(text: str) -> Tuple[str, List[Any]]:
    """
    One-pass left-to-right extraction so placeholder order is correct.

    Returns:
      template: string with {} placeholders
      values: typed values list in order

    Types:
      int, float, bytes, ("uuid", bytes16), ("time", epoch_ms), ("str", "B")
    """
    values: List[Any] = []
    out: List[str] = []
    i = 0

    patterns = [
        ("kvsmall", KV_SMALLSTR_RE),
        ("uuid", UUID_RE),
        ("time", ISO_TS_RE),
        ("hex",  HEX_RE),
        ("float", FLOAT_RE),
        ("int",  INT_RE),
    ]

    while True:
        best = None  # (start, end, kind, matchobj)
        for kind, rx in patterns:
            mm = rx.search(text, i)
            if mm is None:
                continue
            st, en = mm.span()
            if best is None or st < best[0]:
                best = (st, en, kind, mm)

        if best is None:
            out.append(text[i:])
            break

        st, en, kind, mm = best
        out.append(text[i:st])

        if kind == "kvsmall":
            prefix = mm.group(1)
            sval = mm.group(2)
            out.append(prefix)
            out.append("{}")
            values.append(("str", sval))
        else:
            out.append("{}")
            s = mm.group(0)
            if kind == "int":
                values.append(int(s))
            elif kind == "float":
                values.append(float(s))
            elif kind == "hex":
                hexstr = s[2:]
                if len(hexstr) % 2 == 1:
                    hexstr = "0" + hexstr
                values.append(bytes.fromhex(hexstr))
            elif kind == "uuid":
                values.append(("uuid", _uuid_to_bytes(s)))
            elif kind == "time":
                values.append(("time", _parse_iso_to_epoch_ms(s)))
            else:
                values.append(("str", s))

        i = en

    return "".join(out), values


# -------------------------------------------------
# Stream state
# -------------------------------------------------
@dataclass
class StreamStateV3C:
    templates: List[str] = field(default_factory=list)
    temp_index: Dict[str, int] = field(default_factory=dict)
    arity_by_tid: Dict[int, int] = field(default_factory=dict)
    mtf: List[int] = field(default_factory=list)

    seen_tid: Dict[int, bool] = field(default_factory=dict)
    prev_vals_by_tid: Dict[int, List[Any]] = field(default_factory=dict)

    str_to_id: Dict[str, int] = field(default_factory=dict)
    id_to_str: List[str] = field(default_factory=list)


def build_dict_state_from_chunks(chunks: List[str], state: StreamStateV3C | None = None) -> StreamStateV3C:
    if state is None:
        state = StreamStateV3C()

    for ch in chunks:
        t, vals = extract_template_typed(ch)
        if t not in state.temp_index:
            tid = len(state.templates)
            state.temp_index[t] = tid
            state.templates.append(t)
            state.arity_by_tid[tid] = len(vals)
            state.mtf.append(tid)

    return state


# -------------------------------------------------
# DICT packet (tiny)
# -------------------------------------------------
def encode_dict_packet(state: StreamStateV3C, level: int = 10) -> bytes:
    out = bytearray()
    out += MAGIC_DICT
    out += encode_uvarint(len(state.templates))
    for t in state.templates:
        out += _pack_string(t)
    return zstd_compress(bytes(out), level=level)


def apply_dict_packet(packet_bytes: bytes, state: StreamStateV3C | None = None) -> StreamStateV3C:
    if state is None:
        state = StreamStateV3C()

    raw = zstd_decompress(packet_bytes)
    if not raw.startswith(MAGIC_DICT):
        raise ValueError("Not a 3C DICT packet")

    off = len(MAGIC_DICT)
    ntemps, off = decode_uvarint(raw, off)

    for tid in range(ntemps):
        t, off = _unpack_string(raw, off)
        state.templates.append(t)
        state.temp_index[t] = tid
        state.arity_by_tid[tid] = _count_format_fields(t)
        state.mtf.append(tid)

    return state


# -------------------------------------------------
# Typed values encode/decode
# -------------------------------------------------
def _emit_str(state: StreamStateV3C, s: str, out: bytearray):
    if s in state.str_to_id:
        out += encode_uvarint(OP_STR_REF)
        out += encode_uvarint(state.str_to_id[s])
    else:
        sid = len(state.id_to_str)
        state.str_to_id[s] = sid
        state.id_to_str.append(s)
        out += encode_uvarint(OP_STR_NEW)
        out += _pack_string(s)


def _emit_value(state: StreamStateV3C, prev: Any | None, v: Any, out: bytearray):
    # v can be int, float, bytes, ("uuid", b16), ("time", ms), ("str", s)
    if isinstance(v, int):
        if isinstance(prev, int):
            d = v - prev
            out += encode_uvarint(OP_INT_DELTA)
            out += encode_uvarint(_zigzag_encode(d))
        else:
            # ✅ FIX: ALWAYS zigzag full ints (no guessing)
            out += encode_uvarint(OP_INT_FULL)
            out += encode_uvarint(_zigzag_encode(v))

    elif isinstance(v, float):
        out += encode_uvarint(OP_FLOAT64)
        out += struct.pack(">d", v)

    elif isinstance(v, bytes):
        out += encode_uvarint(OP_BYTES)
        out += encode_uvarint(len(v))
        out += v

    elif isinstance(v, tuple) and len(v) == 2 and v[0] == "uuid":
        out += encode_uvarint(OP_UUID16)
        out += v[1]

    elif isinstance(v, tuple) and len(v) == 2 and v[0] == "time":
        out += encode_uvarint(OP_TIME_MS)
        out += encode_uvarint(int(v[1]))

    elif isinstance(v, tuple) and len(v) == 2 and v[0] == "str":
        _emit_str(state, v[1], out)

    else:
        _emit_str(state, str(v), out)


def _read_value(state: StreamStateV3C, raw: bytes, off: int, prev: Any | None) -> Tuple[Any, int]:
    op, off = decode_uvarint(raw, off)

    if op == OP_INT_DELTA:
        z, off = decode_uvarint(raw, off)
        d = _zigzag_decode(z)
        base = prev if isinstance(prev, int) else 0
        return int(base + d), off

    if op == OP_INT_FULL:
        x, off = decode_uvarint(raw, off)
        # ✅ FIX: always zigzag decode full ints
        return int(_zigzag_decode(x)), off

    if op == OP_FLOAT64:
        v = struct.unpack(">d", raw[off:off+8])[0]
        return float(v), off + 8

    if op == OP_UUID16:
        b = raw[off:off+16]
        return ("uuid", b), off + 16

    if op == OP_BYTES:
        n, off2 = decode_uvarint(raw, off)
        b = raw[off2:off2+n]
        return bytes(b), off2 + n

    if op == OP_TIME_MS:
        ms, off = decode_uvarint(raw, off)
        return ("time", int(ms)), off

    if op == OP_STR_REF:
        sid, off = decode_uvarint(raw, off)
        return ("str", state.id_to_str[sid]), off

    if op == OP_STR_NEW:
        s, off = _unpack_string(raw, off)
        sid = len(state.id_to_str)
        state.id_to_str.append(s)
        state.str_to_id[s] = sid
        return ("str", s), off

    return ("str", ""), off


def _format_value(v: Any) -> str:
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return repr(v)
    if isinstance(v, bytes):
        return "0x" + v.hex()
    if isinstance(v, tuple) and v[0] == "uuid":
        h = v[1].hex()
        return f"{h[0:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"
    if isinstance(v, tuple) and v[0] == "time":
        dt = datetime.fromtimestamp(v[1] / 1000, tz=timezone.utc)
        return dt.isoformat().replace("+00:00", "Z")
    if isinstance(v, tuple) and v[0] == "str":
        return v[1]
    return str(v)


# -------------------------------------------------
# DATA packet
# -------------------------------------------------
def encode_data_packet(chunks: List[str], state: StreamStateV3C, level: int = 10) -> bytes:
    tids: List[int] = []
    vals_per: List[List[Any]] = []

    for ch in chunks:
        t, vals = extract_template_typed(ch)
        if t not in state.temp_index:
            raise ValueError("Template not in dict. Send/Apply DICT first.")
        tid = state.temp_index[t]
        tids.append(tid)
        vals_per.append(vals)

    # tids -> mtf positions
    positions: List[int] = []
    for tid in tids:
        pos = state.mtf.index(tid)
        positions.append(pos)
        state.mtf.pop(pos)
        state.mtf.insert(0, tid)

    max_pos = max(positions) if positions else 0
    pos_bits = max(1, max_pos.bit_length())
    packed_positions = _bitpack(positions, pos_bits)

    out = bytearray()
    out += MAGIC_DATA
    out += encode_uvarint(len(chunks))
    out += encode_uvarint(pos_bits)
    out += encode_uvarint(len(packed_positions))
    out += packed_positions

    for tid, vals in zip(tids, vals_per):
        arity = state.arity_by_tid.get(tid, len(vals))
        if len(vals) != arity:
            if len(vals) < arity:
                vals = vals + [("str", "")] * (arity - len(vals))
            else:
                vals = vals[:arity]

        prev_list = state.prev_vals_by_tid.get(tid, [None] * arity)
        new_prev: List[Any] = []

        for i in range(arity):
            prev = prev_list[i] if i < len(prev_list) else None
            v = vals[i]
            _emit_value(state, prev, v, out)
            new_prev.append(v)

        state.seen_tid[tid] = True
        state.prev_vals_by_tid[tid] = new_prev

    return zstd_compress(bytes(out), level=level)


def decode_data_packet(packet_bytes: bytes, state: StreamStateV3C) -> List[str]:
    raw = zstd_decompress(packet_bytes)
    if not raw.startswith(MAGIC_DATA):
        raise ValueError("Not a 3C DATA packet")

    off = len(MAGIC_DATA)

    nchunks, off = decode_uvarint(raw, off)
    pos_bits, off = decode_uvarint(raw, off)
    packed_len, off = decode_uvarint(raw, off)
    packed_positions = raw[off : off + packed_len]
    off += packed_len

    positions = _bitunpack(packed_positions, nchunks, pos_bits)

    tids: List[int] = []
    for pos in positions:
        tid = state.mtf[pos]
        tids.append(tid)
        state.mtf.pop(pos)
        state.mtf.insert(0, tid)

    out_chunks: List[str] = []
    for tid in tids:
        template = state.templates[tid]
        arity = state.arity_by_tid.get(tid, 0)

        prev_list = state.prev_vals_by_tid.get(tid, [None] * arity)
        new_prev: List[Any] = []
        vals: List[Any] = []

        for i in range(arity):
            prev = prev_list[i] if i < len(prev_list) else None
            v, off = _read_value(state, raw, off, prev)
            vals.append(v)
            new_prev.append(v)

        state.prev_vals_by_tid[tid] = new_prev
        state.seen_tid[tid] = True

        fmt_vals = [_format_value(v) for v in vals]
        out_chunks.append(template.format(*fmt_vals))

    return out_chunks
