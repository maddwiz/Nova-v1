from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import string
import re

from usc.mem.varint import encode_uvarint, decode_uvarint
from usc.mem.zstd_codec import zstd_compress, zstd_decompress


MAGIC_DICT = b"USDICT3B"  # smaller DICT packet (no tid, no arity)
MAGIC_DATA = b"USDATAZ3"  # reuse v3 DATA packet format


_INT_RE = re.compile(r"-?\d+")


def _extract_template_ints_only(text: str) -> Tuple[str, List[int]]:
    """
    Lossless + safe:
    - replaces ONLY signed decimal integers with {}
    - returns list of ints in appearance order
    - DOES NOT convert letters like 'B' -> 66
    """
    vals: List[int] = []

    def repl(m: re.Match) -> str:
        vals.append(int(m.group(0)))
        return "{}"

    templ = _INT_RE.sub(repl, text)
    return templ, vals


def _pack_string(s: str) -> bytes:
    b = s.encode("utf-8")
    return encode_uvarint(len(b)) + b


def _unpack_string(data: bytes, offset: int) -> Tuple[str, int]:
    n, off = decode_uvarint(data, offset)
    b = data[off : off + n]
    off += n
    return b.decode("utf-8"), off


def _zigzag_encode(n: int) -> int:
    return (n * 2) if n >= 0 else (-n * 2 - 1)


def _zigzag_decode(z: int) -> int:
    return (z // 2) if (z % 2 == 0) else -(z // 2) - 1


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


def _count_format_fields(fmt: str) -> int:
    """
    Robust placeholder counter for Python format strings.
    Handles {}, {0}, {name}, escaped braces {{ }} etc.
    """
    n = 0
    for literal_text, field_name, format_spec, conversion in string.Formatter().parse(fmt):
        if field_name is not None:
            n += 1
    return n


@dataclass
class StreamStateV3B:
    templates: List[str] = field(default_factory=list)
    temp_index: Dict[str, int] = field(default_factory=dict)
    arity_by_tid: Dict[int, int] = field(default_factory=dict)

    mtf: List[int] = field(default_factory=list)

    seen_tid: Dict[int, bool] = field(default_factory=dict)
    prev_vals_by_tid: Dict[int, List[int]] = field(default_factory=dict)


def build_dict_state_from_chunks(chunks: List[str], state: StreamStateV3B | None = None) -> StreamStateV3B:
    if state is None:
        state = StreamStateV3B()

    for ch in chunks:
        t, vals = _extract_template_ints_only(ch)
        if t not in state.temp_index:
            tid = len(state.templates)
            state.temp_index[t] = tid
            state.templates.append(t)
            state.arity_by_tid[tid] = len(vals)
            state.mtf.append(tid)

    return state


def encode_dict_packet(state: StreamStateV3B, level: int = 10) -> bytes:
    """
    Smaller DICT packet:
    - stores ONLY templates, in order
    - tid is implied by position
    - arity is recomputed on receiver robustly via format parser
    """
    out = bytearray()
    out += MAGIC_DICT

    out += encode_uvarint(len(state.templates))
    for t in state.templates:
        out += _pack_string(t)

    return zstd_compress(bytes(out), level=level)


def apply_dict_packet(packet_bytes: bytes, state: StreamStateV3B | None = None) -> StreamStateV3B:
    if state is None:
        state = StreamStateV3B()

    raw = zstd_decompress(packet_bytes)
    if not raw.startswith(MAGIC_DICT):
        raise ValueError("Not a 3B DICT packet")

    off = len(MAGIC_DICT)

    ntemps, off = decode_uvarint(raw, off)
    for tid in range(ntemps):
        t, off = _unpack_string(raw, off)

        state.templates.append(t)
        state.temp_index[t] = tid

        # robust arity count
        arity = _count_format_fields(t)
        state.arity_by_tid[tid] = arity

        if tid not in state.mtf:
            state.mtf.append(tid)

    return state


def encode_data_packet(chunks: List[str], state: StreamStateV3B, level: int = 10) -> bytes:
    tids: List[int] = []
    values_per_chunk: List[List[int]] = []

    for ch in chunks:
        t, vals = _extract_template_ints_only(ch)
        if t not in state.temp_index:
            raise ValueError("Template not in dict. Send/Apply DICT first.")
        tid = state.temp_index[t]
        tids.append(tid)
        values_per_chunk.append(vals)

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

    for tid, vals in zip(tids, values_per_chunk):
        arity = state.arity_by_tid.get(tid, len(vals))

        if len(vals) != arity:
            if len(vals) < arity:
                vals = vals + [0] * (arity - len(vals))
            else:
                vals = vals[:arity]

        if not state.seen_tid.get(tid, False):
            for v in vals:
                out += encode_uvarint(_zigzag_encode(v))
            state.seen_tid[tid] = True
            state.prev_vals_by_tid[tid] = list(vals)
        else:
            prev = state.prev_vals_by_tid.get(tid, [0] * arity)
            new_prev: List[int] = []
            for i in range(arity):
                v = vals[i]
                pv = prev[i] if i < len(prev) else 0
                d = v - pv
                out += encode_uvarint(_zigzag_encode(d))
                new_prev.append(v)
            state.prev_vals_by_tid[tid] = new_prev

    return zstd_compress(bytes(out), level=level)
