from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from usc.mem.varint import encode_uvarint, decode_uvarint
from usc.mem.templatepack import _extract_template
from usc.mem.zstd_codec import zstd_compress, zstd_decompress


MAGIC = b"USSWCZ2"  # USC Stream Window CANZ v2 (arity stored once)


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


@dataclass
class StreamStateV2:
    templates: List[str] = field(default_factory=list)
    temp_index: Dict[str, int] = field(default_factory=dict)
    mtf: List[int] = field(default_factory=list)

    # template arity (# of slots) stored ONCE when template first appears
    arity_by_tid: Dict[int, int] = field(default_factory=dict)

    seen_tid: Dict[int, bool] = field(default_factory=dict)
    prev_vals_by_tid: Dict[int, List[int]] = field(default_factory=dict)


def encode_stream_window_canz_v2(chunks: List[str], state: StreamStateV2 | None = None) -> bytes:
    """
    Stream Window CANZ v2:
    - persistent template dict + mtf + delta history in state
    - packet contains only NEW templates since last call
    - within this window, template IDs are encoded as MTF positions and bitpacked
    - values are delta-only per template across stream
    - ✅ removed `nvals` from every chunk by storing template arity once
    """
    if state is None:
        state = StreamStateV2()

    tids: List[int] = []
    values_per_chunk: List[List[int]] = []

    new_templates: List[Tuple[int, str, int]] = []

    # Extract templates + values, update dictionary if new
    for ch in chunks:
        t, vals = _extract_template(ch)

        if t not in state.temp_index:
            tid = len(state.templates)
            state.temp_index[t] = tid
            state.templates.append(t)
            state.mtf.append(tid)

            arity = len(vals)
            state.arity_by_tid[tid] = arity

            new_templates.append((tid, t, arity))
        else:
            tid = state.temp_index[t]

        tids.append(tid)
        values_per_chunk.append(vals)

    # Convert tids -> mtf positions (mutating mtf)
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
    out += MAGIC

    # new templates in this window
    out += encode_uvarint(len(new_templates))
    for tid, t, arity in new_templates:
        out += encode_uvarint(tid)
        out += encode_uvarint(arity)
        out += _pack_string(t)

    # window chunk count
    out += encode_uvarint(len(chunks))

    # bitpacked mtf positions
    out += encode_uvarint(pos_bits)
    out += encode_uvarint(len(packed_positions))
    out += packed_positions

    # values (delta-only across stream per template, fixed arity)
    for tid, vals in zip(tids, values_per_chunk):
        arity = state.arity_by_tid.get(tid, len(vals))

        # ensure vals length matches arity
        if len(vals) != arity:
            # pad or trim to arity
            if len(vals) < arity:
                vals = vals + [0] * (arity - len(vals))
            else:
                vals = vals[:arity]

        if not state.seen_tid.get(tid, False):
            for v in vals:
                out += encode_uvarint(v)
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

    return zstd_compress(bytes(out), level=10)


def decode_stream_window_canz_v2(packet_bytes: bytes, state: StreamStateV2 | None = None) -> List[str]:
    """
    Decode a window packet using state.
    """
    if state is None:
        state = StreamStateV2()

    raw = zstd_decompress(packet_bytes)
    if not raw.startswith(MAGIC):
        raise ValueError("Not a Stream Window CANZ v2 packet")

    off = len(MAGIC)

    # read new templates
    nnew, off = decode_uvarint(raw, off)
    for _ in range(nnew):
        tid, off = decode_uvarint(raw, off)
        arity, off = decode_uvarint(raw, off)
        t, off = _unpack_string(raw, off)

        while len(state.templates) <= tid:
            state.templates.append("")
        state.templates[tid] = t
        state.temp_index[t] = tid
        state.arity_by_tid[tid] = arity

        if tid not in state.mtf:
            state.mtf.append(tid)

    # chunk count
    nchunks, off = decode_uvarint(raw, off)

    # positions
    pos_bits, off = decode_uvarint(raw, off)
    packed_len, off = decode_uvarint(raw, off)
    packed_positions = raw[off : off + packed_len]
    off += packed_len

    positions = _bitunpack(packed_positions, nchunks, pos_bits)

    # positions -> tids using mtf
    tids: List[int] = []
    for pos in positions:
        tid = state.mtf[pos]
        tids.append(tid)
        state.mtf.pop(pos)
        state.mtf.insert(0, tid)

    out_chunks: List[str] = []
    for tid in tids:
        arity = state.arity_by_tid.get(tid, 0)
        vals: List[int] = []

        if not state.seen_tid.get(tid, False):
            for _ in range(arity):
                v, off = decode_uvarint(raw, off)
                vals.append(v)
            state.seen_tid[tid] = True
            state.prev_vals_by_tid[tid] = list(vals)
        else:
            prev = state.prev_vals_by_tid.get(tid, [0] * arity)
            new_prev: List[int] = []
            for i in range(arity):
                z, off = decode_uvarint(raw, off)
                d = _zigzag_decode(z)
                pv = prev[i] if i < len(prev) else 0
                v = pv + d
                vals.append(v)
                new_prev.append(v)
            state.prev_vals_by_tid[tid] = new_prev

        template = state.templates[tid]
        out_chunks.append(template.format(*vals))

    return out_chunks
