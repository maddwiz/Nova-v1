from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import re

from usc.mem.varint import encode_uvarint, decode_uvarint
from usc.mem.templatepack import _extract_template
from usc.mem.zstd_codec import zstd_compress, zstd_decompress


MAGIC_DICT = b"USDICTZ4"  # tokenized dictionary packet
MAGIC_DATA = b"USDATAZ3"  # reuse v3 DATA packet format


# Tokenization:
# We capture:
#  - "{}" placeholder
#  - words
#  - numbers
#  - punctuation/whitespace blocks
_TOKEN_RE = re.compile(r"(\{\}|\d+|[A-Za-z_]+|[^A-Za-z0-9_]+)")


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


def _tokenize_template(t: str) -> List[str]:
    parts = _TOKEN_RE.findall(t)
    if not parts:
        return [t]
    return parts


@dataclass
class StreamStateV4:
    templates: List[str] = field(default_factory=list)
    temp_index: Dict[str, int] = field(default_factory=dict)
    arity_by_tid: Dict[int, int] = field(default_factory=dict)
    mtf: List[int] = field(default_factory=list)

    seen_tid: Dict[int, bool] = field(default_factory=dict)
    prev_vals_by_tid: Dict[int, List[int]] = field(default_factory=dict)


def build_dict_state_from_chunks(chunks: List[str], state: StreamStateV4 | None = None) -> StreamStateV4:
    if state is None:
        state = StreamStateV4()

    for ch in chunks:
        t, vals = _extract_template(ch)
        if t not in state.temp_index:
            tid = len(state.templates)
            state.temp_index[t] = tid
            state.templates.append(t)
            state.arity_by_tid[tid] = len(vals)
            state.mtf.append(tid)

    return state


def encode_dict_packet_tokenized(state: StreamStateV4, level: int = 10) -> bytes:
    """
    Tokenized DICT packet:
    - build a global token dictionary from all template strings
    - store templates as token-id sequences
    This usually shrinks DICT size.
    """
    # Build token dict
    token_to_id: Dict[str, int] = {}
    tokens: List[str] = []

    templ_token_ids: List[List[int]] = []
    templ_arities: List[int] = []

    for tid, t in enumerate(state.templates):
        arity = state.arity_by_tid.get(tid, 0)
        templ_arities.append(arity)

        toks = _tokenize_template(t)
        ids: List[int] = []
        for tok in toks:
            if tok not in token_to_id:
                token_to_id[tok] = len(tokens)
                tokens.append(tok)
            ids.append(token_to_id[tok])
        templ_token_ids.append(ids)

    out = bytearray()
    out += MAGIC_DICT

    # Token dictionary
    out += encode_uvarint(len(tokens))
    for tok in tokens:
        out += _pack_string(tok)

    # Templates
    out += encode_uvarint(len(state.templates))
    for tid in range(len(state.templates)):
        out += encode_uvarint(tid)
        out += encode_uvarint(templ_arities[tid])

        ids = templ_token_ids[tid]
        out += encode_uvarint(len(ids))
        for x in ids:
            out += encode_uvarint(x)

    return zstd_compress(bytes(out), level=level)


def apply_dict_packet_tokenized(packet_bytes: bytes, state: StreamStateV4 | None = None) -> StreamStateV4:
    if state is None:
        state = StreamStateV4()

    raw = zstd_decompress(packet_bytes)
    if not raw.startswith(MAGIC_DICT):
        raise ValueError("Not a tokenized DICT packet")

    off = len(MAGIC_DICT)

    # tokens
    ntoks, off = decode_uvarint(raw, off)
    tokens: List[str] = []
    for _ in range(ntoks):
        tok, off = _unpack_string(raw, off)
        tokens.append(tok)

    # templates
    ntemps, off = decode_uvarint(raw, off)
    for _ in range(ntemps):
        tid, off = decode_uvarint(raw, off)
        arity, off = decode_uvarint(raw, off)
        seq_len, off = decode_uvarint(raw, off)

        ids: List[int] = []
        for _ in range(seq_len):
            x, off = decode_uvarint(raw, off)
            ids.append(x)

        t = "".join(tokens[i] for i in ids)

        while len(state.templates) <= tid:
            state.templates.append("")
        state.templates[tid] = t
        state.temp_index[t] = tid
        state.arity_by_tid[tid] = arity

        if tid not in state.mtf:
            state.mtf.append(tid)

    return state


def encode_data_packet(chunks: List[str], state: StreamStateV4, level: int = 10) -> bytes:
    """
    DATA packet same as v3:
    - MTF positions bitpacked
    - delta-only values
    - no templates included
    """
    tids: List[int] = []
    values_per_chunk: List[List[int]] = []

    for ch in chunks:
        t, vals = _extract_template(ch)
        if t not in state.temp_index:
            raise ValueError("Template not in dictionary state. Send/Apply DICT first.")
        tid = state.temp_index[t]
        tids.append(tid)
        values_per_chunk.append(vals)

    # convert tids -> mtf positions, mutate mtf list
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

    return zstd_compress(bytes(out), level=level)


def decode_data_packet(packet_bytes: bytes, state: StreamStateV4) -> List[str]:
    raw = zstd_decompress(packet_bytes)
    if not raw.startswith(MAGIC_DATA):
        raise ValueError("Not a DATA packet")

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
