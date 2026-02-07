import gzip
from typing import List, Dict, Tuple

from usc.mem.varint import encode_uvarint, decode_uvarint
from usc.mem.templatemtf_bits_deltaonly import (
    _mtf_encode,
    _mtf_decode,
    _bitpack,
    _bitunpack,
    _zigzag_encode,
    _zigzag_decode,
)
from usc.mem.drain_templates import drain_extract_templates, convert_drain_to_numeric


MAGIC = b"USCDR1"  # DrainPack v0.1


def _pack_string(s: str) -> bytes:
    b = s.encode("utf-8")
    return encode_uvarint(len(b)) + b


def _unpack_string(data: bytes, off: int) -> Tuple[str, int]:
    n, off = decode_uvarint(data, off)
    b = data[off : off + n]
    return b.decode("utf-8"), off + n


def encode_chunks_drainpack(lines: List[str]) -> bytes:
    """
    DrainPack:
    - Drain3 finds templates + params
    - We encode template IDs with MTF+bitpack
    - We encode numeric params with delta-only per template stream (like TMTFDO)
    """
    templates_per_line, params_per_line = drain_extract_templates(lines)
    vals_per_line = convert_drain_to_numeric(params_per_line)

    # Build template table
    template_table: List[str] = []
    t_index: Dict[str, int] = {}

    tids: List[int] = []
    for t in templates_per_line:
        if t not in t_index:
            t_index[t] = len(template_table)
            template_table.append(t)
        tids.append(t_index[t])

    # Template id stream: MTF + bitpack
    positions = _mtf_encode(tids, alphabet_size=len(template_table))
    max_pos = max(positions) if positions else 0
    pos_bits = max(1, max_pos.bit_length())
    packed_positions = _bitpack(positions, pos_bits)

    out = bytearray()
    out += MAGIC

    # store template table
    out += encode_uvarint(len(template_table))
    for t in template_table:
        out += _pack_string(t)

    # line count
    out += encode_uvarint(len(lines))

    # store positions
    out += encode_uvarint(pos_bits)
    out += encode_uvarint(len(packed_positions))
    out += packed_positions

    # store values: delta-only per template stream
    seen_tid: Dict[int, bool] = {}
    prev_vals_by_tid: Dict[int, List[int]] = {}

    for tid, vals in zip(tids, vals_per_line):
        out += encode_uvarint(len(vals))

        if not seen_tid.get(tid, False):
            # first time -> ABS
            for v in vals:
                out += encode_uvarint(v)
            seen_tid[tid] = True
            prev_vals_by_tid[tid] = list(vals)
        else:
            prev = prev_vals_by_tid.get(tid, [0] * len(vals))
            new_prev: List[int] = []
            for i, v in enumerate(vals):
                pv = prev[i] if i < len(prev) else 0
                d = v - pv
                out += encode_uvarint(_zigzag_encode(d))
                new_prev.append(v)
            prev_vals_by_tid[tid] = new_prev

    return gzip.compress(bytes(out), compresslevel=9)


def decode_chunks_drainpack(packet_bytes: bytes) -> List[str]:
    raw = gzip.decompress(packet_bytes)
    if not raw.startswith(MAGIC):
        raise ValueError("Not a DrainPack packet")

    off = len(MAGIC)

    # template table
    ntemps, off = decode_uvarint(raw, off)
    table: List[str] = []
    for _ in range(ntemps):
        s, off = _unpack_string(raw, off)
        table.append(s)

    nlines, off = decode_uvarint(raw, off)

    # positions
    pos_bits, off = decode_uvarint(raw, off)
    plen, off = decode_uvarint(raw, off)
    pdata = raw[off : off + plen]
    off += plen

    positions = _bitunpack(pdata, nlines, pos_bits)
    tids = _mtf_decode(positions, alphabet_size=len(table))

    seen_tid: Dict[int, bool] = {}
    prev_vals_by_tid: Dict[int, List[int]] = {}

    out_lines: List[str] = []

    for tid in tids:
        nvals, off = decode_uvarint(raw, off)
        vals: List[int] = []

        if not seen_tid.get(tid, False):
            for _ in range(nvals):
                v, off = decode_uvarint(raw, off)
                vals.append(v)
            seen_tid[tid] = True
            prev_vals_by_tid[tid] = list(vals)
        else:
            prev = prev_vals_by_tid.get(tid, [0] * nvals)
            new_prev: List[int] = []
            for i in range(nvals):
                z, off = decode_uvarint(raw, off)
                d = _zigzag_decode(z)
                pv = prev[i] if i < len(prev) else 0
                v = pv + d
                vals.append(v)
                new_prev.append(v)
            prev_vals_by_tid[tid] = new_prev

        # For now we just return the template as-is (Drain uses "<*>" wildcards)
        # Later we will reinflate params into the template properly.
        out_lines.append(table[tid])

    return out_lines
