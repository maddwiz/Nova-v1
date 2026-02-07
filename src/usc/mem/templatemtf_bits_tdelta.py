import gzip
from typing import Dict, List, Tuple

from usc.mem.varint import encode_uvarint, decode_uvarint
from usc.mem.templatepack import _extract_template


MAGIC = b"USCBD1"  # TemplateMTF(BitPack IDs) + Adaptive Value Delta v0.1


def _pack_string(s: str) -> bytes:
    b = s.encode("utf-8")
    return encode_uvarint(len(b)) + b


def _unpack_string(data: bytes, offset: int) -> Tuple[str, int]:
    n, off = decode_uvarint(data, offset)
    b = data[off : off + n]
    off += n
    return b.decode("utf-8"), off


def _mtf_encode(tids: List[int], alphabet_size: int) -> List[int]:
    mtf = list(range(alphabet_size))
    out: List[int] = []
    for tid in tids:
        pos = mtf.index(tid)
        out.append(pos)
        mtf.pop(pos)
        mtf.insert(0, tid)
    return out


def _mtf_decode(positions: List[int], alphabet_size: int) -> List[int]:
    mtf = list(range(alphabet_size))
    tids: List[int] = []
    for pos in positions:
        tid = mtf[pos]
        tids.append(tid)
        mtf.pop(pos)
        mtf.insert(0, tid)
    return tids


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


def _zigzag_encode(n: int) -> int:
    # Safe zigzag for Python ints
    return (n * 2) if n >= 0 else (-n * 2 - 1)


def _zigzag_decode(z: int) -> int:
    return (z // 2) if (z % 2 == 0) else -(z // 2) - 1


def _encode_value_adaptive(prev_v: int, cur_v: int) -> bytes:
    """
    Store whichever is smaller:
      ABS: cur_v
      DELTA: cur_v - prev_v (zigzag)
    Encode into one uvarint:
      code = (payload << 1) | tag
      tag=0 => ABS payload=cur_v
      tag=1 => DELTA payload=zigzag(delta)
    """
    delta = cur_v - prev_v
    abs_payload = cur_v
    delta_payload = _zigzag_encode(delta)

    abs_code = (abs_payload << 1) | 0
    delta_code = (delta_payload << 1) | 1

    abs_bytes = encode_uvarint(abs_code)
    delta_bytes = encode_uvarint(delta_code)

    return abs_bytes if len(abs_bytes) <= len(delta_bytes) else delta_bytes


def _decode_value_adaptive(prev_v: int, data: bytes, off: int) -> Tuple[int, int]:
    code, off2 = decode_uvarint(data, off)
    tag = code & 1
    payload = code >> 1

    if tag == 0:
        return payload, off2

    d = _zigzag_decode(payload)
    return prev_v + d, off2


def encode_chunks_with_template_mtf_bits_tdelta(chunks: List[str]) -> bytes:
    """
    TMTFBD v0.1
    ✅ Template IDs: MTF + bitpack
    ✅ Values: adaptive ABS/DELTA per template-id stream (per-slot)
    """
    templates: List[str] = []
    temp_index: Dict[str, int] = {}

    tids: List[int] = []
    values_per_chunk: List[List[int]] = []

    for ch in chunks:
        t, vals = _extract_template(ch)
        if t not in temp_index:
            temp_index[t] = len(templates)
            templates.append(t)
        tid = temp_index[t]
        tids.append(tid)
        values_per_chunk.append(vals)

    # --- Template ID stream ---
    positions = _mtf_encode(tids, alphabet_size=len(templates))
    max_pos = max(positions) if positions else 0
    pos_bits = max(1, max_pos.bit_length())
    packed_positions = _bitpack(positions, pos_bits)

    out = bytearray()
    out += MAGIC

    # templates
    out += encode_uvarint(len(templates))
    for t in templates:
        out += _pack_string(t)

    # chunk count
    out += encode_uvarint(len(chunks))

    # positions bitpack
    out += encode_uvarint(pos_bits)
    out += encode_uvarint(len(packed_positions))
    out += packed_positions

    # --- Values stream with per-template prev state ---
    prev_vals_by_tid: Dict[int, List[int]] = {}

    for tid, vals in zip(tids, values_per_chunk):
        out += encode_uvarint(len(vals))

        prev = prev_vals_by_tid.get(tid)
        if prev is None:
            prev = [0] * len(vals)

        new_prev: List[int] = []

        for i, v in enumerate(vals):
            pv = prev[i] if i < len(prev) else 0
            out += _encode_value_adaptive(pv, v)
            new_prev.append(v)

        prev_vals_by_tid[tid] = new_prev

    return gzip.compress(bytes(out), compresslevel=9)


def decode_chunks_with_template_mtf_bits_tdelta(packet_bytes: bytes) -> List[str]:
    raw = gzip.decompress(packet_bytes)
    if not raw.startswith(MAGIC):
        raise ValueError("Not a TMTFBD packet")

    off = len(MAGIC)

    # templates
    ntemps, off = decode_uvarint(raw, off)
    templates: List[str] = []
    for _ in range(ntemps):
        s, off = _unpack_string(raw, off)
        templates.append(s)

    nchunks, off = decode_uvarint(raw, off)

    # positions
    pos_bits, off = decode_uvarint(raw, off)
    packed_len, off = decode_uvarint(raw, off)
    packed_positions = raw[off : off + packed_len]
    off += packed_len

    positions = _bitunpack(packed_positions, nchunks, pos_bits)
    tids = _mtf_decode(positions, alphabet_size=len(templates))

    prev_vals_by_tid: Dict[int, List[int]] = {}
    out_chunks: List[str] = []

    for tid in tids:
        nvals, off = decode_uvarint(raw, off)

        prev = prev_vals_by_tid.get(tid)
        if prev is None:
            prev = [0] * nvals

        vals: List[int] = []
        new_prev: List[int] = []

        for i in range(nvals):
            pv = prev[i] if i < len(prev) else 0
            v, off = _decode_value_adaptive(pv, raw, off)
            vals.append(v)
            new_prev.append(v)

        prev_vals_by_tid[tid] = new_prev

        template = templates[tid]

        # Convert ord('A') -> 'A' when needed
        if "variant={" in template:
            vals2 = []
            for x in vals:
                if 65 <= x <= 122:
                    vals2.append(chr(x))
                else:
                    vals2.append(x)
            out_chunks.append(template.format(*vals2))
        else:
            out_chunks.append(template.format(*vals))

    return out_chunks
