import gzip
from typing import Dict, List, Tuple

from usc.mem.varint import encode_uvarint, decode_uvarint
from usc.mem.templatepack import _extract_template


MAGIC = b"USCTD2"  # TemplateDeltaPack v0.2 (adaptive abs/delta)


def _pack_string(s: str) -> bytes:
    b = s.encode("utf-8")
    return encode_uvarint(len(b)) + b


def _unpack_string(data: bytes, offset: int) -> Tuple[str, int]:
    n, off = decode_uvarint(data, offset)
    b = data[off : off + n]
    off += n
    return b.decode("utf-8"), off


def _zigzag_encode(n: int) -> int:
    # signed -> unsigned (works fine for small/normal ints)
    return (n << 1) ^ (n >> 63)


def _zigzag_decode(z: int) -> int:
    return (z >> 1) ^ -(z & 1)


def _encode_value_adaptive(prev_v: int, cur_v: int) -> bytes:
    """
    Store whichever is smaller:
      ABS: cur_v
      DELTA: cur_v - prev_v (zigzag)
    We pack into a single uvarint:
      code = (payload << 1) | tag
      tag=0 => ABS, payload=cur_v
      tag=1 => DELTA, payload=zigzag(delta)
    """
    delta = cur_v - prev_v
    abs_payload = cur_v
    delta_payload = _zigzag_encode(delta)

    abs_code = (abs_payload << 1) | 0
    delta_code = (delta_payload << 1) | 1

    abs_bytes = encode_uvarint(abs_code)
    delta_bytes = encode_uvarint(delta_code)

    if len(abs_bytes) <= len(delta_bytes):
        return abs_bytes
    return delta_bytes


def _decode_value_adaptive(prev_v: int, data: bytes, off: int) -> Tuple[int, int]:
    code, off2 = decode_uvarint(data, off)
    tag = code & 1
    payload = code >> 1

    if tag == 0:
        # ABS
        return payload, off2

    # DELTA
    d = _zigzag_decode(payload)
    return prev_v + d, off2


def encode_chunks_with_template_deltas(chunks: List[str]) -> bytes:
    """
    TemplateDeltaPack v0.2:
    - Uses same template extraction as TemplatePack
    - Adaptive per-slot encoding: ABS or DELTA (whichever smaller)
    """
    templates: List[str] = []
    temp_index: Dict[str, int] = {}
    chunk_records: List[Tuple[int, List[int]]] = []

    for ch in chunks:
        t, vals = _extract_template(ch)
        if t not in temp_index:
            temp_index[t] = len(templates)
            templates.append(t)
        tid = temp_index[t]
        chunk_records.append((tid, vals))

    out = bytearray()
    out += MAGIC

    # templates table
    out += encode_uvarint(len(templates))
    for t in templates:
        out += _pack_string(t)

    # chunk count
    out += encode_uvarint(len(chunk_records))

    prev_vals_by_tid: Dict[int, List[int]] = {}

    for tid, vals in chunk_records:
        out += encode_uvarint(tid)
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


def decode_chunks_with_template_deltas(packet_bytes: bytes) -> List[str]:
    raw = gzip.decompress(packet_bytes)
    if not raw.startswith(MAGIC):
        raise ValueError("Not a TemplateDeltaPack v0.2 packet")

    off = len(MAGIC)

    # templates
    ntemps, off = decode_uvarint(raw, off)
    templates: List[str] = []
    for _ in range(ntemps):
        s, off = _unpack_string(raw, off)
        templates.append(s)

    # chunks
    nchunks, off = decode_uvarint(raw, off)

    prev_vals_by_tid: Dict[int, List[int]] = {}
    out_chunks: List[str] = []

    for _ in range(nchunks):
        tid, off = decode_uvarint(raw, off)
        nvals, off = decode_uvarint(raw, off)

        prev = prev_vals_by_tid.get(tid)
        if prev is None:
            prev = [0] * nvals

        vals: List[int] = []
        for i in range(nvals):
            pv = prev[i] if i < len(prev) else 0
            v, off = _decode_value_adaptive(pv, raw, off)
            vals.append(v)

        prev_vals_by_tid[tid] = vals

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
