from typing import Dict, List, Tuple

from usc.mem.varint import encode_uvarint, decode_uvarint
from usc.mem.templatepack import _extract_template
from usc.mem.zstd_codec import zstd_compress, zstd_decompress


MAGIC = b"USSTRMCZ1"


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


def _mtf_move_to_front(mtf: List[int], tid: int) -> int:
    """
    Returns the MTF position (index) of tid, and moves it to front.
    """
    pos = mtf.index(tid)
    mtf.pop(pos)
    mtf.insert(0, tid)
    return pos


def _mtf_take_at(mtf: List[int], pos: int) -> int:
    """
    Returns the tid at position pos, and moves it to front.
    """
    tid = mtf[pos]
    mtf.pop(pos)
    mtf.insert(0, tid)
    return tid


def encode_stream_tmtfdo_canz(chunks: List[str]) -> bytes:
    """
    Streaming TMTFDO_CANZ:
    - Maintains a persistent template dictionary across the stream
    - Maintains MTF ordering across the stream
    - Delta-only values per template across the whole stream
    - Uses zstd directly

    This is designed for incremental agent memory logs.
    """
    templates: List[str] = []
    temp_index: Dict[str, int] = {}

    mtf: List[int] = []  # MTF list of template IDs
    seen_tid: Dict[int, bool] = {}
    prev_vals_by_tid: Dict[int, List[int]] = {}

    out = bytearray()
    out += MAGIC
    out += encode_uvarint(len(chunks))

    for ch in chunks:
        # Extract template + values for this chunk
        t, vals = _extract_template(ch)

        new_template = False
        if t not in temp_index:
            tid = len(templates)
            temp_index[t] = tid
            templates.append(t)
            mtf.append(tid)  # new id appears at end initially
            new_template = True
        else:
            tid = temp_index[t]

        # Emit template definition if new
        out += encode_uvarint(1 if new_template else 0)
        if new_template:
            out += encode_uvarint(tid)
            out += _pack_string(t)

        # Emit MTF position for tid
        pos = _mtf_move_to_front(mtf, tid)
        out += encode_uvarint(pos)

        # Emit values (delta-only across stream for same tid)
        out += encode_uvarint(len(vals))

        if not seen_tid.get(tid, False):
            # first time this template ever appears
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

    return zstd_compress(bytes(out), level=10)


def decode_stream_tmtfdo_canz(packet_bytes: bytes) -> List[str]:
    raw = zstd_decompress(packet_bytes)
    if not raw.startswith(MAGIC):
        raise ValueError("Not a streaming USC packet")

    off = len(MAGIC)

    nchunks, off = decode_uvarint(raw, off)

    templates: List[str] = []
    mtf: List[int] = []
    seen_tid: Dict[int, bool] = {}
    prev_vals_by_tid: Dict[int, List[int]] = {}

    out_chunks: List[str] = []

    for _ in range(nchunks):
        # New template?
        is_new, off = decode_uvarint(raw, off)
        if is_new == 1:
            tid, off = decode_uvarint(raw, off)
            t, off = _unpack_string(raw, off)

            # Ensure list is large enough
            while len(templates) <= tid:
                templates.append("")
            templates[tid] = t

            if tid not in mtf:
                mtf.append(tid)

        # MTF position -> tid
        pos, off = decode_uvarint(raw, off)
        tid = _mtf_take_at(mtf, pos)

        # Values
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

        template = templates[tid]
        out_chunks.append(template.format(*vals))

    return out_chunks
