import gzip
from typing import Dict, List, Tuple

from usc.mem.varint import encode_uvarint, decode_uvarint
from usc.mem.templatepack import _extract_template
from usc.mem.canonicalize_lossless import canonicalize_lossless, reinflate_placeholders


MAGIC = b"USCLCN2"  # Lossless Canonicalized TMTFDO v0.2 (stores per-template arity; no per-chunk nvals)


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
    return (n * 2) if n >= 0 else (-n * 2 - 1)


def _zigzag_decode(z: int) -> int:
    return (z // 2) if (z % 2 == 0) else -(z // 2) - 1


def encode_chunks_with_template_mtf_bits_deltaonly_lcanon(chunks: List[str]) -> bytes:
    """
    TMTFDO_LCAN v0.2 (lossless):
    - Canonicalize losslessly -> placeholders + tokens side-stream
    - Template + MTF + bitpack positions
    - Delta-only values per template stream
    - Store token side-stream so decode restores exact original text

    v0.2 change:
    - Store per-template arity ONCE -> remove per-chunk `nvals`
    """
    templates: List[str] = []
    temp_index: Dict[str, int] = {}

    tids: List[int] = []
    values_per_chunk: List[List[int]] = []
    tokens_per_chunk: List[List[str]] = []

    # Step 1: canonicalize each chunk losslessly
    canon_chunks: List[str] = []
    for ch in chunks:
        canon, toks = canonicalize_lossless(ch)
        canon_chunks.append(canon)
        tokens_per_chunk.append(toks)

    # Step 2: template extraction over canonical chunks
    for canon in canon_chunks:
        t, vals = _extract_template(canon)
        if t not in temp_index:
            temp_index[t] = len(templates)
            templates.append(t)
        tid = temp_index[t]
        tids.append(tid)
        values_per_chunk.append(vals)

    # Compute per-template arity once
    arity_by_tid: Dict[int, int] = {}
    for tid, vals in zip(tids, values_per_chunk):
        a = len(vals)
        if tid not in arity_by_tid:
            arity_by_tid[tid] = a
        else:
            if arity_by_tid[tid] != a:
                raise ValueError(
                    f"Template arity changed for tid={tid}: expected {arity_by_tid[tid]}, got {a}"
                )

    arities: List[int] = [0] * len(templates)
    for tid in range(len(templates)):
        arities[tid] = arity_by_tid.get(tid, 0)

    # Template IDs: MTF + bitpack
    positions = _mtf_encode(tids, alphabet_size=len(templates))
    max_pos = max(positions) if positions else 0
    pos_bits = max(1, max_pos.bit_length())
    packed_positions = _bitpack(positions, pos_bits)

    out = bytearray()
    out += MAGIC

    # Store templates
    out += encode_uvarint(len(templates))
    for t in templates:
        out += _pack_string(t)

    # Store per-template arity list
    out += encode_uvarint(len(arities))
    for a in arities:
        out += encode_uvarint(a)

    # Store chunk count
    out += encode_uvarint(len(chunks))

    # Store packed positions stream
    out += encode_uvarint(pos_bits)
    out += encode_uvarint(len(packed_positions))
    out += packed_positions

    # Store values with delta-only per template (no per-chunk nvals)
    seen_tid: Dict[int, bool] = {}
    prev_vals_by_tid: Dict[int, List[int]] = {}

    for tid, vals in zip(tids, values_per_chunk):
        nvals = arities[tid]
        if nvals != len(vals):
            raise ValueError(f"Arity mismatch for tid={tid}: arity={nvals}, len(vals)={len(vals)}")

        if not seen_tid.get(tid, False):
            for v in vals:
                out += encode_uvarint(v)
            seen_tid[tid] = True
            prev_vals_by_tid[tid] = list(vals)
        else:
            prev = prev_vals_by_tid.get(tid, [0] * nvals)
            new_prev: List[int] = []
            for i, v in enumerate(vals):
                pv = prev[i] if i < len(prev) else 0
                d = v - pv
                out += encode_uvarint(_zigzag_encode(d))
                new_prev.append(v)
            prev_vals_by_tid[tid] = new_prev

    # Store side-stream tokens (lossless reinflation)
    for toks in tokens_per_chunk:
        out += encode_uvarint(len(toks))
        for t in toks:
            out += _pack_string(t)

    return gzip.compress(bytes(out), compresslevel=9)


def decode_chunks_with_template_mtf_bits_deltaonly_lcanon(packet_bytes: bytes) -> List[str]:
    raw = gzip.decompress(packet_bytes)
    if not raw.startswith(MAGIC):
        raise ValueError("Not a TMTFDO_LCAN v0.2 packet")

    off = len(MAGIC)

    ntemps, off = decode_uvarint(raw, off)
    templates: List[str] = []
    for _ in range(ntemps):
        s, off = _unpack_string(raw, off)
        templates.append(s)

    # Read per-template arity list
    narities, off = decode_uvarint(raw, off)
    arities: List[int] = []
    for _ in range(narities):
        a, off = decode_uvarint(raw, off)
        arities.append(a)

    if narities != len(templates):
        raise ValueError("Arity list length mismatch vs templates")

    nchunks, off = decode_uvarint(raw, off)

    pos_bits, off = decode_uvarint(raw, off)
    packed_len, off = decode_uvarint(raw, off)
    packed_positions = raw[off : off + packed_len]
    off += packed_len

    positions = _bitunpack(packed_positions, nchunks, pos_bits)
    tids = _mtf_decode(positions, alphabet_size=len(templates))

    # Decode values (no per-chunk nvals)
    seen_tid: Dict[int, bool] = {}
    prev_vals_by_tid: Dict[int, List[int]] = {}

    canon_out: List[str] = []

    for tid in tids:
        nvals = arities[tid]
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

        if "variant={" in template:
            vals2 = []
            for x in vals:
                if 65 <= x <= 122:
                    vals2.append(chr(x))
                else:
                    vals2.append(x)
            canon_out.append(template.format(*vals2))
        else:
            canon_out.append(template.format(*vals))

    # Decode token streams and reinflate
    final_out: List[str] = []
    for canon_line in canon_out:
        tok_count, off = decode_uvarint(raw, off)
        toks: List[str] = []
        for _ in range(tok_count):
            t, off = _unpack_string(raw, off)
            toks.append(t)
        final_out.append(reinflate_placeholders(canon_line, toks))

    return final_out
