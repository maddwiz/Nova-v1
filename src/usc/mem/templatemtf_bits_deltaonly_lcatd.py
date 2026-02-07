import gzip
from typing import Dict, List, Tuple

from usc.mem.varint import encode_uvarint, decode_uvarint
from usc.mem.templatepack import _extract_template
from usc.mem.canonicalize_typed_lossless import (
    canonicalize_typed_lossless,
    reinflate_typed,
    TypedToken,
)


MAGIC = b"USLCATD1"  # Lossless Canonical (Typed + Dict) + TMTFDO v0.1


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


def encode_chunks_with_template_mtf_bits_deltaonly_lcatd(chunks: List[str]) -> bytes:
    """
    TMTFDO_LCATD:
    - Lossless typed canonicalization
    - Template + MTF + bitpack positions
    - Delta-only values per template stream
    - Typed tokens stored as global dictionary:
        dict = unique (ttype + payload) across the batch
        each chunk stores token IDs (uvarint)
    """
    templates: List[str] = []
    temp_index: Dict[str, int] = {}

    tids: List[int] = []
    values_per_chunk: List[List[int]] = []
    canon_chunks: List[str] = []
    tokens_per_chunk: List[List[TypedToken]] = []

    # 1) Canonicalize typed losslessly
    for ch in chunks:
        canon, toks = canonicalize_typed_lossless(ch)
        canon_chunks.append(canon)
        tokens_per_chunk.append(toks)

    # 2) Template extraction over canonical chunks
    for canon in canon_chunks:
        t, vals = _extract_template(canon)
        if t not in temp_index:
            temp_index[t] = len(templates)
            templates.append(t)
        tid = temp_index[t]
        tids.append(tid)
        values_per_chunk.append(vals)

    # 3) IDs stream: MTF + bitpack
    positions = _mtf_encode(tids, alphabet_size=len(templates))
    max_pos = max(positions) if positions else 0
    pos_bits = max(1, max_pos.bit_length())
    packed_positions = _bitpack(positions, pos_bits)

    # 4) Build global typed-token dictionary
    # key = (ttype, payload_bytes)
    tok_dict: List[Tuple[int, bytes]] = []
    tok_id: Dict[Tuple[int, bytes], int] = {}
    tok_ids_per_chunk: List[List[int]] = []

    for toks in tokens_per_chunk:
        ids: List[int] = []
        for tok in toks:
            key = (int(tok.ttype), tok.payload)
            if key not in tok_id:
                tok_id[key] = len(tok_dict)
                tok_dict.append(key)
            ids.append(tok_id[key])
        tok_ids_per_chunk.append(ids)

    out = bytearray()
    out += MAGIC

    # Templates
    out += encode_uvarint(len(templates))
    for t in templates:
        out += _pack_string(t)

    # Chunk count
    out += encode_uvarint(len(chunks))

    # Packed positions stream
    out += encode_uvarint(pos_bits)
    out += encode_uvarint(len(packed_positions))
    out += packed_positions

    # Values: delta-only per template
    seen_tid: Dict[int, bool] = {}
    prev_vals_by_tid: Dict[int, List[int]] = {}

    for tid, vals in zip(tids, values_per_chunk):
        out += encode_uvarint(len(vals))

        if not seen_tid.get(tid, False):
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

    # Typed token dictionary
    out += encode_uvarint(len(tok_dict))
    for ttype, payload in tok_dict:
        out += encode_uvarint(ttype)
        out += encode_uvarint(len(payload))
        out += payload

    # Token-ID streams per chunk
    for ids in tok_ids_per_chunk:
        out += encode_uvarint(len(ids))
        for x in ids:
            out += encode_uvarint(x)

    return gzip.compress(bytes(out), compresslevel=9)


def decode_chunks_with_template_mtf_bits_deltaonly_lcatd(packet_bytes: bytes) -> List[str]:
    raw = gzip.decompress(packet_bytes)
    if not raw.startswith(MAGIC):
        raise ValueError("Not a TMTFDO_LCATD packet")

    off = len(MAGIC)

    # Templates
    ntemps, off = decode_uvarint(raw, off)
    templates: List[str] = []
    for _ in range(ntemps):
        s, off = _unpack_string(raw, off)
        templates.append(s)

    # Chunk count
    nchunks, off = decode_uvarint(raw, off)

    # Packed positions
    pos_bits, off = decode_uvarint(raw, off)
    packed_len, off = decode_uvarint(raw, off)
    packed_positions = raw[off : off + packed_len]
    off += packed_len

    positions = _bitunpack(packed_positions, nchunks, pos_bits)
    tids = _mtf_decode(positions, alphabet_size=len(templates))

    # Values
    seen_tid: Dict[int, bool] = {}
    prev_vals_by_tid: Dict[int, List[int]] = {}

    canon_out: List[str] = []
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

        template = templates[tid]
        canon_out.append(template.format(*vals))

    # Token dictionary
    ndict, off = decode_uvarint(raw, off)
    dict_entries: List[Tuple[int, bytes]] = []
    for _ in range(ndict):
        ttype, off = decode_uvarint(raw, off)
        plen, off = decode_uvarint(raw, off)
        payload = raw[off : off + plen]
        off += plen
        dict_entries.append((int(ttype), payload))

    # Token-ID streams + reinflate
    final_out: List[str] = []
    for canon_line in canon_out:
        k, off = decode_uvarint(raw, off)
        toks: List[TypedToken] = []
        for _ in range(k):
            tid, off = decode_uvarint(raw, off)
            ttype, payload = dict_entries[tid]
            toks.append(TypedToken(ttype, payload))
        final_out.append(reinflate_typed(canon_line, toks))

    return final_out
