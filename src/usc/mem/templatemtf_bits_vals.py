import gzip
from typing import Dict, List, Tuple

from usc.mem.varint import encode_uvarint, decode_uvarint
from usc.mem.templatepack import _extract_template


MAGIC = b"USCBV1"  # TemplateMTF(BitPack IDs) + ValueBitPack v0.1


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


def encode_chunks_with_template_mtf_bits_vals(chunks: List[str]) -> bytes:
    """
    TMTFBV v0.1
    ✅ Bitpack MTF positions (template id stream)
    ✅ Bitpack numeric slot values PER TEMPLATE + PER SLOT
       (this is where bigger wins can happen)
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

    # --- Encode template ids as MTF positions, then bitpack ---
    positions = _mtf_encode(tids, alphabet_size=len(templates))
    max_pos = max(positions) if positions else 0
    pos_bits = max(1, max_pos.bit_length())
    packed_positions = _bitpack(positions, pos_bits)

    # --- Group values by template id ---
    by_tid: Dict[int, List[List[int]]] = {}
    for tid, vals in zip(tids, values_per_chunk):
        by_tid.setdefault(tid, []).append(vals)

    out = bytearray()
    out += MAGIC

    # templates
    out += encode_uvarint(len(templates))
    for t in templates:
        out += _pack_string(t)

    # chunks count
    out += encode_uvarint(len(chunks))

    # positions bitpack header + payload
    out += encode_uvarint(pos_bits)
    out += encode_uvarint(len(packed_positions))
    out += packed_positions

    # value packing section:
    # store number of tids present
    out += encode_uvarint(len(by_tid))

    # For each tid:
    #  - store tid
    #  - store number of chunks under this tid
    #  - store slot count
    #  - for each slot index:
    #       store bits needed + packed stream for that column
    for tid, rows in by_tid.items():
        out += encode_uvarint(tid)
        out += encode_uvarint(len(rows))

        slot_count = len(rows[0]) if rows else 0
        out += encode_uvarint(slot_count)

        # Pack column-wise: all values for slot0, then slot1, ...
        for s in range(slot_count):
            col = [r[s] for r in rows]
            max_v = max(col) if col else 0
            bits = max(1, max_v.bit_length())

            packed_col = _bitpack(col, bits)

            out += encode_uvarint(bits)
            out += encode_uvarint(len(packed_col))
            out += packed_col

    return gzip.compress(bytes(out), compresslevel=9)


def decode_chunks_with_template_mtf_bits_vals(packet_bytes: bytes) -> List[str]:
    raw = gzip.decompress(packet_bytes)
    if not raw.startswith(MAGIC):
        raise ValueError("Not a TMTFBV packet")

    off = len(MAGIC)

    # templates
    ntemps, off = decode_uvarint(raw, off)
    templates: List[str] = []
    for _ in range(ntemps):
        s, off = _unpack_string(raw, off)
        templates.append(s)

    # chunk count
    nchunks, off = decode_uvarint(raw, off)

    # positions
    pos_bits, off = decode_uvarint(raw, off)
    packed_len, off = decode_uvarint(raw, off)
    packed_positions = raw[off : off + packed_len]
    off += packed_len

    positions = _bitunpack(packed_positions, nchunks, pos_bits)
    tids = _mtf_decode(positions, alphabet_size=len(templates))

    # read grouped values
    ntids_present, off = decode_uvarint(raw, off)

    # We'll reconstruct the full values_per_chunk in original order.
    # First decode all values grouped by tid, then consume them when rebuilding.
    vals_by_tid: Dict[int, List[List[int]]] = {}

    for _ in range(ntids_present):
        tid, off = decode_uvarint(raw, off)
        nrows, off = decode_uvarint(raw, off)
        slot_count, off = decode_uvarint(raw, off)

        cols: List[List[int]] = []
        for _s in range(slot_count):
            bits, off = decode_uvarint(raw, off)
            plen, off = decode_uvarint(raw, off)
            pdata = raw[off : off + plen]
            off += plen
            col = _bitunpack(pdata, nrows, bits)
            cols.append(col)

        # convert cols -> rows
        rows: List[List[int]] = []
        for i in range(nrows):
            row = [cols[s][i] for s in range(slot_count)]
            rows.append(row)

        vals_by_tid[tid] = rows

    # We'll keep a cursor per tid to pop rows in original order
    cursors: Dict[int, int] = {tid: 0 for tid in vals_by_tid.keys()}

    out_chunks: List[str] = []
    for tid in tids:
        idx = cursors.get(tid, 0)
        row = vals_by_tid[tid][idx]
        cursors[tid] = idx + 1

        template = templates[tid]

        if "variant={" in template:
            vals2 = []
            for x in row:
                if 65 <= x <= 122:
                    vals2.append(chr(x))
                else:
                    vals2.append(x)
            out_chunks.append(template.format(*vals2))
        else:
            out_chunks.append(template.format(*row))

    return out_chunks
