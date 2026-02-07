import gzip
from typing import Dict, List, Tuple

from usc.mem.varint import encode_uvarint, decode_uvarint
from usc.mem.templatepack import _extract_template


MAGIC = b"USCMB1"  # Unified State Codec TemplateMTFBitPack v0.1


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
    """
    Pack small ints into a bitstream using fixed bit-width.
    """
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
    """
    Unpack n ints from a bitstream using fixed bit-width.
    """
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


def encode_chunks_with_template_mtf_bits(chunks: List[str]) -> bytes:
    """
    TemplateMTFBitPack v0.1

    ✅ Same as TMTF but bit-packs the MTF positions.
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

    positions = _mtf_encode(tids, alphabet_size=len(templates))

    max_pos = max(positions) if positions else 0
    bit_width = max(1, max_pos.bit_length())

    packed_positions = _bitpack(positions, bit_width)

    out = bytearray()
    out += MAGIC

    # templates table
    out += encode_uvarint(len(templates))
    for t in templates:
        out += _pack_string(t)

    # chunk count
    out += encode_uvarint(len(chunks))

    # bitpack header + data
    out += encode_uvarint(bit_width)
    out += encode_uvarint(len(packed_positions))
    out += packed_positions

    # values per chunk
    for vals in values_per_chunk:
        out += encode_uvarint(len(vals))
        for v in vals:
            out += encode_uvarint(v)

    return gzip.compress(bytes(out), compresslevel=9)


def decode_chunks_with_template_mtf_bits(packet_bytes: bytes) -> List[str]:
    raw = gzip.decompress(packet_bytes)
    if not raw.startswith(MAGIC):
        raise ValueError("Not a TemplateMTFBitPack v0.1 packet")

    off = len(MAGIC)

    # templates
    ntemps, off = decode_uvarint(raw, off)
    templates: List[str] = []
    for _ in range(ntemps):
        s, off = _unpack_string(raw, off)
        templates.append(s)

    # chunk count
    nchunks, off = decode_uvarint(raw, off)

    # bitpack header + data
    bit_width, off = decode_uvarint(raw, off)
    packed_len, off = decode_uvarint(raw, off)
    packed_positions = raw[off : off + packed_len]
    off += packed_len

    positions = _bitunpack(packed_positions, nchunks, bit_width)
    tids = _mtf_decode(positions, alphabet_size=len(templates))

    out_chunks: List[str] = []

    for i in range(nchunks):
        nvals, off = decode_uvarint(raw, off)
        vals: List[int] = []
        for _ in range(nvals):
            v, off = decode_uvarint(raw, off)
            vals.append(v)

        template = templates[tids[i]]

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
