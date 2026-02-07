import gzip
from typing import Dict, List, Tuple

from usc.mem.varint import encode_uvarint, decode_uvarint
from usc.mem.templatepack import _extract_template


MAGIC = b"USCMT1"  # Unified State Codec TemplateMTFPack v0.1


def _pack_string(s: str) -> bytes:
    b = s.encode("utf-8")
    return encode_uvarint(len(b)) + b


def _unpack_string(data: bytes, offset: int) -> Tuple[str, int]:
    n, off = decode_uvarint(data, offset)
    b = data[off : off + n]
    off += n
    return b.decode("utf-8"), off


def _mtf_encode(tids: List[int], alphabet_size: int) -> List[int]:
    """
    Move-to-front encoding:
    We keep a list [0,1,2,...] and output the position where tid is found.
    Then we move that tid to the front.
    """
    mtf = list(range(alphabet_size))
    out: List[int] = []
    for tid in tids:
        pos = mtf.index(tid)
        out.append(pos)
        # move to front
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


def encode_chunks_with_template_mtf(chunks: List[str]) -> bytes:
    """
    TemplateMTFPack v0.1

    ✅ Same templates as TemplatePack
    ✅ NEW: compress template-id stream with MTF positions
    ✅ Values stored same as TemplatePack
    """
    templates: List[str] = []
    temp_index: Dict[str, int] = {}

    tids: List[int] = []
    values_per_chunk: List[List[int]] = []

    # Build templates and chunk records
    for ch in chunks:
        t, vals = _extract_template(ch)
        if t not in temp_index:
            temp_index[t] = len(templates)
            templates.append(t)
        tid = temp_index[t]
        tids.append(tid)
        values_per_chunk.append(vals)

    # MTF encode ids
    positions = _mtf_encode(tids, alphabet_size=len(templates))

    out = bytearray()
    out += MAGIC

    # templates table
    out += encode_uvarint(len(templates))
    for t in templates:
        out += _pack_string(t)

    # chunk count
    out += encode_uvarint(len(chunks))

    # write MTF positions
    for p in positions:
        out += encode_uvarint(p)

    # write values per chunk
    for vals in values_per_chunk:
        out += encode_uvarint(len(vals))
        for v in vals:
            out += encode_uvarint(v)

    return gzip.compress(bytes(out), compresslevel=9)


def decode_chunks_with_template_mtf(packet_bytes: bytes) -> List[str]:
    raw = gzip.decompress(packet_bytes)
    if not raw.startswith(MAGIC):
        raise ValueError("Not a TemplateMTFPack v0.1 packet")

    off = len(MAGIC)

    # templates
    ntemps, off = decode_uvarint(raw, off)
    templates: List[str] = []
    for _ in range(ntemps):
        s, off = _unpack_string(raw, off)
        templates.append(s)

    # chunks
    nchunks, off = decode_uvarint(raw, off)

    # read MTF positions
    positions: List[int] = []
    for _ in range(nchunks):
        p, off = decode_uvarint(raw, off)
        positions.append(p)

    tids = _mtf_decode(positions, alphabet_size=len(templates))

    out_chunks: List[str] = []

    # read values per chunk and reconstruct
    for i in range(nchunks):
        nvals, off = decode_uvarint(raw, off)
        vals: List[int] = []
        for _ in range(nvals):
            v, off = decode_uvarint(raw, off)
            vals.append(v)

        tid = tids[i]
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
