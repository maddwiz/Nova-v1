import gzip
from typing import Dict, List, Tuple

from usc.mem.varint import encode_uvarint, decode_uvarint
from usc.mem.templatepack import _extract_template


MAGIC = b"USCTR1"  # Unified State Codec TemplateRLEPack v0.1


def _pack_string(s: str) -> bytes:
    b = s.encode("utf-8")
    return encode_uvarint(len(b)) + b


def _unpack_string(data: bytes, offset: int) -> Tuple[str, int]:
    n, off = decode_uvarint(data, offset)
    b = data[off : off + n]
    off += n
    return b.decode("utf-8"), off


def encode_chunks_with_template_rle(chunks: List[str]) -> bytes:
    """
    TemplateRLEPack v0.1

    ✅ Same template extraction as TemplatePack
    ✅ NEW: compress the template-id stream using RLE
       - store (tid, run_length)
    ✅ Store values for each chunk in run order (same as TemplatePack)
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

    # RLE encode tids
    runs: List[Tuple[int, int]] = []
    if tids:
        cur_tid = tids[0]
        run_len = 1
        for tid in tids[1:]:
            if tid == cur_tid:
                run_len += 1
            else:
                runs.append((cur_tid, run_len))
                cur_tid = tid
                run_len = 1
        runs.append((cur_tid, run_len))

    out = bytearray()
    out += MAGIC

    # templates table
    out += encode_uvarint(len(templates))
    for t in templates:
        out += _pack_string(t)

    # number of runs
    out += encode_uvarint(len(runs))

    # run headers
    for tid, run_len in runs:
        out += encode_uvarint(tid)
        out += encode_uvarint(run_len)

    # chunk values (still per chunk, in order)
    out += encode_uvarint(len(values_per_chunk))
    for vals in values_per_chunk:
        out += encode_uvarint(len(vals))
        for v in vals:
            out += encode_uvarint(v)

    return gzip.compress(bytes(out), compresslevel=9)


def decode_chunks_with_template_rle(packet_bytes: bytes) -> List[str]:
    raw = gzip.decompress(packet_bytes)
    if not raw.startswith(MAGIC):
        raise ValueError("Not a TemplateRLEPack v0.1 packet")

    off = len(MAGIC)

    # templates
    ntemps, off = decode_uvarint(raw, off)
    templates: List[str] = []
    for _ in range(ntemps):
        s, off = _unpack_string(raw, off)
        templates.append(s)

    # runs
    nruns, off = decode_uvarint(raw, off)
    tids: List[int] = []
    for _ in range(nruns):
        tid, off = decode_uvarint(raw, off)
        run_len, off = decode_uvarint(raw, off)
        tids.extend([tid] * run_len)

    # values
    nchunks, off = decode_uvarint(raw, off)
    if nchunks != len(tids):
        raise ValueError("Mismatch: RLE tids count != chunk count")

    out_chunks: List[str] = []
    for i in range(nchunks):
        nvals, off = decode_uvarint(raw, off)
        vals: List[int] = []
        for _ in range(nvals):
            v, off = decode_uvarint(raw, off)
            vals.append(v)

        tid = tids[i]
        template = templates[tid]

        # Convert ord('A') back to letters only when needed
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
