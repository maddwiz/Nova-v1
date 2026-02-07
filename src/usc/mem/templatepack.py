import gzip
import re
from typing import List, Tuple

from usc.mem.varint import encode_uvarint, decode_uvarint


MAGIC = b"USCTM4"  # Unified State Codec TemplatePack v0.4


# One-pass matcher:
# 1) KEYED numbers like LOOP=12, ID=999, STEP=3, ROUND=4, EPOCH=5  -> ALWAYS slot
# 2) variant=A -> ALWAYS slot
# 3) any other number -> slot only if big enough
_SLOT_RE = re.compile(
    r"((?:LOOP|ID|STEP|ROUND|EPOCH)=)(\d+)"    # keyed numbers
    r"|(variant=)([A-Za-z])"                  # variant letter
    r"|(\d+)"                                 # any other number
)


def _should_slot_free_number(num_str: str) -> bool:
    """
    For numbers NOT tied to a key like LOOP=/ID=:
    ✅ slot big numbers (IDs, timestamps)
    ❌ keep tiny numbers inline (reduces overhead on VARIED logs)
    """
    # slot if 3+ digits OR numeric value >= 1000
    if len(num_str) >= 3:
        return True
    try:
        return int(num_str) >= 1000
    except Exception:
        return False


def _extract_template(text: str) -> Tuple[str, List[int]]:
    """
    v0.4 extractor (FAST + SMART)

    ✅ One pass only (never hangs)
    ✅ ALWAYS slot:
        - LOOP=number, ID=number, STEP=number, ROUND=number, EPOCH=number
        - variant=A/B/C
    ✅ Slot big free numbers anywhere else
    ❌ Keep tiny free numbers inline
    """
    vals: List[int] = []
    out_parts: List[str] = []

    last = 0
    slot = 0

    for m in _SLOT_RE.finditer(text):
        start, end = m.span()
        out_parts.append(text[last:start])

        if m.group(1) is not None:
            # keyed number: ALWAYS slot
            key = m.group(1)          # "LOOP=" / "ID=" / etc
            num_str = m.group(2)      # digits
            out_parts.append(f"{key}{{{slot}}}")
            vals.append(int(num_str))
            slot += 1

        elif m.group(3) is not None:
            # variant letter: ALWAYS slot
            prefix = m.group(3)       # "variant="
            letter = m.group(4)       # "A"
            out_parts.append(f"{prefix}{{{slot}}}")
            vals.append(ord(letter))
            slot += 1

        else:
            # free number
            num_str = m.group(5)
            if _should_slot_free_number(num_str):
                out_parts.append(f"{{{slot}}}")
                vals.append(int(num_str))
                slot += 1
            else:
                out_parts.append(num_str)

        last = end

    out_parts.append(text[last:])
    return "".join(out_parts), vals


def _pack_string(s: str) -> bytes:
    b = s.encode("utf-8")
    return encode_uvarint(len(b)) + b


def _unpack_string(data: bytes, offset: int) -> Tuple[str, int]:
    n, off = decode_uvarint(data, offset)
    b = data[off : off + n]
    off += n
    return b.decode("utf-8"), off


def encode_chunks_with_templates(chunks: List[str]) -> bytes:
    templates: List[str] = []
    temp_index = {}
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

    # chunks
    out += encode_uvarint(len(chunk_records))
    for tid, vals in chunk_records:
        out += encode_uvarint(tid)
        out += encode_uvarint(len(vals))
        for v in vals:
            out += encode_uvarint(v)

    return gzip.compress(bytes(out), compresslevel=9)


def decode_chunks_with_templates(packet_bytes: bytes) -> List[str]:
    raw = gzip.decompress(packet_bytes)
    if not raw.startswith(MAGIC):
        raise ValueError("Not a TEMPLATEPACK v0.4 packet")

    off = len(MAGIC)

    # templates
    ntemps, off = decode_uvarint(raw, off)
    templates: List[str] = []
    for _ in range(ntemps):
        s, off = _unpack_string(raw, off)
        templates.append(s)

    # chunks
    nchunks, off = decode_uvarint(raw, off)
    out_chunks: List[str] = []

    for _ in range(nchunks):
        tid, off = decode_uvarint(raw, off)
        nvals, off = decode_uvarint(raw, off)
        vals: List[int] = []
        for _ in range(nvals):
            v, off = decode_uvarint(raw, off)
            vals.append(v)

        template = templates[tid]

        # Convert ord('A') ints back into letters only when template includes variant=
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
