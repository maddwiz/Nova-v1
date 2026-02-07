import gzip
import re
from typing import Dict, List, Tuple

from usc.mem.varint import encode_uvarint, decode_uvarint
from usc.mem.dictpack import build_line_table


MAGIC = b"USCHB3"  # Unified State Codec HybridPack v0.3 (normalized templates)

# One-pass slot matcher:
# - variant=A (captures A)
# - any integer number (captures 123)
_SLOT_RE = re.compile(r"(variant=)([A-Za-z])|(\d+)")


def _split_lines_keepends(text: str) -> List[str]:
    return text.splitlines(keepends=True)


def _pack_string(s: str) -> bytes:
    b = s.encode("utf-8")
    return encode_uvarint(len(b)) + b


def _unpack_string(data: bytes, offset: int) -> Tuple[str, int]:
    n, off = decode_uvarint(data, offset)
    b = data[off : off + n]
    off += n
    return b.decode("utf-8"), off


def _pack_table(table: List[str]) -> bytes:
    out = bytearray()
    out += encode_uvarint(len(table))
    for item in table:
        out += _pack_string(item)
    return bytes(out)


def _unpack_table(data: bytes, offset: int) -> Tuple[List[str], int]:
    n, off = decode_uvarint(data, offset)
    out: List[str] = []
    for _ in range(n):
        s, off = _unpack_string(data, off)
        out.append(s)
    return out, off


def _extract_normalized_template(text: str) -> Tuple[str, List[int]]:
    """
    Normalized template extractor (FAST + GENERAL)

    - Replaces variant=X with variant={k} and stores ord(X)
    - Replaces ANY number with {k} and stores int(number)

    IMPORTANT:
    This returns a normalized template string that will match across logs
    even when numbers/variants change.
    """
    vals: List[int] = []
    out_parts: List[str] = []

    last = 0
    slot = 0

    for m in _SLOT_RE.finditer(text):
        start, end = m.span()
        out_parts.append(text[last:start])

        if m.group(1) is not None:
            # variant=X
            prefix = m.group(1)  # "variant="
            letter = m.group(2)  # "A"
            out_parts.append(f"{prefix}{{{slot}}}")
            vals.append(ord(letter))
            slot += 1
        else:
            # number
            num = int(m.group(3))
            out_parts.append(f"{{{slot}}}")
            vals.append(num)
            slot += 1

        last = end

    out_parts.append(text[last:])
    return "".join(out_parts), vals


def encode_chunks_hybrid(chunks: List[str]) -> bytes:
    """
    HybridPack v0.3:

    ✅ Build NORMALIZED templates (numbers/variants become slots)
    ✅ Keep templates that repeat (count >= 2)
    ✅ For each chunk:
       mode 0 = template_id + slot values
       mode 1 = fallback line-table indices (dictpack-style)
    ✅ gzip final binary blob
    """
    # 1) Extract normalized templates for each chunk
    extracted: List[Tuple[str, List[int]]] = []
    template_counts: Dict[str, int] = {}

    for ch in chunks:
        t, vals = _extract_normalized_template(ch)
        extracted.append((t, vals))
        template_counts[t] = template_counts.get(t, 0) + 1

    # 2) Keep repeating templates only
    kept_templates: List[str] = []
    kept_index: Dict[str, int] = {}

    for t, _vals in extracted:
        if template_counts.get(t, 0) >= 2:
            if t not in kept_index:
                kept_index[t] = len(kept_templates)
                kept_templates.append(t)

    # 3) Shared line table for fallback mode
    line_table, line_index = build_line_table(chunks)

    # 4) Encode stream
    out = bytearray()
    out += MAGIC

    # template table
    out += _pack_table(kept_templates)

    # line table
    out += _pack_table(line_table)

    # chunk count
    out += encode_uvarint(len(chunks))

    # chunk records
    for i, ch in enumerate(chunks):
        t, vals = extracted[i]

        if t in kept_index:
            # mode 0 = template
            out += b"\x00"
            tid = kept_index[t]
            out += encode_uvarint(tid)
            out += encode_uvarint(len(vals))
            for v in vals:
                out += encode_uvarint(v)
        else:
            # mode 1 = fallback to line-table
            out += b"\x01"
            lines = _split_lines_keepends(ch)
            out += encode_uvarint(len(lines))
            for ln in lines:
                out += encode_uvarint(line_index[ln])

    return gzip.compress(bytes(out), compresslevel=9)


def decode_chunks_hybrid(packet_bytes: bytes) -> List[str]:
    raw = gzip.decompress(packet_bytes)
    if not raw.startswith(MAGIC):
        raise ValueError("Not a HYBRIDPACK v0.3 packet")

    off = len(MAGIC)

    templates, off = _unpack_table(raw, off)
    line_table, off = _unpack_table(raw, off)

    n_chunks, off = decode_uvarint(raw, off)

    out_chunks: List[str] = []

    for _ in range(n_chunks):
        mode = raw[off]
        off += 1

        if mode == 0:
            tid, off = decode_uvarint(raw, off)
            nvals, off = decode_uvarint(raw, off)
            vals: List[int] = []
            for _ in range(nvals):
                v, off = decode_uvarint(raw, off)
                vals.append(v)

            template = templates[tid]

            # Convert ord values back to letters when template includes variant=
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

        elif mode == 1:
            nlines, off = decode_uvarint(raw, off)
            buf = []
            for _ in range(nlines):
                ix, off = decode_uvarint(raw, off)
                buf.append(line_table[ix])
            out_chunks.append("".join(buf))

        else:
            raise ValueError("Unknown hybrid chunk mode")

    return out_chunks
