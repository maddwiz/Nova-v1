import gzip
from typing import Dict, List, Tuple

from usc.mem.varint import encode_uvarint, decode_uvarint
from usc.mem.dictpack import build_line_table


MAGIC = b"USCDL1"  # Unified State Codec DeltaPack v0.1 (binary)


def _split_lines_keepends(text: str) -> List[str]:
    return text.splitlines(keepends=True)


def _pack_table(table: List[str]) -> bytes:
    out = bytearray()
    out += encode_uvarint(len(table))
    for ln in table:
        b = ln.encode("utf-8")
        out += encode_uvarint(len(b))
        out += b
    return bytes(out)


def _unpack_table(data: bytes, offset: int) -> Tuple[List[str], int]:
    n, off = decode_uvarint(data, offset)
    table: List[str] = []
    for _ in range(n):
        ln_len, off = decode_uvarint(data, off)
        b = data[off : off + ln_len]
        off += ln_len
        table.append(b.decode("utf-8"))
    return table, off


def _encode_seq(seq: List[int]) -> bytes:
    out = bytearray()
    out += encode_uvarint(len(seq))
    for v in seq:
        out += encode_uvarint(v)
    return bytes(out)


def _decode_seq(data: bytes, offset: int) -> Tuple[List[int], int]:
    n, off = decode_uvarint(data, offset)
    seq: List[int] = []
    for _ in range(n):
        v, off = decode_uvarint(data, off)
        seq.append(v)
    return seq, off


def encode_chunks_with_line_deltas(chunks: List[str]) -> bytes:
    """
    DeltaPack:
    - Shared line table for all chunks
    - Chunk 0 stored full
    - Chunk i>0 stored as "diff vs previous" = changed positions only
    Then gzip compress final binary blob.

    This should beat DictPack when chunks are similar across time.
    """
    table, idx = build_line_table(chunks)

    # Convert chunks -> sequences of line indices
    seqs: List[List[int]] = []
    for ch in chunks:
        seqs.append([idx[ln] for ln in _split_lines_keepends(ch)])

    out = bytearray()
    out += MAGIC
    out += _pack_table(table)

    out += encode_uvarint(len(seqs))  # number of chunks

    # Store first chunk fully
    out += b"\x00"  # mode 0 = full
    out += _encode_seq(seqs[0])

    # Store subsequent chunks as deltas
    prev = seqs[0]
    for cur in seqs[1:]:
        out += b"\x01"  # mode 1 = delta

        # compare vs prev
        max_len = max(len(prev), len(cur))
        changes: List[Tuple[int, int]] = []

        for pos in range(max_len):
            prev_val = prev[pos] if pos < len(prev) else None
            cur_val = cur[pos] if pos < len(cur) else None
            if prev_val != cur_val:
                # if cur_val is None, treat as "truncate"
                # represent truncate by storing a special index = table_len (invalid index)
                if cur_val is None:
                    changes.append((pos, len(table)))  # truncate marker
                else:
                    changes.append((pos, cur_val))

        # write: new_len + num_changes + (pos, new_val)...
        out += encode_uvarint(len(cur))
        out += encode_uvarint(len(changes))

        for pos, new_val in changes:
            out += encode_uvarint(pos)
            out += encode_uvarint(new_val)

        prev = cur

    return gzip.compress(bytes(out), compresslevel=9)


def decode_chunks_with_line_deltas(packet_bytes: bytes) -> List[str]:
    raw = gzip.decompress(packet_bytes)

    if not raw.startswith(MAGIC):
        raise ValueError("Not a DELTAPACK v0.1 packet")

    off = len(MAGIC)
    table, off = _unpack_table(raw, off)

    num_chunks, off = decode_uvarint(raw, off)
    if num_chunks <= 0:
        return []

    chunks_out: List[List[int]] = []

    # Read first (must be full)
    mode = raw[off]
    off += 1
    if mode != 0:
        raise ValueError("Expected first chunk to be full")

    first, off = _decode_seq(raw, off)
    chunks_out.append(first)

    prev = first

    # Read deltas
    for _ in range(num_chunks - 1):
        mode = raw[off]
        off += 1
        if mode != 1:
            raise ValueError("Expected delta mode")

        new_len, off = decode_uvarint(raw, off)
        num_changes, off = decode_uvarint(raw, off)

        cur = prev[:new_len]  # start as truncated/kept portion

        # apply changes
        for _ in range(num_changes):
            pos, off = decode_uvarint(raw, off)
            new_val, off = decode_uvarint(raw, off)

            if new_val == len(table):
                # truncate marker: ignore
                continue

            # ensure length
            if pos >= len(cur):
                cur.extend([0] * (pos - len(cur) + 1))
            cur[pos] = new_val

        chunks_out.append(cur)
        prev = cur

    # convert indices -> text chunks
    text_chunks: List[str] = []
    for seq in chunks_out:
        text_chunks.append("".join(table[i] for i in seq if i < len(table)))

    return text_chunks
