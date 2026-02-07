import gzip
from typing import Dict, List, Tuple

from usc.mem.varint import encode_uvarint, decode_uvarint


MAGIC = b"USCDP2"  # Unified State Codec DictPack v0.2 (binary)


def _split_lines_keepends(text: str) -> List[str]:
    return text.splitlines(keepends=True)


def build_line_table(chunks: List[str]) -> Tuple[List[str], Dict[str, int]]:
    """
    Builds a shared table of unique lines across all chunks.
    Returns:
      table: list[index] -> line
      index: dict[line] -> index
    """
    table: List[str] = []
    index: Dict[str, int] = {}

    for chunk in chunks:
        for ln in _split_lines_keepends(chunk):
            if ln not in index:
                index[ln] = len(table)
                table.append(ln)

    return table, index


def _pack_table(table: List[str]) -> bytes:
    """
    Binary pack:
      [num_lines varint]
      repeated:
        [line_len varint][line_bytes]
    """
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


def _pack_chunks(chunks: List[str], idx: Dict[str, int]) -> bytes:
    """
    Binary pack:
      [num_chunks varint]
      repeated:
        [num_lines_in_chunk varint]
        [line_index varint]... (for each line)
    """
    out = bytearray()
    out += encode_uvarint(len(chunks))
    for ch in chunks:
        lines = _split_lines_keepends(ch)
        out += encode_uvarint(len(lines))
        for ln in lines:
            out += encode_uvarint(idx[ln])
    return bytes(out)


def _unpack_chunks(data: bytes, offset: int, table: List[str]) -> Tuple[List[str], int]:
    num_chunks, off = decode_uvarint(data, offset)
    out: List[str] = []
    for _ in range(num_chunks):
        n_lines, off = decode_uvarint(data, off)
        buf = []
        for _ in range(n_lines):
            ix, off = decode_uvarint(data, off)
            buf.append(table[ix])
        out.append("".join(buf))
    return out, off


def encode_chunks_with_table(chunks: List[str]) -> bytes:
    """
    Lossless shared-dictionary chunk pack (BINARY v0.2)

    Layout before gzip:
      MAGIC
      [table_blob]
      [chunks_blob]
    """
    table, idx = build_line_table(chunks)

    table_blob = _pack_table(table)
    chunks_blob = _pack_chunks(chunks, idx)

    raw = MAGIC + table_blob + chunks_blob
    return gzip.compress(raw, compresslevel=9)


def decode_chunks_with_table(packet_bytes: bytes) -> List[str]:
    raw = gzip.decompress(packet_bytes)

    if not raw.startswith(MAGIC):
        raise ValueError("Not a DICTPACK v0.2 packet")

    off = len(MAGIC)

    table, off = _unpack_table(raw, off)
    chunks, off = _unpack_chunks(raw, off, table)

    return chunks
